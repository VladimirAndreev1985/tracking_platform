# -*- coding: utf-8 -*-
"""
Детектор объектов на базе YOLO26 (Ultralytics, март 2026).

Поддерживает два бэкенда:
1. Hailo NPU (AI HAT+ с Hailo-10H) — основной, для продакшена
   Модель конвертируется в .hef через Hailo DFC
2. Ultralytics YOLO26 (CPU) — fallback для разработки и тестирования
   Работает локально на RPi5 через pip install ultralytics

Для детекции дронов рекомендуется дообучить модель на датасетах:
- Anti-UAV (https://github.com/wangdongdut/Anti-UAV)
- DroneDetect (Roboflow)
- DUT Anti-UAV Dataset

Детектор принимает BGR-кадр и возвращает список детекций
в формате (x1, y1, x2, y2, confidence, class_id).
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Результат детекции
# ════════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """Одна детекция объекта."""
    x1: float                 # Левый верхний угол X
    y1: float                 # Левый верхний угол Y
    x2: float                 # Правый нижний угол X
    y2: float                 # Правый нижний угол Y
    confidence: float         # Уверенность (0.0 .. 1.0)
    class_id: int             # ID класса COCO

    @property
    def center(self) -> tuple:
        """Центр bounding box (cx, cy)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_tlwh(self) -> tuple:
        """Формат (top, left, width, height) для трекера."""
        return (self.x1, self.y1, self.width, self.height)

    def to_xyxy(self) -> tuple:
        """Формат (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)


# ════════════════════════════════════════════════════════════════
# Имена классов COCO (подмножество для воздушных целей)
# ════════════════════════════════════════════════════════════════

COCO_NAMES = {
    0: "человек",
    1: "велосипед",
    2: "автомобиль",
    3: "мотоцикл",
    4: "самолёт",
    5: "автобус",
    6: "поезд",
    7: "грузовик",
    8: "лодка",
    14: "птица",
    15: "кот",
    16: "собака",
    24: "рюкзак",
    25: "зонт",
    39: "бутылка",
    63: "ноутбук",
    67: "телефон",
}


def get_class_name(class_id: int) -> str:
    """Получить русское название класса COCO."""
    return COCO_NAMES.get(class_id, f"класс_{class_id}")


# ════════════════════════════════════════════════════════════════
# Детектор YOLO
# ════════════════════════════════════════════════════════════════

class YOLODetector:
    """
    Детектор объектов YOLO с поддержкой Hailo NPU и Ultralytics.

    Выбор бэкенда:
    - use_hailo=True: загрузка .hef модели для Hailo-10H
    - use_hailo=False: загрузка .pt/.onnx через Ultralytics
    """

    def __init__(self, model_path: str = "models/yolo26n.pt",
                 confidence_threshold: float = 0.45,
                 nms_threshold: float = 0.5,
                 target_classes: List[int] = None,
                 input_size: int = 640,
                 use_hailo: bool = True):
        """
        Args:
            model_path: Путь к файлу модели (.hef для Hailo, .pt для Ultralytics)
            confidence_threshold: Порог уверенности
            nms_threshold: Порог NMS
            target_classes: Список ID классов для детекции (None = все)
            input_size: Размер входного изображения
            use_hailo: Использовать Hailo NPU
        """
        self._model_path = model_path
        self._conf_threshold = confidence_threshold
        self._nms_threshold = nms_threshold
        self._target_classes = target_classes or [4]  # По умолчанию: самолёт
        self._input_size = input_size
        self._use_hailo = use_hailo

        self._model = None
        self._initialized = False

        # Статистика
        self._inference_time_ms = 0.0
        self._detection_count = 0

        logger.info(
            f"YOLODetector: модель={model_path}, "
            f"порог={confidence_threshold}, "
            f"классы={target_classes}, "
            f"Hailo={'ДА' if use_hailo else 'НЕТ'}"
        )

    @property
    def inference_time_ms(self) -> float:
        """Время последнего инференса (мс)."""
        return self._inference_time_ms

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ── Инициализация ───────────────────────────────────────

    def initialize(self) -> bool:
        """
        Загрузить модель. Возвращает True при успехе.
        """
        try:
            if self._use_hailo:
                return self._init_hailo()
            else:
                return self._init_ultralytics()
        except Exception as e:
            logger.error(f"Ошибка инициализации детектора: {e}")
            return False

    def _init_hailo(self) -> bool:
        """Инициализация через Hailo Runtime."""
        try:
            from hailo_platform import (
                HEF, VDevice, HailoStreamInterface,
                InferVStreams, ConfigureParams,
                InputVStreamParams, OutputVStreamParams,
                FormatType
            )

            logger.info("Загрузка модели на Hailo NPU...")

            # Загрузка HEF-файла
            self._hef = HEF(self._model_path)

            # Создание виртуального устройства
            self._vdevice = VDevice()

            # Конфигурация
            configure_params = ConfigureParams.create_from_hef(
                self._hef, interface=HailoStreamInterface.PCIe
            )
            self._network_group = self._vdevice.configure(
                self._hef, configure_params
            )[0]

            # Параметры потоков ввода/вывода
            self._input_vstream_params = InputVStreamParams.make_from_network_group(
                self._network_group, quantized=False,
                format_type=FormatType.FLOAT32
            )
            self._output_vstream_params = OutputVStreamParams.make_from_network_group(
                self._network_group, quantized=False,
                format_type=FormatType.FLOAT32
            )

            # Получение информации о входе/выходе
            input_vstreams_info = self._hef.get_input_vstream_infos()
            output_vstreams_info = self._hef.get_output_vstream_infos()

            self._input_shape = input_vstreams_info[0].shape
            logger.info(f"  Вход модели: {self._input_shape}")
            logger.info(f"  Выходов: {len(output_vstreams_info)}")

            self._initialized = True
            logger.info("Hailo NPU: модель загружена успешно")
            return True

        except ImportError:
            logger.warning(
                "Hailo SDK не установлен. Переключаюсь на Ultralytics..."
            )
            return self._init_ultralytics()
        except Exception as e:
            logger.error(f"Ошибка инициализации Hailo: {e}")
            return False

    def _init_ultralytics(self) -> bool:
        """Инициализация через Ultralytics YOLO26 (fallback, локально на RPi5)."""
        try:
            from ultralytics import YOLO

            logger.info("Загрузка модели YOLO26 через Ultralytics...")

            # Если путь к .hef — подменяем на .pt
            model_path = self._model_path
            if model_path.endswith('.hef'):
                model_path = model_path.replace('.hef', '.pt')
                logger.info(f"  HEF → PT fallback: {model_path}")

            self._model = YOLO(model_path)
            self._use_hailo = False
            self._initialized = True

            logger.info(f"Ultralytics YOLO26: модель загружена ({model_path})")
            return True

        except ImportError:
            logger.error("Ни Hailo SDK, ни Ultralytics не установлены!")
            logger.error("Установите: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Ошибка загрузки модели YOLO26: {e}")
            return False

    def shutdown(self):
        """Освободить ресурсы."""
        self._model = None
        self._initialized = False
        logger.info("Детектор выключен")

    # ── Детекция ────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Выполнить детекцию объектов на кадре.

        Args:
            frame: BGR-кадр (numpy array)

        Returns:
            Список детекций Detection
        """
        if not self._initialized or frame is None:
            return []

        start = time.time()

        if self._use_hailo:
            detections = self._detect_hailo(frame)
        else:
            detections = self._detect_ultralytics(frame)

        self._inference_time_ms = (time.time() - start) * 1000
        self._detection_count = len(detections)

        return detections

    def _detect_hailo(self, frame: np.ndarray) -> List[Detection]:
        """Детекция через Hailo NPU."""
        try:
            from hailo_platform import InferVStreams

            # Предобработка: resize + нормализация
            input_h, input_w = self._input_shape[1], self._input_shape[2]
            orig_h, orig_w = frame.shape[:2]

            resized = cv2.resize(frame, (input_w, input_h))
            input_data = resized.astype(np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

            # Инференс
            with InferVStreams(
                self._network_group,
                self._input_vstream_params,
                self._output_vstream_params
            ) as infer_pipeline:
                input_dict = {
                    self._hef.get_input_vstream_infos()[0].name: input_data
                }
                output = infer_pipeline.infer(input_dict)

            # Постобработка (зависит от формата выхода модели)
            detections = self._postprocess_yolo(
                output, orig_w, orig_h, input_w, input_h
            )

            return detections

        except Exception as e:
            logger.error(f"Ошибка детекции Hailo: {e}")
            return []

    def _detect_ultralytics(self, frame: np.ndarray) -> List[Detection]:
        """Детекция через Ultralytics YOLO."""
        try:
            results = self._model(
                frame,
                conf=self._conf_threshold,
                iou=self._nms_threshold,
                classes=self._target_classes,
                verbose=False
            )

            detections = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    detections.append(Detection(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=conf,
                        class_id=cls_id
                    ))

            return detections

        except Exception as e:
            logger.error(f"Ошибка детекции Ultralytics: {e}")
            return []

    def _postprocess_yolo(self, raw_output: dict,
                          orig_w: int, orig_h: int,
                          input_w: int, input_h: int) -> List[Detection]:
        """
        Постобработка сырого выхода YOLO (для Hailo).

        Формат выхода зависит от конкретной модели.
        Здесь реализован типичный формат YOLOv8.
        """
        detections = []

        try:
            # Получаем первый выходной тензор
            output_name = list(raw_output.keys())[0]
            output_data = raw_output[output_name]

            if len(output_data.shape) == 3:
                output_data = output_data[0]  # Убираем batch dimension

            # YOLOv8 формат: (8400, 84) — 8400 якорей, 4 bbox + 80 классов
            num_detections = output_data.shape[0]
            num_classes = output_data.shape[1] - 4

            scale_x = orig_w / input_w
            scale_y = orig_h / input_h

            for i in range(num_detections):
                # Координаты bbox (center x, center y, width, height)
                cx, cy, w, h = output_data[i, :4]

                # Уверенности по классам
                class_scores = output_data[i, 4:]
                class_id = int(np.argmax(class_scores))
                confidence = float(class_scores[class_id])

                if confidence < self._conf_threshold:
                    continue

                if self._target_classes and class_id not in self._target_classes:
                    continue

                # Преобразование в (x1, y1, x2, y2) и масштабирование
                x1 = (cx - w / 2) * scale_x
                y1 = (cy - h / 2) * scale_y
                x2 = (cx + w / 2) * scale_x
                y2 = (cy + h / 2) * scale_y

                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=confidence,
                    class_id=class_id
                ))

            # NMS
            if len(detections) > 1:
                detections = self._nms(detections)

        except Exception as e:
            logger.error(f"Ошибка постобработки: {e}")

        return detections

    def _nms(self, detections: List[Detection]) -> List[Detection]:
        """Non-Maximum Suppression."""
        if not detections:
            return []

        boxes = np.array([d.to_xyxy() for d in detections])
        scores = np.array([d.confidence for d in detections])

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=self._conf_threshold,
            nms_threshold=self._nms_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]

        return detections
