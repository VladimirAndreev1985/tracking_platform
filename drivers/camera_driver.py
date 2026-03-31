# -*- coding: utf-8 -*-
"""
Драйвер камеры Arducam 5MP PTZ (CSI MIPI).

Подключение: CSI шлейф → разъём CAM0 на Raspberry Pi 5.
Управление PTZ: I2C через Arducam Python SDK (libcamera_dev).

Бэкенды захвата (в порядке приоритета):
  1. Picamera2 (libcamera) — основной, нативный для RPi5 + CSI
     Минимальная задержка, полный контроль ISP, автофокус, HDR.
  2. OpenCV VideoCapture — fallback для разработки на ПК (USB-камера)

Управление зумом Arducam PTZ:
  Оптический зум управляется через I2C (Arducam Python SDK).
  При отсутствии SDK — цифровой зум через кроп кадра.

Потокобезопасный захват кадров в отдельном потоке.
"""

import time
import logging
import threading
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Попытка импорта бэкендов ──

# Приоритет 1: Picamera2 (libcamera, нативный для RPi5 CSI)
try:
    from picamera2 import Picamera2
    _HAS_PICAMERA2 = True
except ImportError:
    _HAS_PICAMERA2 = False

# Приоритет 2: OpenCV (fallback для USB-камер и разработки на ПК)
try:
    import cv2
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False

# Опционально: Arducam Python SDK для управления PTZ (I2C)
try:
    from Arducam import Arducam_PTZ
    _HAS_ARDUCAM_SDK = True
except ImportError:
    _HAS_ARDUCAM_SDK = False


class CameraDriver:
    """
    Драйвер камеры с автоматическим выбором бэкенда.

    Приоритет:
      1. Picamera2 (CSI, libcamera) — для Raspberry Pi 5
      2. OpenCV VideoCapture — fallback для разработки

    Кадры захватываются в фоновом потоке и доступны
    через get_frame() без блокировки основного цикла.
    """

    def __init__(self, device_index: int = 0,
                 width: int = 1920, height: int = 1080, fps: int = 30,
                 zoom_min: float = 1.0, zoom_max: float = 5.0,
                 zoom_step: float = 0.1, zoom_default: float = 1.0,
                 fov_h_deg: float = 62.2, fov_v_deg: float = 48.8):
        """
        Args:
            device_index: Индекс CSI-камеры (0 = CAM0, 1 = CAM1) или USB (/dev/videoN)
            width: Ширина кадра (пиксели)
            height: Высота кадра (пиксели)
            fps: Частота кадров
            zoom_min: Минимальный зум (1x)
            zoom_max: Максимальный зум
            zoom_step: Шаг изменения зума
            zoom_default: Зум по умолчанию
            fov_h_deg: Горизонтальное поле зрения при zoom=1x (градусы)
            fov_v_deg: Вертикальное поле зрения при zoom=1x (градусы)
        """
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps

        # Параметры зума
        self._zoom_min = zoom_min
        self._zoom_max = zoom_max
        self._zoom_step = zoom_step
        self._zoom_current = zoom_default

        # Поле зрения (при zoom=1x)
        self._fov_h_base = fov_h_deg
        self._fov_v_base = fov_v_deg

        # Бэкенд захвата
        self._backend: str = "none"       # "picamera2", "opencv", "none"
        self._picam: Optional['Picamera2'] = None
        self._cv_cap = None               # cv2.VideoCapture
        self._arducam_ptz = None          # Arducam PTZ controller (I2C)

        # Кадр и потокобезопасность
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_id: int = 0
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False

        # Цифровой зум (кроп) — используется если нет аппаратного зума
        self._digital_zoom = False
        self._sensor_width = width
        self._sensor_height = height

        # Статистика
        self._fps_actual: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = 0.0

        logger.info(
            f"CameraDriver: камера={device_index}, "
            f"{width}x{height}@{fps}, зум={zoom_default:.1f}x, "
            f"Picamera2={'ДА' if _HAS_PICAMERA2 else 'НЕТ'}, "
            f"OpenCV={'ДА' if _HAS_OPENCV else 'НЕТ'}, "
            f"Arducam SDK={'ДА' if _HAS_ARDUCAM_SDK else 'НЕТ'}"
        )

    # ════════════════════════════════════════════════════════
    # Свойства
    # ════════════════════════════════════════════════════════

    @property
    def zoom(self) -> float:
        """Текущий уровень зума."""
        return self._zoom_current

    @property
    def fov_h(self) -> float:
        """Текущее горизонтальное поле зрения (градусы) с учётом зума."""
        return self._fov_h_base / self._zoom_current

    @property
    def fov_v(self) -> float:
        """Текущее вертикальное поле зрения (градусы) с учётом зума."""
        return self._fov_v_base / self._zoom_current

    @property
    def resolution(self) -> Tuple[int, int]:
        """Разрешение выходного кадра (ширина, высота)."""
        return self._width, self._height

    @property
    def fps_actual(self) -> float:
        """Фактическая частота кадров."""
        return self._fps_actual

    @property
    def frame_id(self) -> int:
        """Номер текущего кадра."""
        return self._frame_id

    @property
    def is_opened(self) -> bool:
        """Камера открыта и поток захвата работает."""
        return self._running

    @property
    def backend(self) -> str:
        """Активный бэкенд: 'picamera2', 'opencv' или 'none'."""
        return self._backend

    # ════════════════════════════════════════════════════════
    # Инициализация
    # ════════════════════════════════════════════════════════

    def open(self) -> bool:
        """
        Открыть камеру и запустить фоновый поток захвата.

        Автоматически выбирает лучший доступный бэкенд:
          1. Picamera2 (CSI, libcamera) — для RPi5
          2. OpenCV (V4L2/USB) — fallback

        Returns:
            True при успехе
        """
        # Попытка 1: Picamera2 (CSI, libcamera)
        if _HAS_PICAMERA2:
            if self._open_picamera2():
                return True
            logger.warning("Picamera2 не удалось открыть, пробую OpenCV...")

        # Попытка 2: OpenCV (fallback)
        if _HAS_OPENCV:
            if self._open_opencv():
                return True

        logger.error("Не удалось открыть камеру ни одним бэкендом!")
        return False

    def _open_picamera2(self) -> bool:
        """Открыть камеру через Picamera2 (libcamera, CSI)."""
        try:
            self._picam = Picamera2(camera_num=self._device_index)

            # Конфигурация для видеозахвата
            # main: полное разрешение для детекции и HUD
            # lores: уменьшенное для предпросмотра (не используем)
            config = self._picam.create_video_configuration(
                main={
                    "size": (self._width, self._height),
                    "format": "BGR888",  # BGR для совместимости с OpenCV/YOLO
                },
                controls={
                    "FrameRate": self._fps,
                    # Фиксированная экспозиция для стабильности (можно настроить)
                    # "ExposureTime": 10000,  # мкс
                    # "AnalogueGain": 2.0,
                },
                buffer_count=2,  # Минимальный буфер для низкой задержки
            )

            self._picam.configure(config)
            self._picam.start()

            # Даём камере время на инициализацию ISP и автоэкспозицию
            time.sleep(0.5)

            # Получаем реальные параметры сенсора
            camera_props = self._picam.camera_properties
            sensor_res = camera_props.get('PixelArraySize', (self._width, self._height))
            self._sensor_width = sensor_res[0]
            self._sensor_height = sensor_res[1]

            self._backend = "picamera2"
            logger.info(
                f"Камера открыта [Picamera2/libcamera CSI]: "
                f"{self._width}x{self._height}@{self._fps}, "
                f"сенсор={self._sensor_width}x{self._sensor_height}"
            )

            # Инициализация Arducam PTZ (I2C) если доступен
            self._init_arducam_ptz()

            # Запуск фонового потока захвата
            self._running = True
            self._fps_timer = time.time()
            self._capture_thread = threading.Thread(
                target=self._capture_loop_picamera2,
                name="CameraCapture-Picamera2",
                daemon=True
            )
            self._capture_thread.start()

            logger.info("Поток захвата камеры запущен [Picamera2]")
            return True

        except Exception as e:
            logger.error(f"Ошибка открытия Picamera2: {e}")
            if self._picam:
                try:
                    self._picam.close()
                except Exception:
                    pass
                self._picam = None
            return False

    def _open_opencv(self) -> bool:
        """Открыть камеру через OpenCV (fallback для USB-камер и разработки)."""
        try:
            import cv2 as cv

            # Попытка через V4L2 (Linux)
            self._cv_cap = cv.VideoCapture(self._device_index, cv.CAP_V4L2)
            if not self._cv_cap.isOpened():
                # Fallback без указания бэкенда
                self._cv_cap = cv.VideoCapture(self._device_index)

            if not self._cv_cap.isOpened():
                logger.error(f"OpenCV: не удалось открыть камеру {self._device_index}")
                return False

            # Установка параметров
            self._cv_cap.set(cv.CAP_PROP_FRAME_WIDTH, self._width)
            self._cv_cap.set(cv.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cv_cap.set(cv.CAP_PROP_FPS, self._fps)
            self._cv_cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

            # Проверка реальных параметров
            actual_w = int(self._cv_cap.get(cv.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cv_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cv_cap.get(cv.CAP_PROP_FPS)

            if actual_w != self._width or actual_h != self._height:
                logger.warning(
                    f"Запрошено {self._width}x{self._height}, "
                    f"получено {actual_w}x{actual_h}"
                )
                self._width = actual_w
                self._height = actual_h

            self._sensor_width = self._width
            self._sensor_height = self._height
            self._backend = "opencv"
            self._digital_zoom = True  # OpenCV fallback — только цифровой зум

            logger.info(
                f"Камера открыта [OpenCV fallback]: "
                f"{actual_w}x{actual_h}@{actual_fps:.0f}"
            )

            # Запуск фонового потока захвата
            self._running = True
            self._fps_timer = time.time()
            self._capture_thread = threading.Thread(
                target=self._capture_loop_opencv,
                name="CameraCapture-OpenCV",
                daemon=True
            )
            self._capture_thread.start()

            logger.info("Поток захвата камеры запущен [OpenCV]")
            return True

        except Exception as e:
            logger.error(f"Ошибка открытия OpenCV: {e}")
            return False

    def _init_arducam_ptz(self):
        """Инициализировать Arducam PTZ контроллер (I2C) для оптического зума."""
        if _HAS_ARDUCAM_SDK:
            try:
                self._arducam_ptz = Arducam_PTZ()
                self._arducam_ptz.init()
                self._digital_zoom = False
                logger.info("Arducam PTZ (I2C): оптический зум доступен")
                # Установить начальный зум
                self._set_hardware_zoom(self._zoom_current)
            except Exception as e:
                logger.warning(f"Arducam PTZ SDK недоступен: {e}")
                self._arducam_ptz = None
                self._digital_zoom = True
        else:
            self._digital_zoom = True
            logger.info(
                "Arducam SDK не установлен — используется цифровой зум (кроп). "
                "Для оптического зума: pip install Arducam-PTZ"
            )

    # ════════════════════════════════════════════════════════
    # Завершение
    # ════════════════════════════════════════════════════════

    def close(self):
        """Остановить захват и закрыть камеру."""
        self._running = False

        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)

        if self._picam:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None

        if self._cv_cap:
            self._cv_cap.release()
            self._cv_cap = None

        self._backend = "none"
        logger.info("Камера закрыта")

    # ════════════════════════════════════════════════════════
    # Захват кадров
    # ════════════════════════════════════════════════════════

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Получить последний захваченный кадр (потокобезопасно).

        Returns:
            numpy-массив BGR-кадра или None
        """
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def get_frame_no_copy(self) -> Optional[np.ndarray]:
        """
        Получить ссылку на последний кадр БЕЗ копирования.
        Быстрее, но не потокобезопасно.
        """
        return self._frame

    def _capture_loop_picamera2(self):
        """Фоновый поток захвата через Picamera2."""
        logger.debug("Поток захвата [Picamera2]: старт")

        while self._running and self._picam:
            try:
                # capture_array возвращает BGR888 numpy array
                frame = self._picam.capture_array("main")

                if frame is not None:
                    with self._frame_lock:
                        self._frame = frame
                        self._frame_id += 1
                    self._update_fps()

            except Exception as e:
                logger.error(f"Ошибка захвата Picamera2: {e}")
                time.sleep(0.01)

        logger.debug("Поток захвата [Picamera2]: стоп")

    def _capture_loop_opencv(self):
        """Фоновый поток захвата через OpenCV."""
        logger.debug("Поток захвата [OpenCV]: старт")

        while self._running and self._cv_cap and self._cv_cap.isOpened():
            ret, frame = self._cv_cap.read()

            if ret and frame is not None:
                with self._frame_lock:
                    self._frame = frame
                    self._frame_id += 1
                self._update_fps()
            else:
                time.sleep(0.001)

        logger.debug("Поток захвата [OpenCV]: стоп")

    def _update_fps(self):
        """Обновить счётчик FPS."""
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._fps_timer
        if elapsed >= 1.0:
            self._fps_actual = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = now

    # ════════════════════════════════════════════════════════
    # Управление зумом
    # ════════════════════════════════════════════════════════

    def zoom_in(self):
        """Увеличить зум на один шаг."""
        new_zoom = min(self._zoom_current + self._zoom_step, self._zoom_max)
        if new_zoom != self._zoom_current:
            self._zoom_current = new_zoom
            self._set_hardware_zoom(self._zoom_current)
            logger.debug(f"Зум: {self._zoom_current:.1f}x")

    def zoom_out(self):
        """Уменьшить зум на один шаг."""
        new_zoom = max(self._zoom_current - self._zoom_step, self._zoom_min)
        if new_zoom != self._zoom_current:
            self._zoom_current = new_zoom
            self._set_hardware_zoom(self._zoom_current)
            logger.debug(f"Зум: {self._zoom_current:.1f}x")

    def set_zoom(self, level: float):
        """Установить конкретный уровень зума."""
        self._zoom_current = max(self._zoom_min, min(level, self._zoom_max))
        self._set_hardware_zoom(self._zoom_current)

    def _set_hardware_zoom(self, level: float):
        """
        Установить зум на аппаратном уровне.

        Приоритет:
          1. Arducam PTZ SDK (I2C) — оптический зум
          2. Picamera2 ScalerCrop — цифровой зум через ISP
          3. Ничего (зум учитывается только в FOV для баллистики)
        """
        # Способ 1: Arducam PTZ SDK (оптический зум через I2C)
        if self._arducam_ptz is not None:
            try:
                # Arducam PTZ принимает значение зума 100-500 (1x-5x)
                hw_zoom = int(level * 100)
                self._arducam_ptz.zoom(hw_zoom)
                return
            except Exception as e:
                logger.warning(f"Arducam PTZ zoom ошибка: {e}")

        # Способ 2: Picamera2 ScalerCrop (цифровой зум через ISP RPi)
        if self._picam is not None and level > 1.0:
            try:
                # ScalerCrop: кроп из центра сенсора
                # При zoom=2x берём центральную 1/2 сенсора
                crop_w = int(self._sensor_width / level)
                crop_h = int(self._sensor_height / level)
                offset_x = (self._sensor_width - crop_w) // 2
                offset_y = (self._sensor_height - crop_h) // 2

                self._picam.set_controls({
                    "ScalerCrop": (offset_x, offset_y, crop_w, crop_h)
                })
                return
            except Exception as e:
                logger.debug(f"Picamera2 ScalerCrop: {e}")

    # ════════════════════════════════════════════════════════
    # Утилиты: пиксели ↔ углы
    # ════════════════════════════════════════════════════════

    def pixel_to_angle(self, px: float, py: float) -> Tuple[float, float]:
        """
        Преобразовать координаты пикселя в угловое смещение
        от центра кадра (в градусах).
        """
        cx = self._width / 2.0
        cy = self._height / 2.0
        dx = px - cx
        dy = py - cy
        angle_h = (dx / self._width) * self.fov_h
        angle_v = (dy / self._height) * self.fov_v
        return angle_h, angle_v

    def angle_to_pixel(self, angle_h: float, angle_v: float) -> Tuple[float, float]:
        """
        Преобразовать угловое смещение (градусы) в координаты пикселя.
        """
        cx = self._width / 2.0
        cy = self._height / 2.0
        px = cx + (angle_h / self.fov_h) * self._width
        py = cy + (angle_v / self.fov_v) * self._height
        return px, py
