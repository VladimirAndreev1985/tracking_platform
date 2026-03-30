# -*- coding: utf-8 -*-
"""
Драйвер камеры Arducam 5MP 1080p Pan-Tilt-Zoom.

Захват видеопотока через OpenCV (V4L2) и управление
оптическим зумом камеры. Камера имеет собственные
сервоприводы для pan/tilt, но мы используем их только
для fine-коррекции — основное наведение через Cubemars.

Потокобезопасный захват кадров в отдельном потоке
для минимизации задержки.
"""

import time
import logging
import threading
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class CameraDriver:
    """
    Драйвер камеры с потоковым захватом и управлением зумом.

    Кадры захватываются в фоновом потоке и доступны
    через метод get_frame() без блокировки основного цикла.
    """

    def __init__(self, device_index: int = 0,
                 width: int = 1920, height: int = 1080, fps: int = 30,
                 zoom_min: float = 1.0, zoom_max: float = 5.0,
                 zoom_step: float = 0.1, zoom_default: float = 1.0,
                 fov_h_deg: float = 62.2, fov_v_deg: float = 48.8):
        """
        Args:
            device_index: Индекс устройства V4L2 (/dev/videoN)
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

        # Захват видео
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._frame_id: int = 0
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False

        # Статистика
        self._fps_actual: float = 0.0
        self._frame_count: int = 0
        self._fps_timer: float = 0.0

        logger.info(
            f"CameraDriver: устройство={device_index}, "
            f"{width}x{height}@{fps}, зум={zoom_default:.1f}x"
        )

    # ── Свойства ────────────────────────────────────────────

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
        """Разрешение кадра (ширина, высота)."""
        return self._width, self._height

    @property
    def fps_actual(self) -> float:
        """Фактическая частота кадров."""
        return self._fps_actual

    @property
    def frame_id(self) -> int:
        """Номер текущего кадра (инкрементируется при каждом новом кадре)."""
        return self._frame_id

    @property
    def is_opened(self) -> bool:
        """Камера открыта и поток захвата работает."""
        return self._running and self._cap is not None and self._cap.isOpened()

    # ── Инициализация / завершение ──────────────────────────

    def open(self) -> bool:
        """
        Открыть камеру и запустить фоновый поток захвата.
        Возвращает True при успехе.
        """
        if cv2 is None:
            logger.error("Библиотека OpenCV (cv2) не установлена!")
            return False

        try:
            # Открытие камеры через V4L2 (Linux) или DirectShow (Windows)
            self._cap = cv2.VideoCapture(self._device_index, cv2.CAP_V4L2)

            if not self._cap.isOpened():
                # Fallback без указания бэкенда
                self._cap = cv2.VideoCapture(self._device_index)

            if not self._cap.isOpened():
                logger.error(f"Не удалось открыть камеру {self._device_index}")
                return False

            # Установка параметров
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            # Минимальный буфер для снижения задержки
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Проверка реальных параметров
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"Камера открыта: {actual_w}x{actual_h}@{actual_fps:.0f}")

            if actual_w != self._width or actual_h != self._height:
                logger.warning(
                    f"Запрошено {self._width}x{self._height}, "
                    f"получено {actual_w}x{actual_h}"
                )
                self._width = actual_w
                self._height = actual_h

            # Установка начального зума
            self._set_hardware_zoom(self._zoom_current)

            # Запуск фонового потока захвата
            self._running = True
            self._fps_timer = time.time()
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name="CameraCapture",
                daemon=True
            )
            self._capture_thread.start()

            logger.info("Поток захвата камеры запущен")
            return True

        except Exception as e:
            logger.error(f"Ошибка открытия камеры: {e}")
            return False

    def close(self):
        """Остановить захват и закрыть камеру."""
        self._running = False
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Камера закрыта")

    # ── Захват кадров ───────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Получить последний захваченный кадр (потокобезопасно).

        Returns:
            numpy-массив BGR-кадра или None, если кадр недоступен
        """
        with self._frame_lock:
            if self._frame is not None:
                return self._frame.copy()
        return None

    def get_frame_no_copy(self) -> Optional[np.ndarray]:
        """
        Получить ссылку на последний кадр БЕЗ копирования.
        Быстрее, но не потокобезопасно — использовать только
        если кадр не будет модифицирован.
        """
        return self._frame

    def _capture_loop(self):
        """Фоновый поток непрерывного захвата кадров."""
        logger.debug("Поток захвата камеры: старт")

        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()

            if ret and frame is not None:
                with self._frame_lock:
                    self._frame = frame
                    self._frame_id += 1

                # Подсчёт FPS
                self._frame_count += 1
                now = time.time()
                elapsed = now - self._fps_timer
                if elapsed >= 1.0:
                    self._fps_actual = self._frame_count / elapsed
                    self._frame_count = 0
                    self._fps_timer = now
            else:
                # Ошибка чтения — небольшая пауза
                time.sleep(0.001)

        logger.debug("Поток захвата камеры: стоп")

    # ── Управление зумом ────────────────────────────────────

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
        """
        Установить конкретный уровень зума.

        Args:
            level: Уровень зума (zoom_min .. zoom_max)
        """
        self._zoom_current = max(self._zoom_min, min(level, self._zoom_max))
        self._set_hardware_zoom(self._zoom_current)

    def _set_hardware_zoom(self, level: float):
        """
        Отправить команду зума на аппаратуру камеры.

        Arducam PTZ поддерживает управление зумом через:
        1. V4L2 CAP_PROP_ZOOM (если поддерживается драйвером)
        2. I2C команды напрямую
        3. Arducam Python SDK

        TODO: Реализовать конкретный протокол для вашей модели Arducam.
        """
        if self._cap is not None:
            try:
                # Попытка через стандартный V4L2
                # Масштабируем zoom level в диапазон, понятный камере
                hw_zoom = int((level - self._zoom_min) /
                              (self._zoom_max - self._zoom_min) * 100)
                self._cap.set(cv2.CAP_PROP_ZOOM, hw_zoom)
            except Exception:
                pass  # Не все камеры поддерживают программный зум

    # ── Утилиты ─────────────────────────────────────────────

    def pixel_to_angle(self, px: float, py: float) -> Tuple[float, float]:
        """
        Преобразовать координаты пикселя в угловое смещение
        от центра кадра (в градусах).

        Args:
            px: X-координата пикселя
            py: Y-координата пикселя

        Returns:
            (angle_h, angle_v) — угловое смещение в градусах
        """
        cx = self._width / 2.0
        cy = self._height / 2.0

        # Смещение от центра в пикселях
        dx = px - cx
        dy = py - cy

        # Преобразование в градусы через текущее поле зрения
        angle_h = (dx / self._width) * self.fov_h
        angle_v = (dy / self._height) * self.fov_v

        return angle_h, angle_v

    def angle_to_pixel(self, angle_h: float, angle_v: float) -> Tuple[float, float]:
        """
        Преобразовать угловое смещение (градусы) в координаты пикселя.

        Args:
            angle_h: Горизонтальный угол от центра (градусы)
            angle_v: Вертикальный угол от центра (градусы)

        Returns:
            (px, py) — координаты пикселя
        """
        cx = self._width / 2.0
        cy = self._height / 2.0

        px = cx + (angle_h / self.fov_h) * self._width
        py = cy + (angle_v / self.fov_v) * self._height

        return px, py
