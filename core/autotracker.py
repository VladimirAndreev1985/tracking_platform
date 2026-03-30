# -*- coding: utf-8 -*-
"""
Модуль автоматического слежения за целью.

Использует PID-регулятор для удержания обнаруженной цели
в центре кадра камеры. Вычисляет угловые скорости для
моторов на основе отклонения центра цели от центра кадра.

Алгоритм:
1. Получить bbox цели из трекера
2. Вычислить смещение центра цели от центра кадра (в пикселях)
3. Преобразовать смещение в угловую ошибку (градусы)
4. PID-регулятор вычисляет требуемую угловую скорость
5. Отправить скорость на моторы
"""

import time
import math
import logging
from typing import Optional, Tuple

from config.app_config import PIDConfig

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# PID-регулятор
# ════════════════════════════════════════════════════════════════

class PIDController:
    """
    Классический PID-регулятор с ограничением интегральной составляющей
    (anti-windup) и фильтрацией дифференциальной составляющей.
    """

    def __init__(self, kp: float = 0.5, ki: float = 0.05, kd: float = 0.15,
                 output_limit: float = 1.0, integral_limit: float = 50.0):
        """
        Args:
            kp: Пропорциональный коэффициент
            ki: Интегральный коэффициент
            kd: Дифференциальный коэффициент
            output_limit: Ограничение выходного сигнала (±)
            integral_limit: Ограничение интегральной суммы (anti-windup)
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._output_limit = output_limit
        self._integral_limit = integral_limit

        # Внутреннее состояние
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = 0.0
        self._initialized = False

        # Фильтр дифференциальной составляющей (low-pass)
        self._d_filter = 0.0
        self._d_filter_alpha = 0.2  # Коэффициент фильтрации (0..1)

    def reset(self):
        """Сбросить внутреннее состояние PID."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = 0.0
        self._d_filter = 0.0
        self._initialized = False

    def update(self, error: float, dt: float = None) -> float:
        """
        Вычислить управляющий сигнал на основе ошибки.

        Args:
            error: Текущая ошибка (цель - факт)
            dt: Шаг времени (сек). Если None — вычисляется автоматически.

        Returns:
            Управляющий сигнал (ограниченный ±output_limit)
        """
        now = time.time()

        if not self._initialized:
            self._prev_error = error
            self._prev_time = now
            self._initialized = True
            return 0.0

        # Вычисление dt
        if dt is None:
            dt = now - self._prev_time
        if dt <= 0:
            dt = 0.001  # Защита от деления на ноль

        # P — пропорциональная составляющая
        p_term = self.kp * error

        # I — интегральная составляющая с anti-windup
        self._integral += error * dt
        self._integral = max(-self._integral_limit,
                             min(self._integral_limit, self._integral))
        i_term = self.ki * self._integral

        # D — дифференциальная составляющая с фильтрацией
        raw_derivative = (error - self._prev_error) / dt
        self._d_filter += self._d_filter_alpha * (raw_derivative - self._d_filter)
        d_term = self.kd * self._d_filter

        # Суммарный выход
        output = p_term + i_term + d_term

        # Ограничение выхода
        output = max(-self._output_limit, min(self._output_limit, output))

        # Сохранение состояния
        self._prev_error = error
        self._prev_time = now

        return output


# ════════════════════════════════════════════════════════════════
# Автотрекер
# ════════════════════════════════════════════════════════════════

class AutoTracker:
    """
    Автоматическое слежение за целью.

    Принимает координаты центра цели в пикселях,
    вычисляет угловую ошибку и через PID-регулятор
    формирует команды скорости для моторов платформы.
    """

    def __init__(self, pid_yaw_cfg: PIDConfig, pid_pitch_cfg: PIDConfig,
                 center_deadzone_px: int = 15,
                 max_speed_dps: float = 80.0,
                 frame_width: int = 1920, frame_height: int = 1080,
                 fov_h_deg: float = 62.2, fov_v_deg: float = 48.8):
        """
        Args:
            pid_yaw_cfg: Конфигурация PID для горизонта
            pid_pitch_cfg: Конфигурация PID для вертикали
            center_deadzone_px: Мёртвая зона в центре кадра (пиксели)
            max_speed_dps: Максимальная скорость автослежения (°/с)
            frame_width: Ширина кадра (пиксели)
            frame_height: Высота кадра (пиксели)
            fov_h_deg: Горизонтальное поле зрения (градусы)
            fov_v_deg: Вертикальное поле зрения (градусы)
        """
        # PID-регуляторы для каждой оси
        self._pid_yaw = PIDController(
            kp=pid_yaw_cfg.kp,
            ki=pid_yaw_cfg.ki,
            kd=pid_yaw_cfg.kd,
            output_limit=max_speed_dps
        )
        self._pid_pitch = PIDController(
            kp=pid_pitch_cfg.kp,
            ki=pid_pitch_cfg.ki,
            kd=pid_pitch_cfg.kd,
            output_limit=max_speed_dps
        )

        self._deadzone_px = center_deadzone_px
        self._max_speed = max_speed_dps

        # Параметры кадра
        self._frame_w = frame_width
        self._frame_h = frame_height
        self._fov_h = fov_h_deg
        self._fov_v = fov_v_deg

        # Центр кадра
        self._cx = frame_width / 2.0
        self._cy = frame_height / 2.0

        # Статистика
        self._tracking_active = False
        self._last_error_yaw = 0.0
        self._last_error_pitch = 0.0

        logger.info(
            f"AutoTracker: deadzone={center_deadzone_px}px, "
            f"max_speed={max_speed_dps}°/с, "
            f"PID_yaw=({pid_yaw_cfg.kp}, {pid_yaw_cfg.ki}, {pid_yaw_cfg.kd}), "
            f"PID_pitch=({pid_pitch_cfg.kp}, {pid_pitch_cfg.ki}, {pid_pitch_cfg.kd})"
        )

    # ── Обновление параметров камеры ────────────────────────

    def update_camera_params(self, frame_width: int, frame_height: int,
                              fov_h: float, fov_v: float):
        """Обновить параметры камеры (при изменении зума)."""
        self._frame_w = frame_width
        self._frame_h = frame_height
        self._fov_h = fov_h
        self._fov_v = fov_v
        self._cx = frame_width / 2.0
        self._cy = frame_height / 2.0

    # ── Основной метод слежения ─────────────────────────────

    def compute(self, target_cx: float, target_cy: float,
                dt: float = None) -> Tuple[float, float]:
        """
        Вычислить скорости моторов для удержания цели в центре.

        Args:
            target_cx: X-координата центра цели (пиксели)
            target_cy: Y-координата центра цели (пиксели)
            dt: Шаг времени (сек)

        Returns:
            (yaw_speed_dps, pitch_speed_dps) — скорости в °/с
        """
        # Ошибка в пикселях (от центра кадра)
        error_px_x = target_cx - self._cx
        error_px_y = target_cy - self._cy

        # Применение мёртвой зоны
        if abs(error_px_x) < self._deadzone_px:
            error_px_x = 0.0
        if abs(error_px_y) < self._deadzone_px:
            error_px_y = 0.0

        # Преобразование ошибки в градусы
        error_yaw_deg = (error_px_x / self._frame_w) * self._fov_h
        error_pitch_deg = (error_px_y / self._frame_h) * self._fov_v

        # PID-регуляторы
        yaw_speed = self._pid_yaw.update(error_yaw_deg, dt)
        pitch_speed = self._pid_pitch.update(error_pitch_deg, dt)

        # Сохранение для отладки
        self._last_error_yaw = error_yaw_deg
        self._last_error_pitch = error_pitch_deg
        self._tracking_active = True

        return yaw_speed, pitch_speed

    def reset(self):
        """Сбросить PID-регуляторы (при потере цели или смене режима)."""
        self._pid_yaw.reset()
        self._pid_pitch.reset()
        self._tracking_active = False
        self._last_error_yaw = 0.0
        self._last_error_pitch = 0.0
        logger.debug("AutoTracker: PID сброшен")

    # ── Свойства ────────────────────────────────────────────

    @property
    def is_tracking(self) -> bool:
        """Слежение активно."""
        return self._tracking_active

    @property
    def error_yaw_deg(self) -> float:
        """Последняя ошибка по горизонту (градусы)."""
        return self._last_error_yaw

    @property
    def error_pitch_deg(self) -> float:
        """Последняя ошибка по вертикали (градусы)."""
        return self._last_error_pitch

    @property
    def is_centered(self) -> bool:
        """Цель находится в мёртвой зоне (считается отцентрированной)."""
        px_err_x = abs(self._last_error_yaw / self._fov_h * self._frame_w)
        px_err_y = abs(self._last_error_pitch / self._fov_v * self._frame_h)
        return px_err_x < self._deadzone_px and px_err_y < self._deadzone_px
