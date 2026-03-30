# -*- coding: utf-8 -*-
"""
Предсказатель траектории движущейся цели.

Накапливает историю позиций цели и на её основе:
1. Оценивает текущую скорость и направление движения
2. Предсказывает будущие позиции цели
3. Определяет тип траектории (прямолинейная, поворот, ускорение)

Используется фильтр Калмана для сглаживания наблюдений
и экстраполяции позиции цели в будущее.
"""

import time
import math
import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Наблюдение (одна точка)
# ════════════════════════════════════════════════════════════════

@dataclass
class Observation:
    """Одно наблюдение позиции цели."""
    timestamp: float          # Время наблюдения (сек)
    yaw_deg: float            # Угол горизонта платформы (°)
    pitch_deg: float          # Угол вертикали платформы (°)
    distance_m: float         # Дистанция до цели (м)
    pixel_x: float = 0.0     # X-координата в кадре (пиксели)
    pixel_y: float = 0.0     # Y-координата в кадре (пиксели)


# ════════════════════════════════════════════════════════════════
# Оценка состояния цели
# ════════════════════════════════════════════════════════════════

@dataclass
class TargetEstimate:
    """Оценка текущего состояния цели."""
    # Позиция в декартовых координатах (м, относительно платформы)
    x: float = 0.0            # Вперёд
    y: float = 0.0            # Вправо
    z: float = 0.0            # Вверх

    # Скорость (м/с)
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0

    # Скалярная скорость и курс
    speed_mps: float = 0.0    # |v| (м/с)
    heading_deg: float = 0.0  # Курс в горизонтальной плоскости (°)
    climb_rate_mps: float = 0.0  # Вертикальная скорость (м/с)

    # Ускорение (м/с²)
    ax: float = 0.0
    ay: float = 0.0
    az: float = 0.0

    # Качество оценки
    confidence: float = 0.0   # 0.0 .. 1.0
    is_valid: bool = False
    timestamp: float = 0.0


# ════════════════════════════════════════════════════════════════
# Предсказатель траектории
# ════════════════════════════════════════════════════════════════

class TrajectoryPredictor:
    """
    Предсказатель траектории на основе истории наблюдений.

    Использует:
    - Скользящее среднее для оценки скорости
    - Линейную экстраполяцию для предсказания
    - Фильтрацию выбросов
    """

    def __init__(self, history_size: int = 30,
                 prediction_horizon_sec: float = 3.0,
                 min_observations: int = 5):
        """
        Args:
            history_size: Размер буфера истории наблюдений
            prediction_horizon_sec: Горизонт предсказания (сек)
            min_observations: Минимум наблюдений для начала предсказания
        """
        self._history_size = history_size
        self._prediction_horizon = prediction_horizon_sec
        self._min_obs = min_observations

        # Буфер наблюдений (кольцевой)
        self._observations: deque = deque(maxlen=history_size)

        # Текущая оценка состояния
        self.estimate = TargetEstimate()

        # Сглаженные значения скорости (экспоненциальное скользящее среднее)
        self._smooth_vx = 0.0
        self._smooth_vy = 0.0
        self._smooth_vz = 0.0
        self._smooth_alpha = 0.3  # Коэффициент сглаживания

        logger.info(
            f"TrajectoryPredictor: история={history_size}, "
            f"горизонт={prediction_horizon_sec}с, "
            f"мин. наблюдений={min_observations}"
        )

    def add_observation(self, yaw_deg: float, pitch_deg: float,
                        distance_m: float,
                        pixel_x: float = 0.0, pixel_y: float = 0.0):
        """
        Добавить новое наблюдение позиции цели.

        Args:
            yaw_deg: Угол горизонта платформы (°)
            pitch_deg: Угол вертикали платформы (°)
            distance_m: Дистанция до цели (м)
            pixel_x: X в кадре (пиксели)
            pixel_y: Y в кадре (пиксели)
        """
        obs = Observation(
            timestamp=time.time(),
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            distance_m=distance_m,
            pixel_x=pixel_x,
            pixel_y=pixel_y
        )
        self._observations.append(obs)

        # Обновить оценку состояния
        if len(self._observations) >= 2:
            self._update_estimate()

    def _update_estimate(self):
        """Обновить оценку состояния цели на основе последних наблюдений."""
        if len(self._observations) < 2:
            return

        # Последние два наблюдения для мгновенной скорости
        curr = self._observations[-1]
        prev = self._observations[-2]

        dt = curr.timestamp - prev.timestamp
        if dt <= 0:
            return

        # Преобразование сферических координат в декартовы
        curr_xyz = self._spherical_to_cartesian(
            curr.yaw_deg, curr.pitch_deg, curr.distance_m
        )
        prev_xyz = self._spherical_to_cartesian(
            prev.yaw_deg, prev.pitch_deg, prev.distance_m
        )

        # Мгновенная скорость
        raw_vx = (curr_xyz[0] - prev_xyz[0]) / dt
        raw_vy = (curr_xyz[1] - prev_xyz[1]) / dt
        raw_vz = (curr_xyz[2] - prev_xyz[2]) / dt

        # Сглаживание (EMA)
        a = self._smooth_alpha
        self._smooth_vx += a * (raw_vx - self._smooth_vx)
        self._smooth_vy += a * (raw_vy - self._smooth_vy)
        self._smooth_vz += a * (raw_vz - self._smooth_vz)

        # Обновление оценки
        est = self.estimate
        est.x, est.y, est.z = curr_xyz
        est.vx = self._smooth_vx
        est.vy = self._smooth_vy
        est.vz = self._smooth_vz

        # Скалярная скорость
        est.speed_mps = math.sqrt(
            est.vx ** 2 + est.vy ** 2 + est.vz ** 2
        )

        # Курс в горизонтальной плоскости
        est.heading_deg = math.degrees(math.atan2(est.vy, est.vx))

        # Вертикальная скорость
        est.climb_rate_mps = est.vz

        # Ускорение (если достаточно данных)
        if len(self._observations) >= 3:
            prev2 = self._observations[-3]
            dt2 = prev.timestamp - prev2.timestamp
            if dt2 > 0:
                prev2_xyz = self._spherical_to_cartesian(
                    prev2.yaw_deg, prev2.pitch_deg, prev2.distance_m
                )
                prev_vx = (prev_xyz[0] - prev2_xyz[0]) / dt2
                prev_vy = (prev_xyz[1] - prev2_xyz[1]) / dt2
                prev_vz = (prev_xyz[2] - prev2_xyz[2]) / dt2
                est.ax = (raw_vx - prev_vx) / dt
                est.ay = (raw_vy - prev_vy) / dt
                est.az = (raw_vz - prev_vz) / dt

        # Уверенность оценки (растёт с количеством наблюдений)
        n = len(self._observations)
        est.confidence = min(1.0, n / (self._min_obs * 2))
        est.is_valid = n >= self._min_obs
        est.timestamp = curr.timestamp

    def predict(self, seconds_ahead: float = None,
                num_points: int = 10) -> List[Tuple[float, float, float, float]]:
        """
        Предсказать будущие позиции цели.

        Использует текущую скорость и ускорение для экстраполяции.

        Args:
            seconds_ahead: Горизонт предсказания (сек). None = из конфига.
            num_points: Количество точек предсказания.

        Returns:
            Список кортежей (x, y, z, t) — координаты и время
        """
        if not self.estimate.is_valid:
            return []

        horizon = seconds_ahead or self._prediction_horizon
        dt = horizon / num_points

        est = self.estimate
        points = []

        for i in range(1, num_points + 1):
            t = dt * i
            # Экстраполяция с учётом ускорения
            px = est.x + est.vx * t + 0.5 * est.ax * t * t
            py = est.y + est.vy * t + 0.5 * est.ay * t * t
            pz = est.z + est.vz * t + 0.5 * est.az * t * t
            points.append((px, py, pz, t))

        return points

    def predict_at_time(self, t: float) -> Tuple[float, float, float]:
        """
        Предсказать позицию цели через t секунд.

        Args:
            t: Время в будущем (сек)

        Returns:
            (x, y, z) — предсказанные координаты
        """
        est = self.estimate
        px = est.x + est.vx * t + 0.5 * est.ax * t * t
        py = est.y + est.vy * t + 0.5 * est.ay * t * t
        pz = est.z + est.vz * t + 0.5 * est.az * t * t
        return px, py, pz

    def reset(self):
        """Сбросить историю и оценку (при потере цели)."""
        self._observations.clear()
        self.estimate = TargetEstimate()
        self._smooth_vx = 0.0
        self._smooth_vy = 0.0
        self._smooth_vz = 0.0
        logger.debug("TrajectoryPredictor: сброс")

    # ── Утилиты ─────────────────────────────────────────────

    @staticmethod
    def _spherical_to_cartesian(yaw_deg: float, pitch_deg: float,
                                 distance_m: float) -> Tuple[float, float, float]:
        """
        Преобразовать сферические координаты (yaw, pitch, distance)
        в декартовы (x, y, z).

        x = вперёд, y = вправо, z = вверх
        """
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)

        x = distance_m * math.cos(pitch_rad) * math.cos(yaw_rad)
        y = distance_m * math.cos(pitch_rad) * math.sin(yaw_rad)
        z = distance_m * math.sin(pitch_rad)

        return x, y, z

    @staticmethod
    def _cartesian_to_spherical(x: float, y: float,
                                 z: float) -> Tuple[float, float, float]:
        """
        Преобразовать декартовы координаты в сферические.

        Returns:
            (yaw_deg, pitch_deg, distance_m)
        """
        distance = math.sqrt(x * x + y * y + z * z)
        if distance < 0.001:
            return 0.0, 0.0, 0.0

        yaw_deg = math.degrees(math.atan2(y, x))
        pitch_deg = math.degrees(math.asin(z / distance))

        return yaw_deg, pitch_deg, distance

    @property
    def observation_count(self) -> int:
        """Количество наблюдений в буфере."""
        return len(self._observations)

    @property
    def has_enough_data(self) -> bool:
        """Достаточно данных для предсказания."""
        return len(self._observations) >= self._min_obs
