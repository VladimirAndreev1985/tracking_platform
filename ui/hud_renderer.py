# -*- coding: utf-8 -*-
"""
HUD-прицел (Head-Up Display) — рисование оверлея на видеокадре.

Рисует:
- Прицельный крест с мил-дотами
- Bounding box цели и точку перехвата
- Информационную панель (дистанция, скорость, ToF, зум, режим)
- Компас (текущий азимут)
- Статус захвата цели
- Предсказанную траекторию

Весь текст интерфейса — на русском языке.
"""

import math
import time
import logging
from typing import Optional, List, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from core.state_machine import SharedState, PlatformState, TargetLockState
from config.app_config import HUDConfig

logger = logging.getLogger(__name__)


class HUDRenderer:
    """
    Рендерер HUD-прицела.

    Принимает BGR-кадр и SharedState, рисует на кадре
    все элементы прицела и информационные панели.
    """

    def __init__(self, config: HUDConfig, frame_width: int = 1920,
                 frame_height: int = 1080):
        """
        Args:
            config: Конфигурация HUD
            frame_width: Ширина кадра
            frame_height: Высота кадра
        """
        self._cfg = config
        self._w = frame_width
        self._h = frame_height
        self._cx = frame_width // 2
        self._cy = frame_height // 2

        # Цвета (BGR)
        self._c_reticle = tuple(config.color_reticle)
        self._c_locked = tuple(config.color_locked)
        self._c_lead = tuple(config.color_lead)
        self._c_info = tuple(config.color_info)
        self._c_warning = (0, 128, 255)    # Оранжевый
        self._c_estop = (0, 0, 255)        # Красный
        self._c_bg = (0, 0, 0)             # Чёрный фон панелей

        self._thickness = config.line_thickness
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = config.font_scale
        self._mil_size = config.mil_dot_size
        self._mil_count = config.mil_dots_count
        self._mil_spacing = config.mil_dot_spacing

        logger.info(f"HUDRenderer: {frame_width}x{frame_height}")

    def update_resolution(self, width: int, height: int):
        """Обновить разрешение кадра."""
        self._w = width
        self._h = height
        self._cx = width // 2
        self._cy = height // 2

    # ════════════════════════════════════════════════════════
    # Главный метод рендеринга
    # ════════════════════════════════════════════════════════

    def render(self, frame: np.ndarray, state: SharedState) -> np.ndarray:
        """
        Нарисовать HUD на кадре.

        Args:
            frame: BGR-кадр (будет модифицирован in-place)
            state: Текущее состояние системы

        Returns:
            Кадр с нарисованным HUD
        """
        if frame is None or cv2 is None:
            return frame

        # Определяем цвет прицела по статусу
        if state.platform_state == PlatformState.ESTOP:
            reticle_color = self._c_estop
        elif state.target_lock == TargetLockState.LOCKED:
            reticle_color = self._c_locked
        else:
            reticle_color = self._c_reticle

        # 1. Прицельный крест
        self._draw_reticle(frame, reticle_color)

        # 2. Мил-доты
        self._draw_mil_dots(frame, reticle_color)

        # 3. Bounding box цели
        if state.target_bbox is not None:
            self._draw_target_bbox(frame, state)

        # 4. Точка перехвата (lead point)
        if state.lead_point_px is not None and state.intercept_possible:
            self._draw_lead_point(frame, state)

        # 5. Предсказанная траектория
        if state.predicted_positions:
            self._draw_trajectory(frame, state)

        # 6. Информационная панель (правый верх)
        if self._cfg.show_info_panel:
            self._draw_info_panel(frame, state)

        # 7. Статус режима (левый верх)
        self._draw_mode_status(frame, state)

        # 8. Компас (нижний центр)
        if self._cfg.show_compass:
            self._draw_compass(frame, state)

        # 9. Предупреждения / E-STOP
        if state.platform_state == PlatformState.ESTOP:
            self._draw_estop_warning(frame)
        elif state.platform_state == PlatformState.ERROR:
            self._draw_error_warning(frame, state.error_message)

        return frame

    # ════════════════════════════════════════════════════════
    # Элементы прицела
    # ════════════════════════════════════════════════════════

    def _draw_reticle(self, frame: np.ndarray, color: tuple):
        """Прицельный крест с зазором в центре."""
        cx, cy = self._cx, self._cy
        gap = 20   # Зазор в центре
        arm = 60   # Длина линии

        # Горизонтальные линии
        cv2.line(frame, (cx - arm - gap, cy), (cx - gap, cy), color, self._thickness)
        cv2.line(frame, (cx + gap, cy), (cx + arm + gap, cy), color, self._thickness)

        # Вертикальные линии
        cv2.line(frame, (cx, cy - arm - gap), (cx, cy - gap), color, self._thickness)
        cv2.line(frame, (cx, cy + gap), (cx, cy + arm + gap), color, self._thickness)

        # Центральная точка
        cv2.circle(frame, (cx, cy), 2, color, -1)

    def _draw_mil_dots(self, frame: np.ndarray, color: tuple):
        """Мил-доты (шкала дальности/углов)."""
        cx, cy = self._cx, self._cy
        ms = self._mil_size
        spacing = self._mil_spacing

        for i in range(1, self._mil_count + 1):
            offset = i * spacing

            # Горизонтальные мил-доты
            cv2.circle(frame, (cx - offset, cy), ms, color, -1)
            cv2.circle(frame, (cx + offset, cy), ms, color, -1)

            # Вертикальные мил-доты
            cv2.circle(frame, (cx, cy - offset), ms, color, -1)
            cv2.circle(frame, (cx, cy + offset), ms, color, -1)

    def _draw_target_bbox(self, frame: np.ndarray, state: SharedState):
        """Bounding box вокруг цели."""
        x1, y1, x2, y2 = [int(v) for v in state.target_bbox]

        # Цвет зависит от статуса захвата
        if state.target_lock == TargetLockState.LOCKED:
            color = self._c_locked
            thickness = 2
        else:
            color = self._c_reticle
            thickness = 1

        # Рисуем уголки вместо полного прямоугольника (стиль HUD)
        corner_len = max(15, min(x2 - x1, y2 - y1) // 4)

        # Верхний левый
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        # Верхний правый
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        # Нижний левый
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        # Нижний правый
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

        # Метка класса и уверенности
        from perception.detector import get_class_name
        label = f"{get_class_name(state.target_class_id)} {state.target_confidence:.0%}"
        self._draw_text_with_bg(frame, label, (x1, y1 - 8), color, scale=0.4)

        # ID трека
        if state.track_id >= 0:
            tid_label = f"ID:{state.track_id}"
            self._draw_text_with_bg(frame, tid_label, (x2 - 50, y1 - 8), color, scale=0.35)

    def _draw_lead_point(self, frame: np.ndarray, state: SharedState):
        """Точка перехвата (lead point)."""
        lx, ly = int(state.lead_point_px[0]), int(state.lead_point_px[1])

        # Перекрестие точки перехвата
        size = 12
        cv2.line(frame, (lx - size, ly), (lx + size, ly), self._c_lead, 2)
        cv2.line(frame, (lx, ly - size), (lx, ly + size), self._c_lead, 2)
        cv2.circle(frame, (lx, ly), size, self._c_lead, 1)

        # Линия от центра цели до точки перехвата
        if state.target_center_px:
            tcx, tcy = int(state.target_center_px[0]), int(state.target_center_px[1])
            cv2.line(frame, (tcx, tcy), (lx, ly), self._c_lead, 1, cv2.LINE_AA)

        # Метка
        label = f"ПЕРЕХВАТ ToF:{state.time_of_flight_sec:.2f}с"
        self._draw_text_with_bg(frame, label, (lx + 15, ly - 5), self._c_lead, scale=0.35)

    def _draw_trajectory(self, frame: np.ndarray, state: SharedState):
        """Предсказанная траектория (пунктирная линия)."""
        # TODO: Преобразовать 3D-точки в пиксели через камеру
        # Пока рисуем как серию точек, если есть lead_point
        pass

    # ════════════════════════════════════════════════════════
    # Информационные панели
    # ════════════════════════════════════════════════════════

    def _draw_info_panel(self, frame: np.ndarray, state: SharedState):
        """Информационная панель справа."""
        x = self._w - 280
        y = 15
        line_h = 22
        scale = 0.42

        lines = [
            # Вооружение (верхняя строка — выделена)
            f"[{state.weapon_name}] {state.weapon_caliber}" if state.weapon_name else "ОРУЖИЕ: ---",
            f"",
            f"ДИСТ: {state.distance_m:.0f} м" if state.distance_valid else "ДИСТ: --- м",
            f"СКОР: {state.target_speed_mps:.1f} м/с",
            f"КУРС: {state.target_heading_deg:.0f}°",
            f"ЗУМ:  {state.zoom_level:.1f}x",
            f"",
            f"ToF:  {state.time_of_flight_sec:.2f} с" if state.time_of_flight_sec > 0 else "ToF:  --- с",
            f"УПРЖ: {state.lead_yaw_deg:+.1f}° / {state.lead_pitch_deg:+.1f}°" if state.intercept_possible else "УПРЖ: ---",
            f"E:    {state.bullet_energy_j:.0f} Дж" if state.bullet_energy_j > 0 else "E:    --- Дж",
            f"МАХ:  {state.mach_at_target:.2f}" if state.mach_at_target > 0 else "МАХ:  ---",
            f"",
            f"YAW:  {state.yaw_deg:+.1f}°",
            f"PITCH:{state.pitch_deg:+.1f}°",
            f"FPS:  {state.camera_fps:.0f}",
            f"ЦИКЛ: {state.loop_time_ms:.1f} мс",
        ]

        # Фон панели
        panel_h = len(lines) * line_h + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 5), (self._w - 5, y + panel_h),
                      self._c_bg, -1)
        cv2.addWeighted(overlay, self._cfg.overlay_alpha, frame,
                        1 - self._cfg.overlay_alpha, 0, frame)

        # Текст
        for i, line in enumerate(lines):
            if line:
                color = self._c_info
                # Подсветка строки вооружения (первая строка — жёлтым)
                if i == 0 and state.weapon_name:
                    color = self._c_lead
                # Подсветка дистанции
                elif "ДИСТ:" in line and state.distance_valid:
                    color = self._c_lead
                cv2.putText(frame, line, (x, y + i * line_h + 15),
                            self._font, scale, color, 1, cv2.LINE_AA)

    def _draw_mode_status(self, frame: np.ndarray, state: SharedState):
        """Статус режима (левый верхний угол)."""
        x, y = 15, 15

        # Режим работы
        mode_names = {
            PlatformState.INIT: ("ИНИЦИАЛИЗАЦИЯ", self._c_warning),
            PlatformState.MANUAL: ("РУЧНОЙ", self._c_info),
            PlatformState.AUTO: ("АВТО", self._c_lead),
            PlatformState.ESTOP: ("АВАРИЙНАЯ ОСТАНОВКА", self._c_estop),
            PlatformState.ERROR: ("ОШИБКА", self._c_estop),
            PlatformState.SHUTDOWN: ("ЗАВЕРШЕНИЕ", self._c_warning),
        }
        mode_text, mode_color = mode_names.get(
            state.platform_state, ("---", self._c_info)
        )

        # Фон
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 5), (x + 250, y + 55),
                      self._c_bg, -1)
        cv2.addWeighted(overlay, self._cfg.overlay_alpha, frame,
                        1 - self._cfg.overlay_alpha, 0, frame)

        # Режим
        cv2.putText(frame, f"РЕЖИМ: {mode_text}", (x, y + 18),
                    self._font, 0.55, mode_color, 1, cv2.LINE_AA)

        # Статус захвата
        lock_names = {
            TargetLockState.NO_TARGET: ("НЕТ ЦЕЛИ", self._c_info),
            TargetLockState.SEARCHING: ("ПОИСК...", self._c_warning),
            TargetLockState.DETECTED: ("ОБНАРУЖЕНА", self._c_lead),
            TargetLockState.LOCKED: ("ЗАХВАТ ●", self._c_locked),
            TargetLockState.LOST: ("ПОТЕРЯНА", self._c_estop),
        }
        lock_text, lock_color = lock_names.get(
            state.target_lock, ("---", self._c_info)
        )
        cv2.putText(frame, f"ЦЕЛЬ:  {lock_text}", (x, y + 45),
                    self._font, 0.50, lock_color, 1, cv2.LINE_AA)

    def _draw_compass(self, frame: np.ndarray, state: SharedState):
        """Компас (азимут) внизу по центру."""
        cx = self._cx
        y = self._h - 40
        width = 300

        # Фон
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx - width // 2, y - 20), (cx + width // 2, y + 15),
                      self._c_bg, -1)
        cv2.addWeighted(overlay, self._cfg.overlay_alpha, frame,
                        1 - self._cfg.overlay_alpha, 0, frame)

        # Шкала азимута
        yaw = state.yaw_deg
        for deg_offset in range(-30, 31, 5):
            angle = yaw + deg_offset
            px = cx + int(deg_offset * (width / 60))

            if abs(deg_offset) % 10 == 0:
                # Основные деления
                cv2.line(frame, (px, y - 12), (px, y), self._c_info, 1)
                label = f"{int(angle % 360)}°"
                text_size = cv2.getTextSize(label, self._font, 0.3, 1)[0]
                cv2.putText(frame, label, (px - text_size[0] // 2, y + 12),
                            self._font, 0.3, self._c_info, 1, cv2.LINE_AA)
            else:
                # Мелкие деления
                cv2.line(frame, (px, y - 6), (px, y), self._c_info, 1)

        # Центральная метка
        cv2.drawMarker(frame, (cx, y - 15), self._c_lead,
                        cv2.MARKER_TRIANGLE_DOWN, 8, 1)

    # ════════════════════════════════════════════════════════
    # Предупреждения
    # ════════════════════════════════════════════════════════

    def _draw_estop_warning(self, frame: np.ndarray):
        """Предупреждение аварийной остановки — мигающее."""
        if int(time.time() * 3) % 2 == 0:  # Мигание 3 Гц
            text = "!!! АВАРИЙНАЯ ОСТАНОВКА !!!"
            text_size = cv2.getTextSize(text, self._font, 1.0, 2)[0]
            tx = (self._w - text_size[0]) // 2
            ty = self._h // 2

            # Фон
            cv2.rectangle(frame, (tx - 20, ty - 40), (tx + text_size[0] + 20, ty + 15),
                          (0, 0, 100), -1)
            cv2.putText(frame, text, (tx, ty), self._font, 1.0,
                        self._c_estop, 2, cv2.LINE_AA)

    def _draw_error_warning(self, frame: np.ndarray, message: str):
        """Предупреждение об ошибке."""
        text = f"ОШИБКА: {message}"
        text_size = cv2.getTextSize(text, self._font, 0.6, 1)[0]
        tx = (self._w - text_size[0]) // 2
        ty = self._h // 2 + 50

        cv2.rectangle(frame, (tx - 10, ty - 25), (tx + text_size[0] + 10, ty + 5),
                      (0, 0, 80), -1)
        cv2.putText(frame, text, (tx, ty), self._font, 0.6,
                    self._c_warning, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════
    # Утилиты
    # ════════════════════════════════════════════════════════

    def _draw_text_with_bg(self, frame: np.ndarray, text: str,
                           pos: tuple, color: tuple,
                           scale: float = 0.4, thickness: int = 1):
        """Нарисовать текст с полупрозрачным фоном."""
        text_size = cv2.getTextSize(text, self._font, scale, thickness)[0]
        x, y = int(pos[0]), int(pos[1])

        # Фон
        cv2.rectangle(frame, (x - 2, y - text_size[1] - 4),
                      (x + text_size[0] + 2, y + 2),
                      self._c_bg, -1)
        # Текст
        cv2.putText(frame, text, (x, y), self._font, scale,
                    color, thickness, cv2.LINE_AA)
