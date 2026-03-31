# -*- coding: utf-8 -*-
"""
Боевой HUD-прицел (Head-Up Display).

Оптимизирован для быстрого принятия решений оператором.
Три уровня информации:
  1. Периферийное зрение (цвета/формы): прицел, bbox, пульсация, ГОТОВ
  2. Быстрый взгляд (1 сек): дистанция, тип цели, статус готовности
  3. Детальный анализ (если есть время): ToF, скорость, зум

Компоновка экрана (1920×1080):
  ┌──────────────────────────────────────────────────────┐
  │ [РЕЖИМ/ЦЕЛЬ]                    [ОГНЕВЫЕ ДАННЫЕ]     │
  │                                                       │
  │ ▐ шкала                  ┌──┐                         │
  │ ▐ дальн.                 │TИП│→вектор                 │
  │ ▐                        └──┘                         │
  │ ▐                            ◇ перехват               │
  │                                                       │
  │              ● ГОТОВ К СТРЕЛЬБЕ ●                     │
  │         ▼ углы платформы ▼                            │
  └──────────────────────────────────────────────────────┘

Весь текст — на русском языке.
Компас убран (нет магнитометра). Вместо него — шкала углов платформы.
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
    Боевой HUD-прицел.

    Принимает BGR-кадр и SharedState, рисует на кадре
    все элементы боевого интерфейса.
    """

    def __init__(self, config: HUDConfig, frame_width: int = 1920,
                 frame_height: int = 1080):
        self._cfg = config
        self._w = frame_width
        self._h = frame_height
        self._cx = frame_width // 2
        self._cy = frame_height // 2

        # ── Цвета (BGR) ──
        self._c_reticle = tuple(config.color_reticle)    # Зелёный — прицел
        self._c_locked = tuple(config.color_locked)       # Красный — захват
        self._c_lead = tuple(config.color_lead)           # Жёлтый — перехват/авто
        self._c_info = tuple(config.color_info)           # Зелёный — текст
        self._c_warning = (0, 128, 255)                   # Оранжевый
        self._c_estop = (0, 0, 255)                       # Красный
        self._c_bg = (0, 0, 0)                            # Чёрный фон
        self._c_ready = (0, 255, 0)                       # Ярко-зелёный — ГОТОВ
        self._c_not_ready = (0, 0, 180)                   # Тёмно-красный — НЕ ГОТОВ
        self._c_white = (255, 255, 255)                   # Белый
        self._c_threat_high = (0, 0, 255)                 # Красный — высокая угроза
        self._c_threat_mid = (0, 165, 255)                # Оранжевый — средняя
        self._c_threat_low = (200, 200, 200)              # Серый — низкая (птица)
        self._c_range_ok = (0, 200, 0)                    # Зелёный — в зоне
        self._c_range_out = (80, 80, 80)                  # Серый — вне зоны
        self._c_firing = (0, 100, 255)                    # Оранжево-красный — стрельба

        self._thickness = config.line_thickness
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = config.font_scale
        self._mil_size = config.mil_dot_size
        self._mil_count = config.mil_dots_count
        self._mil_spacing = config.mil_dot_spacing

        logger.info(f"HUDRenderer [боевой]: {frame_width}x{frame_height}")

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
        """Нарисовать боевой HUD на кадре."""
        if frame is None or cv2 is None:
            return frame

        # Цвет прицела по статусу
        if state.platform_state == PlatformState.ESTOP:
            reticle_color = self._c_estop
        elif state.target_lock == TargetLockState.LOCKED:
            reticle_color = self._c_locked
        else:
            reticle_color = self._c_reticle

        # ── Центральная зона (прицел) ──
        self._draw_reticle(frame, reticle_color)
        self._draw_mil_dots(frame, reticle_color)

        # ── Цель ──
        if state.target_bbox is not None:
            self._draw_target_bbox(frame, state)
            self._draw_velocity_vector(frame, state)

        # ── Точка перехвата ──
        if state.lead_point_px is not None and state.intercept_possible:
            self._draw_lead_point(frame, state)

        # ── Левый верх: режим и статус захвата + PTZ ──
        self._draw_mode_status(frame, state)

        # ── Правый верх: огневые данные ──
        self._draw_fire_data_panel(frame, state)

        # ── Левый край: шкала дальности ──
        if state.effective_range_m > 0:
            self._draw_range_bar(frame, state)

        # ── Нижний центр: индикатор готовности ──
        self._draw_readiness_indicator(frame, state)

        # ── Нижний центр (под готовностью): шкала углов платформы ──
        self._draw_platform_angles(frame, state)

        # ── Индикатор стрельбы ──
        if state.firing_active:
            self._draw_firing_indicator(frame, state)

        # ── Предупреждения ──
        if state.platform_state == PlatformState.ESTOP:
            self._draw_estop_warning(frame)
        elif state.platform_state == PlatformState.ERROR:
            self._draw_error_warning(frame, state.error_message)

        return frame

    # ════════════════════════════════════════════════════════
    # Прицел
    # ════════════════════════════════════════════════════════

    def _draw_reticle(self, frame: np.ndarray, color: tuple):
        """Прицельный крест с зазором."""
        cx, cy = self._cx, self._cy
        gap = 20
        arm = 60

        cv2.line(frame, (cx - arm - gap, cy), (cx - gap, cy), color, self._thickness)
        cv2.line(frame, (cx + gap, cy), (cx + arm + gap, cy), color, self._thickness)
        cv2.line(frame, (cx, cy - arm - gap), (cx, cy - gap), color, self._thickness)
        cv2.line(frame, (cx, cy + gap), (cx, cy + arm + gap), color, self._thickness)
        cv2.circle(frame, (cx, cy), 2, color, -1)

    def _draw_mil_dots(self, frame: np.ndarray, color: tuple):
        """Мил-доты."""
        cx, cy = self._cx, self._cy
        ms = self._mil_size
        sp = self._mil_spacing
        for i in range(1, self._mil_count + 1):
            off = i * sp
            cv2.circle(frame, (cx - off, cy), ms, color, -1)
            cv2.circle(frame, (cx + off, cy), ms, color, -1)
            cv2.circle(frame, (cx, cy - off), ms, color, -1)
            cv2.circle(frame, (cx, cy + off), ms, color, -1)

    # ════════════════════════════════════════════════════════
    # Цель (bbox + тип + вектор)
    # ════════════════════════════════════════════════════════

    def _draw_target_bbox(self, frame: np.ndarray, state: SharedState):
        """Bounding box с уголками + тип цели крупно."""
        x1, y1, x2, y2 = [int(v) for v in state.target_bbox]

        if state.target_lock == TargetLockState.LOCKED:
            color = self._c_locked
            thickness = 2
        else:
            color = self._c_reticle
            thickness = 1

        # Уголки (стиль боевого HUD)
        cl = max(15, min(x2 - x1, y2 - y1) // 4)
        cv2.line(frame, (x1, y1), (x1 + cl, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + cl), color, thickness)
        cv2.line(frame, (x2, y1), (x2 - cl, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + cl), color, thickness)
        cv2.line(frame, (x1, y2), (x1 + cl, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - cl), color, thickness)
        cv2.line(frame, (x2, y2), (x2 - cl, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - cl), color, thickness)

        # Тип цели — КРУПНО, с цветом по уровню угрозы
        if state.target_type_name:
            threat = state.threat_level
            if threat >= 8:
                type_color = self._c_threat_high
            elif threat >= 4:
                type_color = self._c_threat_mid
            else:
                type_color = self._c_threat_low

            label = state.target_type_name.upper()
            self._draw_text_with_bg(frame, label, (x1, y1 - 10),
                                     type_color, scale=0.55, thickness=2)

        # ID трека (мелко, справа)
        if state.track_id >= 0:
            tid = f"#{state.track_id}"
            self._draw_text_with_bg(frame, tid, (x2 - 40, y1 - 10),
                                     color, scale=0.35)

    def _draw_velocity_vector(self, frame: np.ndarray, state: SharedState):
        """Стрелка направления движения цели от центра bbox."""
        if not state.target_velocity_px or not state.target_center_px:
            return

        cx, cy = int(state.target_center_px[0]), int(state.target_center_px[1])
        dx, dy = state.target_velocity_px

        if abs(dx) < 2 and abs(dy) < 2:
            return  # Слишком маленький вектор

        ex, ey = int(cx + dx), int(cy + dy)

        # Стрелка
        cv2.arrowedLine(frame, (cx, cy), (ex, ey),
                        self._c_lead, 2, cv2.LINE_AA, tipLength=0.3)

    # ════════════════════════════════════════════════════════
    # Точка перехвата (lead point)
    # ════════════════════════════════════════════════════════

    def _draw_lead_point(self, frame: np.ndarray, state: SharedState):
        """Пульсирующая точка перехвата."""
        lx, ly = int(state.lead_point_px[0]), int(state.lead_point_px[1])

        # Пульсация 2 Гц
        pulse = math.sin(time.time() * 2.0 * math.pi * 2.0)
        pulse_n = (pulse + 1.0) / 2.0  # 0..1

        base = 14
        size = int(base + pulse_n * 6)
        thick = 2 if pulse_n > 0.3 else 1

        # Цвет пульсирует
        r = int(self._c_lead[0] + (255 - self._c_lead[0]) * pulse_n * 0.5)
        g = int(self._c_lead[1] + (255 - self._c_lead[1]) * pulse_n * 0.5)
        b = int(self._c_lead[2] + (255 - self._c_lead[2]) * pulse_n * 0.5)
        pc = (min(r, 255), min(g, 255), min(b, 255))

        # Перекрестие + круг
        cv2.line(frame, (lx - size, ly), (lx + size, ly), pc, thick)
        cv2.line(frame, (lx, ly - size), (lx, ly + size), pc, thick)
        cv2.circle(frame, (lx, ly), size, pc, 1)

        # Ромб
        ds = size // 2
        pts = np.array([[lx, ly - ds], [lx + ds, ly],
                         [lx, ly + ds], [lx - ds, ly]], np.int32)
        cv2.polylines(frame, [pts], True, pc, 1, cv2.LINE_AA)

        # Пунктирная линия от цели
        if state.target_center_px:
            tcx, tcy = int(state.target_center_px[0]), int(state.target_center_px[1])
            self._draw_dashed_line(frame, (tcx, tcy), (lx, ly), self._c_lead, 1, 8)

        # Метка ToF
        label = f"ToF {state.time_of_flight_sec:.2f}c"
        self._draw_text_with_bg(frame, label, (lx + 18, ly - 5), pc, scale=0.38)

    # ════════════════════════════════════════════════════════
    # Левый верх: режим + статус захвата
    # ════════════════════════════════════════════════════════

    def _draw_mode_status(self, frame: np.ndarray, state: SharedState):
        """Режим работы и статус захвата."""
        x, y = 15, 15

        mode_map = {
            PlatformState.INIT: ("ИНИЦИАЛИЗАЦИЯ", self._c_warning),
            PlatformState.MANUAL: ("РУЧНОЙ", self._c_info),
            PlatformState.AUTO: ("АВТО", self._c_lead),
            PlatformState.ESTOP: ("СТОП", self._c_estop),
            PlatformState.ERROR: ("ОШИБКА", self._c_estop),
            PlatformState.SHUTDOWN: ("ВЫКЛ", self._c_warning),
        }
        mode_text, mode_color = mode_map.get(state.platform_state, ("---", self._c_info))

        lock_map = {
            TargetLockState.NO_TARGET: ("НЕТ ЦЕЛИ", self._c_info),
            TargetLockState.SEARCHING: ("ПОИСК...", self._c_warning),
            TargetLockState.DETECTED: ("ОБНАРУЖЕНА", self._c_lead),
            TargetLockState.LOCKED: ("ЗАХВАТ", self._c_locked),
            TargetLockState.LOST: ("ПОТЕРЯНА", self._c_estop),
        }
        lock_text, lock_color = lock_map.get(state.target_lock, ("---", self._c_info))

        # Высота панели зависит от наличия PTZ
        panel_h = 80 if state.split_aiming_active else 55

        # Фон панели
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 5, y - 5), (x + 220, y + panel_h), self._c_bg, -1)
        cv2.addWeighted(overlay, self._cfg.overlay_alpha, frame,
                        1 - self._cfg.overlay_alpha, 0, frame)

        cv2.putText(frame, f"РЕЖИМ: {mode_text}", (x, y + 18),
                    self._font, 0.55, mode_color, 1, cv2.LINE_AA)

        # Статус захвата — с мигающим кружком при ЗАХВАТ
        lock_display = lock_text
        if state.target_lock == TargetLockState.LOCKED:
            blink = int(time.time() * 4) % 2 == 0
            lock_display = f"ЗАХВАТ {'●' if blink else '○'}"

        cv2.putText(frame, f"ЦЕЛЬ:  {lock_display}", (x, y + 45),
                    self._font, 0.50, lock_color, 1, cv2.LINE_AA)

        # Индикатор раздельного наведения (PTZ-компенсация)
        if state.split_aiming_active:
            cv2.putText(frame, "PTZ: РАЗД.НАВЕД.", (x, y + 72),
                        self._font, 0.40, self._c_lead, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════
    # Правый верх: огневые данные (минимум для решения)
    # ════════════════════════════════════════════════════════

    def _draw_fire_data_panel(self, frame: np.ndarray, state: SharedState):
        """Компактная панель огневых данных."""
        x = self._w - 260
        y = 15
        lh = 24  # Высота строки
        sc = 0.45

        lines = []

        # Строка 1: оружие (выделена жёлтым)
        if state.weapon_name:
            lines.append((f"[{state.weapon_name}] {state.weapon_caliber}", self._c_lead))
        else:
            lines.append(("ОРУЖИЕ: ---", self._c_info))

        # Пустая строка-разделитель
        lines.append(("", self._c_info))

        # Строка 2: дистанция — ГЛАВНАЯ ЦИФРА
        if state.distance_valid:
            dist_color = self._c_lead if state.in_effective_range else self._c_estop
            lines.append((f"ДИСТ: {state.distance_m:.0f} м", dist_color))
        else:
            lines.append(("ДИСТ: --- м", self._c_info))

        # Строка 3: ToF
        if state.time_of_flight_sec > 0:
            lines.append((f"ToF:  {state.time_of_flight_sec:.2f} с", self._c_info))
        else:
            lines.append(("ToF:  --- с", self._c_info))

        # Строка 4: скорость цели + направление
        if state.target_speed_mps > 0.5:
            heading = state.target_heading_deg
            direction = self._heading_to_direction(heading)
            lines.append((f"СКОР: {state.target_speed_mps:.0f} м/с {direction}", self._c_info))
        else:
            lines.append(("СКОР: --- м/с", self._c_info))

        # Строка 5: зум
        lines.append((f"ЗУМ:  {state.zoom_level:.1f}x", self._c_info))

        # Фон панели
        panel_h = len(lines) * lh + 10
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 5), (self._w - 5, y + panel_h),
                      self._c_bg, -1)
        cv2.addWeighted(overlay, self._cfg.overlay_alpha, frame,
                        1 - self._cfg.overlay_alpha, 0, frame)

        # Рисуем строки
        for i, (text, color) in enumerate(lines):
            if text:
                cv2.putText(frame, text, (x, y + i * lh + 18),
                            self._font, sc, color, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════
    # Левый край: шкала дальности (range bar)
    # ════════════════════════════════════════════════════════

    def _draw_range_bar(self, frame: np.ndarray, state: SharedState):
        """Вертикальная шкала дальности слева."""
        x = 25
        y_top = 90
        y_bot = self._h - 120
        bar_w = 12
        bar_h = y_bot - y_top

        max_range = state.effective_range_m * 1.5  # Шкала до 150% эфф. дальности
        eff_range = state.effective_range_m

        if max_range <= 0:
            return

        # Фон шкалы
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 3, y_top - 3), (x + bar_w + 40, y_bot + 20),
                      self._c_bg, -1)
        cv2.addWeighted(overlay, self._cfg.overlay_alpha * 0.6, frame,
                        1 - self._cfg.overlay_alpha * 0.6, 0, frame)

        # Зона поражения (зелёная часть)
        eff_y = y_bot - int((eff_range / max_range) * bar_h)
        cv2.rectangle(frame, (x, eff_y), (x + bar_w, y_bot), self._c_range_ok, -1)

        # Зона вне поражения (серая часть)
        cv2.rectangle(frame, (x, y_top), (x + bar_w, eff_y), self._c_range_out, -1)

        # Рамка
        cv2.rectangle(frame, (x, y_top), (x + bar_w, y_bot), self._c_info, 1)

        # Маркер текущей дистанции
        if state.distance_valid and state.distance_m > 0:
            dist_ratio = min(state.distance_m / max_range, 1.0)
            marker_y = y_bot - int(dist_ratio * bar_h)
            marker_y = max(y_top, min(y_bot, marker_y))

            marker_color = self._c_ready if state.in_effective_range else self._c_estop
            # Треугольный маркер
            pts = np.array([
                [x + bar_w + 2, marker_y],
                [x + bar_w + 12, marker_y - 6],
                [x + bar_w + 12, marker_y + 6]
            ], np.int32)
            cv2.fillPoly(frame, [pts], marker_color)

            # Цифра дистанции рядом с маркером
            dist_text = f"{state.distance_m:.0f}"
            cv2.putText(frame, dist_text, (x + bar_w + 15, marker_y + 5),
                        self._font, 0.35, marker_color, 1, cv2.LINE_AA)

        # Подписи шкалы
        # Верх: max range
        cv2.putText(frame, f"{int(max_range)}", (x - 2, y_top - 5),
                    self._font, 0.28, self._c_info, 1, cv2.LINE_AA)
        # Эфф. дальность
        cv2.putText(frame, f"{int(eff_range)}", (x + bar_w + 3, eff_y + 4),
                    self._font, 0.28, self._c_range_ok, 1, cv2.LINE_AA)
        # Низ: 0
        cv2.putText(frame, "0", (x + 2, y_bot + 14),
                    self._font, 0.28, self._c_info, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════
    # Нижний центр: индикатор готовности к стрельбе
    # ════════════════════════════════════════════════════════

    def _draw_readiness_indicator(self, frame: np.ndarray, state: SharedState):
        """Крупный индикатор ГОТОВ / НЕ ГОТОВ."""
        cx = self._cx
        y = self._h - 85

        if state.ready_to_fire:
            text = "ГОТОВ"
            color = self._c_ready
            # Пульсация яркости
            pulse = (math.sin(time.time() * 2.0 * math.pi * 1.5) + 1.0) / 2.0
            thickness = 2 if pulse > 0.5 else 1
        elif state.target_lock == TargetLockState.LOCKED and not state.distance_valid:
            text = "ЗАМЕРЬ ДИСТ"
            color = self._c_warning
            thickness = 1
        elif state.target_lock == TargetLockState.LOCKED and not state.in_effective_range:
            text = "ВНЕ ЗОНЫ"
            color = self._c_estop
            thickness = 1
        else:
            text = "НЕ ГОТОВ"
            color = self._c_not_ready
            thickness = 1

        text_size = cv2.getTextSize(text, self._font, 0.7, thickness)[0]
        tx = cx - text_size[0] // 2
        ty = y

        # Фон
        pad = 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (tx - pad, ty - text_size[1] - pad),
                      (tx + text_size[0] + pad, ty + pad), self._c_bg, -1)
        cv2.addWeighted(overlay, self._cfg.overlay_alpha, frame,
                        1 - self._cfg.overlay_alpha, 0, frame)

        # Рамка (цветная)
        cv2.rectangle(frame, (tx - pad, ty - text_size[1] - pad),
                      (tx + text_size[0] + pad, ty + pad), color, 1)

        cv2.putText(frame, text, (tx, ty), self._font, 0.7,
                    color, thickness, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════
    # Нижний центр: шкала углов платформы (вместо компаса)
    # ════════════════════════════════════════════════════════

    def _draw_platform_angles(self, frame: np.ndarray, state: SharedState):
        """Шкала углов платформы внизу по центру."""
        cx = self._cx
        y = self._h - 35
        width = 300

        # Фон
        overlay = frame.copy()
        cv2.rectangle(overlay, (cx - width // 2, y - 18),
                      (cx + width // 2, y + 14), self._c_bg, -1)
        cv2.addWeighted(overlay, self._cfg.overlay_alpha, frame,
                        1 - self._cfg.overlay_alpha, 0, frame)

        # Шкала yaw
        yaw = state.yaw_deg
        for deg_offset in range(-30, 31, 5):
            angle = yaw + deg_offset
            px = cx + int(deg_offset * (width / 60))

            if abs(deg_offset) % 10 == 0:
                cv2.line(frame, (px, y - 10), (px, y), self._c_info, 1)
                label = f"{angle:+.0f}"
                ts = cv2.getTextSize(label, self._font, 0.28, 1)[0]
                cv2.putText(frame, label, (px - ts[0] // 2, y + 12),
                            self._font, 0.28, self._c_info, 1, cv2.LINE_AA)
            else:
                cv2.line(frame, (px, y - 5), (px, y), self._c_info, 1)

        # Центральная метка
        cv2.drawMarker(frame, (cx, y - 13), self._c_lead,
                        cv2.MARKER_TRIANGLE_DOWN, 8, 1)

        # Pitch справа от шкалы
        pitch_text = f"E:{state.pitch_deg:+.1f}"
        cv2.putText(frame, pitch_text, (cx + width // 2 + 8, y + 2),
                    self._font, 0.32, self._c_info, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════
    # Индикатор стрельбы
    # ════════════════════════════════════════════════════════

    def _draw_firing_indicator(self, frame: np.ndarray, state: SharedState):
        """Индикатор активной стрельбы + счётчик выстрелов."""
        cx = self._cx
        y = self._h - 110

        # Мигающая метка ОГОНЬ
        blink = int(time.time() * 6) % 2 == 0
        if blink:
            text = f"ОГОНЬ"
            if state.shots_fired > 0:
                text += f" [{state.shots_fired}]"
            text_size = cv2.getTextSize(text, self._font, 0.5, 2)[0]
            tx = cx - text_size[0] // 2
            cv2.putText(frame, text, (tx, y), self._font, 0.5,
                        self._c_firing, 2, cv2.LINE_AA)

        # Маленькая полоска отдачи (если boost активен)
        if state.recoil_boost_active:
            bar_w = 60
            bar_h = 4
            bx = cx - bar_w // 2
            by = y + 8
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h),
                          self._c_warning, -1)

    # ════════════════════════════════════════════════════════
    # Предупреждения
    # ════════════════════════════════════════════════════════

    def _draw_estop_warning(self, frame: np.ndarray):
        """Мигающее предупреждение E-STOP."""
        if int(time.time() * 3) % 2 == 0:
            text = "!!! АВАРИЙНАЯ ОСТАНОВКА !!!"
            ts = cv2.getTextSize(text, self._font, 1.0, 2)[0]
            tx = (self._w - ts[0]) // 2
            ty = self._h // 2
            cv2.rectangle(frame, (tx - 20, ty - 40),
                          (tx + ts[0] + 20, ty + 15), (0, 0, 100), -1)
            cv2.putText(frame, text, (tx, ty), self._font, 1.0,
                        self._c_estop, 2, cv2.LINE_AA)

    def _draw_error_warning(self, frame: np.ndarray, message: str):
        """Предупреждение об ошибке."""
        text = f"ОШИБКА: {message}"
        ts = cv2.getTextSize(text, self._font, 0.6, 1)[0]
        tx = (self._w - ts[0]) // 2
        ty = self._h // 2 + 50
        cv2.rectangle(frame, (tx - 10, ty - 25),
                      (tx + ts[0] + 10, ty + 5), (0, 0, 80), -1)
        cv2.putText(frame, text, (tx, ty), self._font, 0.6,
                    self._c_warning, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════
    # Утилиты
    # ════════════════════════════════════════════════════════

    def _draw_text_with_bg(self, frame: np.ndarray, text: str,
                           pos: tuple, color: tuple,
                           scale: float = 0.4, thickness: int = 1):
        """Текст с полупрозрачным фоном."""
        ts = cv2.getTextSize(text, self._font, scale, thickness)[0]
        x, y = int(pos[0]), int(pos[1])
        cv2.rectangle(frame, (x - 2, y - ts[1] - 4),
                      (x + ts[0] + 2, y + 4), self._c_bg, -1)
        cv2.putText(frame, text, (x, y), self._font, scale,
                    color, thickness, cv2.LINE_AA)

    def _draw_dashed_line(self, frame: np.ndarray, pt1: tuple, pt2: tuple,
                          color: tuple, thickness: int = 1, dash_len: int = 8):
        """Пунктирная линия."""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if dist < 1:
            return
        dx = (x2 - x1) / dist
        dy = (y2 - y1) / dist
        n = int(dist / (dash_len * 2))
        for i in range(n + 1):
            s = int(i * dash_len * 2)
            e = min(int(s + dash_len), int(dist))
            sx, sy = int(x1 + dx * s), int(y1 + dy * s)
            ex, ey = int(x1 + dx * e), int(y1 + dy * e)
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

    @staticmethod
    def _heading_to_direction(heading_deg: float) -> str:
        """Преобразовать курс в буквенное обозначение направления."""
        # Нормализуем в 0..360
        h = heading_deg % 360
        directions = [
            (0, "С"), (22.5, "ССВ"), (45, "СВ"), (67.5, "ВСВ"),
            (90, "В"), (112.5, "ВЮВ"), (135, "ЮВ"), (157.5, "ЮЮВ"),
            (180, "Ю"), (202.5, "ЮЮЗ"), (225, "ЮЗ"), (247.5, "ЗЮЗ"),
            (270, "З"), (292.5, "ЗСЗ"), (315, "СЗ"), (337.5, "ССЗ"),
        ]
        # Находим ближайшее направление
        best = "С"
        best_diff = 360
        for angle, name in directions:
            diff = abs(h - angle)
            if diff > 180:
                diff = 360 - diff
            if diff < best_diff:
                best_diff = diff
                best = name
        return best
