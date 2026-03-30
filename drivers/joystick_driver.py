# -*- coding: utf-8 -*-
"""
Драйвер джойстика Logitech X56 HOTAS.

Читает оси, кнопки и hat-переключатели через pygame.
Поддерживает мёртвую зону, кривую чувствительности,
инверсию осей и callback-и на нажатие кнопок.

Для платформы слежения используется только правая ручка (РУС):
  - Ось X (yaw): горизонтальное наведение
  - Ось Y (pitch): вертикальное наведение
  - Колёсико (zoom): управление оптическим зумом камеры
  - Кнопки: режимы, дальномер, захват цели, E-STOP
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Callable, Optional, List

try:
    import pygame
except ImportError:
    pygame = None

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Состояние джойстика
# ════════════════════════════════════════════════════════════════

@dataclass
class JoystickState:
    """Текущее состояние всех входов джойстика."""
    # Сырые значения осей (-1.0 .. 1.0)
    axes: List[float] = field(default_factory=list)
    # Состояния кнопок (True/False)
    buttons: List[bool] = field(default_factory=list)
    # Hat-переключатели ((x, y) каждый -1, 0 или 1)
    hats: List[tuple] = field(default_factory=list)

    # Обработанные значения для управления платформой
    yaw: float = 0.0           # Горизонт (-1.0 .. 1.0)
    pitch: float = 0.0         # Вертикаль (-1.0 .. 1.0)
    zoom_delta: float = 0.0    # Дельта зума (-1.0 .. 1.0)

    # Метаданные
    connected: bool = False
    device_name: str = ""
    timestamp: float = 0.0


# ════════════════════════════════════════════════════════════════
# Драйвер джойстика
# ════════════════════════════════════════════════════════════════

class JoystickDriver:
    """
    Драйвер джойстика с поддержкой кривых чувствительности,
    мёртвых зон и callback-ов на кнопки.
    """

    def __init__(self, device_index: int = 0,
                 axis_yaw: int = 0, axis_pitch: int = 1, axis_zoom: int = 5,
                 invert_yaw: bool = False, invert_pitch: bool = True,
                 deadzone: float = 0.08, sensitivity_exponent: float = 2.0,
                 speed_multiplier: float = 1.0):
        """
        Args:
            device_index: Индекс устройства pygame (0 — первый джойстик)
            axis_yaw: Номер оси горизонтального наведения
            axis_pitch: Номер оси вертикального наведения
            axis_zoom: Номер оси колёсика зума
            invert_yaw: Инвертировать ось yaw
            invert_pitch: Инвертировать ось pitch
            deadzone: Мёртвая зона (0.0 .. 1.0)
            sensitivity_exponent: Экспонента кривой чувствительности
            speed_multiplier: Общий множитель скорости
        """
        self._device_index = device_index
        self._axis_yaw = axis_yaw
        self._axis_pitch = axis_pitch
        self._axis_zoom = axis_zoom
        self._invert_yaw = invert_yaw
        self._invert_pitch = invert_pitch
        self._deadzone = deadzone
        self._exponent = sensitivity_exponent
        self._speed_mult = speed_multiplier

        self._joystick: Optional[pygame.joystick.Joystick] = None
        self._initialized = False

        # Callback-и: кнопка → функция (вызывается по нарастающему фронту)
        self._button_callbacks: Dict[int, Callable] = {}

        # Публичное состояние
        self.state = JoystickState()

        logger.info(
            f"JoystickDriver: устройство={device_index}, "
            f"yaw=ось{axis_yaw}, pitch=ось{axis_pitch}, zoom=ось{axis_zoom}"
        )

    # ── Инициализация / завершение ──────────────────────────

    def initialize(self) -> bool:
        """
        Инициализировать pygame и подключиться к джойстику.
        Возвращает True при успехе.
        """
        if pygame is None:
            logger.error("Библиотека pygame не установлена!")
            return False

        try:
            if not pygame.get_init():
                pygame.init()
            pygame.joystick.init()

            count = pygame.joystick.get_count()
            logger.info(f"Найдено джойстиков: {count}")

            if count == 0:
                logger.error("Джойстики не найдены! Подключите Logitech X56 HOTAS.")
                return False

            # Вывод списка всех джойстиков
            for i in range(count):
                js = pygame.joystick.Joystick(i)
                js.init()
                logger.info(
                    f"  [{i}] {js.get_name()} — "
                    f"Осей: {js.get_numaxes()}, "
                    f"Кнопок: {js.get_numbuttons()}, "
                    f"Hat: {js.get_numhats()}"
                )

            if self._device_index >= count:
                logger.error(
                    f"Устройство {self._device_index} недоступно "
                    f"(всего {count} джойстиков)"
                )
                return False

            # Подключаемся к выбранному джойстику
            self._joystick = pygame.joystick.Joystick(self._device_index)
            self._joystick.init()

            n_axes = self._joystick.get_numaxes()
            n_buttons = self._joystick.get_numbuttons()
            n_hats = self._joystick.get_numhats()

            self.state.device_name = self._joystick.get_name()
            self.state.connected = True
            self.state.axes = [0.0] * n_axes
            self.state.buttons = [False] * n_buttons
            self.state.hats = [(0, 0)] * n_hats

            self._initialized = True

            logger.info(f"Джойстик подключён: {self.state.device_name}")
            logger.info(f"  Осей: {n_axes}, Кнопок: {n_buttons}, Hat: {n_hats}")

            # Проверка валидности назначенных осей
            if self._axis_yaw >= n_axes:
                logger.warning(f"Ось yaw={self._axis_yaw} вне диапазона, используется 0")
                self._axis_yaw = 0
            if self._axis_pitch >= n_axes:
                logger.warning(f"Ось pitch={self._axis_pitch} вне диапазона, используется 1")
                self._axis_pitch = min(1, n_axes - 1)
            if self._axis_zoom >= n_axes:
                logger.warning(f"Ось zoom={self._axis_zoom} вне диапазона, зум отключён")
                self._axis_zoom = -1

            return True

        except Exception as e:
            logger.error(f"Ошибка инициализации джойстика: {e}")
            return False

    def shutdown(self):
        """Завершить работу с джойстиком."""
        self.state.connected = False
        self._initialized = False
        if self._joystick:
            self._joystick.quit()
            self._joystick = None
        pygame.joystick.quit()
        logger.info("Джойстик отключён")

    # ── Callback-и на кнопки ────────────────────────────────

    def set_button_callback(self, button_index: int, callback: Callable):
        """
        Установить callback на нажатие кнопки (нарастающий фронт).

        Args:
            button_index: Номер кнопки
            callback: Функция без аргументов
        """
        self._button_callbacks[button_index] = callback
        logger.debug(f"Callback установлен на кнопку {button_index}")

    def is_button_pressed(self, button_index: int) -> bool:
        """Проверить, удерживается ли кнопка прямо сейчас."""
        if 0 <= button_index < len(self.state.buttons):
            return self.state.buttons[button_index]
        return False

    # ── Обновление состояния ────────────────────────────────

    def update(self) -> JoystickState:
        """
        Прочитать текущее состояние джойстика.
        Должен вызываться в каждой итерации главного цикла.

        Returns:
            Обновлённый JoystickState
        """
        if not self._initialized or not self._joystick:
            return self.state

        try:
            # Обработка очереди событий pygame
            pygame.event.pump()

            # Чтение всех осей
            for i in range(self._joystick.get_numaxes()):
                self.state.axes[i] = self._joystick.get_axis(i)

            # Чтение кнопок (с детекцией нарастающего фронта)
            prev_buttons = self.state.buttons.copy()
            for i in range(self._joystick.get_numbuttons()):
                self.state.buttons[i] = bool(self._joystick.get_button(i))

            # Чтение hat-переключателей
            for i in range(self._joystick.get_numhats()):
                self.state.hats[i] = self._joystick.get_hat(i)

            # ── Обработка осей управления ──

            # Yaw (горизонт)
            raw_yaw = self.state.axes[self._axis_yaw]
            if self._invert_yaw:
                raw_yaw = -raw_yaw
            self.state.yaw = self._apply_curve(raw_yaw) * self._speed_mult

            # Pitch (вертикаль)
            raw_pitch = self.state.axes[self._axis_pitch]
            if self._invert_pitch:
                raw_pitch = -raw_pitch
            self.state.pitch = self._apply_curve(raw_pitch) * self._speed_mult

            # Zoom (колёсико)
            if self._axis_zoom >= 0 and self._axis_zoom < len(self.state.axes):
                self.state.zoom_delta = self.state.axes[self._axis_zoom]
            else:
                self.state.zoom_delta = 0.0

            self.state.timestamp = time.time()

            # ── Callback-и на нарастающий фронт кнопок ──
            for btn_idx, callback in self._button_callbacks.items():
                if btn_idx < len(self.state.buttons):
                    if self.state.buttons[btn_idx] and not prev_buttons[btn_idx]:
                        try:
                            callback()
                        except Exception as e:
                            logger.error(f"Ошибка в callback кнопки {btn_idx}: {e}")

            return self.state

        except Exception as e:
            logger.error(f"Ошибка чтения джойстика: {e}")
            self.state.connected = False
            return self.state

    # ── Кривая чувствительности ─────────────────────────────

    def _apply_curve(self, value: float) -> float:
        """
        Применить мёртвую зону и экспоненциальную кривую чувствительности.

        При exponent=1.0 — линейная зависимость.
        При exponent=2.0 — квадратичная (больше точности в центре).
        При exponent=3.0 — кубическая (ещё больше точности).

        Args:
            value: Сырое значение оси (-1.0 .. 1.0)

        Returns:
            Обработанное значение (-1.0 .. 1.0)
        """
        if abs(value) < self._deadzone:
            return 0.0

        sign = 1.0 if value > 0 else -1.0
        normalized = (abs(value) - self._deadzone) / (1.0 - self._deadzone)
        normalized = min(normalized, 1.0)

        curved = math.pow(normalized, self._exponent)
        return sign * curved

    # ── Утилиты ─────────────────────────────────────────────

    def get_control_input(self) -> tuple:
        """
        Получить обработанные значения для управления платформой.

        Returns:
            (yaw, pitch) — каждое от -1.0 до 1.0
        """
        return self.state.yaw, self.state.pitch

    def get_zoom_input(self) -> float:
        """
        Получить значение оси зума.

        Returns:
            Дельта зума от -1.0 до 1.0
        """
        return self.state.zoom_delta
