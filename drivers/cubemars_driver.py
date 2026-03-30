# -*- coding: utf-8 -*-
"""
Драйвер моторов Cubemars AK64-80.

Реализует MIT Mode протокол управления через CAN-шину.
Каждый мотор управляется позицией с заданными коэффициентами kp/kd.

Протокол MIT Mode (8 байт команда):
  Байты 0-1: Позиция (16 бит, масштабированная)
  Байты 2-3: Скорость (12 бит) + KP старшие 4 бита
  Байты 4-5: KP младшие 8 бит + KD (12 бит)
  Байты 6-7: Момент (12 бит)

Протокол MIT Mode (8 байт ответ):
  Байт 0:    ID мотора
  Байты 1-2: Позиция (16 бит)
  Байты 3-4: Скорость (12 бит) + Ток старшие 4 бита
  Байт 5:    Ток младшие 8 бит
"""

import time
import math
import logging
import threading
from dataclasses import dataclass
from typing import Optional, Dict

try:
    import can
except ImportError:
    can = None  # Позволяет импортировать модуль для тестирования без python-can

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Параметры мотора AK64-80
# ════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AK80_64_Limits:
    """
    Пределы параметров мотора Cubemars AK80-64 (24V) для MIT-кодирования.
    Источник: https://www.cubemars.com/ru/product/ak80-64-kv80-robotic-actuator.html
    """
    P_MIN: float = -12.5       # Позиция мин (рад)
    P_MAX: float = 12.5        # Позиция макс (рад)
    V_MIN: float = -45.0       # Скорость мин (рад/с)
    V_MAX: float = 45.0        # Скорость макс (рад/с)
    KP_MIN: float = 0.0        # Жёсткость мин
    KP_MAX: float = 500.0      # Жёсткость макс
    KD_MIN: float = 0.0        # Демпфирование мин
    KD_MAX: float = 5.0        # Демпфирование макс
    T_MIN: float = -18.0       # Момент мин (Нм)
    T_MAX: float = 18.0        # Момент макс (Нм)
    GEAR_RATIO: float = 64.0   # Передаточное число (64:1)


# Глобальный экземпляр пределов
LIMITS = AK80_64_Limits()


# ════════════════════════════════════════════════════════════════
# Состояние мотора
# ════════════════════════════════════════════════════════════════

@dataclass
class MotorState:
    """Текущее состояние мотора (из ответа по CAN)."""
    motor_id: int = 0
    position_rad: float = 0.0      # Позиция (рад)
    velocity_rads: float = 0.0     # Скорость (рад/с)
    current_a: float = 0.0         # Ток (А)
    timestamp: float = 0.0         # Время последнего обновления
    is_enabled: bool = False
    has_error: bool = False

    @property
    def position_deg(self) -> float:
        """Позиция в градусах."""
        return math.degrees(self.position_rad)

    @property
    def velocity_dps(self) -> float:
        """Скорость в градусах/сек."""
        return math.degrees(self.velocity_rads)


# ════════════════════════════════════════════════════════════════
# Утилиты кодирования MIT-протокола
# ════════════════════════════════════════════════════════════════

def _float_to_uint(value: float, v_min: float, v_max: float, bits: int) -> int:
    """Преобразование float в unsigned int с масштабированием."""
    value = max(min(value, v_max), v_min)
    span = v_max - v_min
    return int((value - v_min) * ((1 << bits) - 1) / span)


def _uint_to_float(value: int, v_min: float, v_max: float, bits: int) -> float:
    """Преобразование unsigned int обратно в float."""
    span = v_max - v_min
    return value * span / ((1 << bits) - 1) + v_min


# ════════════════════════════════════════════════════════════════
# Драйвер одного мотора
# ════════════════════════════════════════════════════════════════

class CubemarsMotor:
    """
    Драйвер одного мотора Cubemars AK64-80.
    Общается через CAN-шину по MIT Mode протоколу.
    """

    # Специальные CAN-команды управления режимом мотора
    CMD_ENTER_MODE = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
    CMD_EXIT_MODE = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])
    CMD_SET_ZERO = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE])

    def __init__(self, bus: 'can.Bus', motor_id: int):
        """
        Инициализация драйвера мотора.

        Args:
            bus: Экземпляр python-can Bus
            motor_id: CAN ID мотора (1-32)
        """
        self._bus = bus
        self._motor_id = motor_id
        self._limits = LIMITS
        self._lock = threading.Lock()

        # Публичное состояние мотора
        self.state = MotorState(motor_id=motor_id)

        logger.debug(f"Мотор {motor_id}: драйвер создан")

    @property
    def motor_id(self) -> int:
        return self._motor_id

    @property
    def is_enabled(self) -> bool:
        return self.state.is_enabled

    # ── Управление режимом ──────────────────────────────────

    def enable(self) -> bool:
        """Включить мотор (войти в MIT Mode). Возвращает True при успехе."""
        logger.info(f"Мотор {self._motor_id}: включение...")
        try:
            self._send_raw(self.CMD_ENTER_MODE)
            time.sleep(0.01)
            self._read_response(timeout=0.1)
            self.state.is_enabled = True
            logger.info(f"Мотор {self._motor_id}: ВКЛЮЧЁН")
            return True
        except Exception as e:
            logger.error(f"Мотор {self._motor_id}: ошибка включения — {e}")
            return False

    def disable(self) -> bool:
        """Выключить мотор (свободное вращение). Возвращает True при успехе."""
        logger.info(f"Мотор {self._motor_id}: выключение...")
        try:
            self._send_raw(self.CMD_EXIT_MODE)
            time.sleep(0.01)
            self.state.is_enabled = False
            logger.info(f"Мотор {self._motor_id}: ВЫКЛЮЧЕН")
            return True
        except Exception as e:
            logger.error(f"Мотор {self._motor_id}: ошибка выключения — {e}")
            return False

    def set_zero(self) -> bool:
        """Установить текущую позицию как нулевую."""
        logger.info(f"Мотор {self._motor_id}: установка нуля...")
        try:
            self._send_raw(self.CMD_SET_ZERO)
            time.sleep(0.01)
            self._read_response(timeout=0.1)
            self.state.position_rad = 0.0
            logger.info(f"Мотор {self._motor_id}: ноль установлен")
            return True
        except Exception as e:
            logger.error(f"Мотор {self._motor_id}: ошибка установки нуля — {e}")
            return False

    def emergency_stop(self):
        """Аварийная остановка — немедленное отключение мотора."""
        logger.warning(f"Мотор {self._motor_id}: АВАРИЙНАЯ ОСТАНОВКА!")
        self.disable()

    # ── Команды управления ──────────────────────────────────

    def send_mit_command(self, position_rad: float, velocity_rads: float,
                         kp: float, kd: float, torque_nm: float) -> MotorState:
        """
        Отправить MIT Mode команду мотору.

        Мотор вычисляет:
            torque_out = kp*(pos_cmd - pos_actual) + kd*(vel_cmd - vel_actual) + torque_ff

        Args:
            position_rad: Целевая позиция (рад)
            velocity_rads: Целевая скорость (рад/с)
            kp: Коэффициент жёсткости
            kd: Коэффициент демпфирования
            torque_nm: Момент прямой связи (Нм)

        Returns:
            Обновлённое состояние мотора
        """
        if not self.state.is_enabled:
            logger.warning(f"Мотор {self._motor_id}: команда отклонена — мотор не включён")
            return self.state

        lim = self._limits

        # Масштабирование в unsigned int
        p_int = _float_to_uint(position_rad, lim.P_MIN, lim.P_MAX, 16)
        v_int = _float_to_uint(velocity_rads, lim.V_MIN, lim.V_MAX, 12)
        kp_int = _float_to_uint(kp, lim.KP_MIN, lim.KP_MAX, 12)
        kd_int = _float_to_uint(kd, lim.KD_MIN, lim.KD_MAX, 12)
        t_int = _float_to_uint(torque_nm, lim.T_MIN, lim.T_MAX, 12)

        # Упаковка в 8 байт по протоколу MIT
        data = bytes([
            (p_int >> 8) & 0xFF,                              # Байт 0
            p_int & 0xFF,                                      # Байт 1
            (v_int >> 4) & 0xFF,                              # Байт 2
            ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F),  # Байт 3
            kp_int & 0xFF,                                     # Байт 4
            (kd_int >> 4) & 0xFF,                             # Байт 5
            ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F),  # Байт 6
            t_int & 0xFF                                       # Байт 7
        ])

        try:
            self._send_raw(data)
            self._read_response(timeout=0.005)
        except Exception as e:
            logger.error(f"Мотор {self._motor_id}: ошибка отправки — {e}")
            self.state.has_error = True

        return self.state

    def send_position(self, position_rad: float, kp: float, kd: float,
                      velocity_ff: float = 0.0, torque_ff: float = 0.0) -> MotorState:
        """
        Упрощённая команда позиционирования.

        Args:
            position_rad: Целевая позиция (рад)
            kp: Жёсткость позиционирования
            kd: Демпфирование
            velocity_ff: Прямая связь по скорости (рад/с)
            torque_ff: Прямая связь по моменту (Нм)
        """
        return self.send_mit_command(position_rad, velocity_ff, kp, kd, torque_ff)

    def hold(self) -> MotorState:
        """Удерживать текущую позицию."""
        return self.send_position(self.state.position_rad, kp=30.0, kd=3.0)

    # ── Низкоуровневый CAN ──────────────────────────────────

    def _send_raw(self, data: bytes):
        """Отправить сырые данные по CAN."""
        with self._lock:
            msg = can.Message(
                arbitration_id=self._motor_id,
                data=data,
                is_extended_id=False
            )
            self._bus.send(msg)

    def _read_response(self, timeout: float = 0.01) -> Optional[MotorState]:
        """
        Прочитать и разобрать ответ мотора из CAN-шины.

        Формат ответа (8 байт):
            Байт 0:    ID мотора
            Байты 1-2: Позиция (16 бит)
            Байты 3-4: Скорость (12 бит) | Ток старшие 4 бита
            Байт 5:    Ток младшие 8 бит
        """
        try:
            response = self._bus.recv(timeout=timeout)
            if response is None or len(response.data) < 6:
                return None

            d = response.data
            lim = self._limits

            # Разбор полей
            p_int = (d[1] << 8) | d[2]
            v_int = (d[3] << 4) | (d[4] >> 4)
            i_int = ((d[4] & 0x0F) << 8) | d[5]

            self.state.motor_id = d[0]
            self.state.position_rad = _uint_to_float(p_int, lim.P_MIN, lim.P_MAX, 16)
            self.state.velocity_rads = _uint_to_float(v_int, lim.V_MIN, lim.V_MAX, 12)
            self.state.current_a = _uint_to_float(i_int, lim.T_MIN, lim.T_MAX, 12)
            self.state.timestamp = time.time()
            self.state.has_error = False

            return self.state

        except Exception as e:
            logger.error(f"Мотор {self._motor_id}: ошибка чтения — {e}")
            return None


# ════════════════════════════════════════════════════════════════
# Менеджер CAN-шины
# ════════════════════════════════════════════════════════════════

class CANBusManager:
    """
    Управляет CAN-шиной и набором моторов.
    Обеспечивает подключение, отключение и аварийную остановку.
    """

    def __init__(self, interface: str = "socketcan",
                 channel: str = "can0", bitrate: int = 1_000_000):
        """
        Args:
            interface: Тип CAN-интерфейса (socketcan для Linux/RPi)
            channel: Имя канала (can0, can1)
            bitrate: Скорость CAN-шины (1 000 000 для Cubemars)
        """
        self._interface = interface
        self._channel = channel
        self._bitrate = bitrate
        self._bus: Optional['can.Bus'] = None
        self._motors: Dict[int, CubemarsMotor] = {}

        logger.info(f"CANBusManager: {interface}:{channel} @ {bitrate} бит/с")

    @property
    def is_connected(self) -> bool:
        return self._bus is not None

    def connect(self) -> bool:
        """Открыть CAN-соединение. Возвращает True при успехе."""
        if can is None:
            logger.error("Библиотека python-can не установлена!")
            return False
        try:
            self._bus = can.Bus(
                interface=self._interface,
                channel=self._channel,
                bitrate=self._bitrate
            )
            logger.info(f"CAN-шина подключена: {self._channel}")
            return True
        except Exception as e:
            logger.error(f"Ошибка подключения CAN: {e}")
            logger.error(
                f"Убедитесь, что интерфейс настроен:\n"
                f"  sudo ip link set {self._channel} up type can bitrate {self._bitrate}\n"
                f"  sudo ifconfig {self._channel} up"
            )
            return False

    def disconnect(self):
        """Отключить все моторы и закрыть CAN-шину."""
        for motor in self._motors.values():
            if motor.is_enabled:
                motor.disable()
        if self._bus:
            self._bus.shutdown()
            self._bus = None
            logger.info("CAN-шина отключена")

    def add_motor(self, motor_id: int) -> CubemarsMotor:
        """
        Добавить мотор в менеджер.

        Args:
            motor_id: CAN ID мотора

        Returns:
            Экземпляр CubemarsMotor
        """
        if self._bus is None:
            raise RuntimeError("CAN-шина не подключена. Вызовите connect() сначала.")
        motor = CubemarsMotor(self._bus, motor_id)
        self._motors[motor_id] = motor
        logger.info(f"Мотор добавлен: CAN ID={motor_id}")
        return motor

    def get_motor(self, motor_id: int) -> Optional[CubemarsMotor]:
        """Получить мотор по ID."""
        return self._motors.get(motor_id)

    def enable_all(self) -> bool:
        """Включить все моторы."""
        ok = True
        for motor in self._motors.values():
            if not motor.enable():
                ok = False
        return ok

    def disable_all(self):
        """Выключить все моторы."""
        for motor in self._motors.values():
            motor.disable()

    def emergency_stop_all(self):
        """Аварийная остановка всех моторов."""
        logger.warning("АВАРИЙНАЯ ОСТАНОВКА ВСЕХ МОТОРОВ!")
        for motor in self._motors.values():
            motor.emergency_stop()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
