# -*- coding: utf-8 -*-
"""
Драйвер лазерного дальномера Meskernel TS1224.

Подключение по UART. Дальномер возвращает дистанцию
в метрах по запросу или в непрерывном режиме.

Протокол TS1224 (типичный для китайских LRF-модулей):
  Команда замера: 0x55 0x55 0x01 0x00 0xAA
  Ответ: заголовок + 2 байта дистанции (мм или 0.1м)

ВНИМАНИЕ: Точный протокол зависит от прошивки модуля.
Проверьте документацию вашего конкретного TS1224 и
скорректируйте _parse_response() при необходимости.
"""

import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional

try:
    import serial
except ImportError:
    serial = None

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Состояние дальномера
# ════════════════════════════════════════════════════════════════

@dataclass
class RangefinderState:
    """Текущее состояние лазерного дальномера."""
    distance_m: float = 0.0        # Последняя измеренная дистанция (метры)
    is_valid: bool = False         # Последнее измерение валидно
    timestamp: float = 0.0        # Время последнего измерения
    measurements_count: int = 0   # Общее количество замеров
    errors_count: int = 0         # Количество ошибок
    is_connected: bool = False    # Порт открыт


# ════════════════════════════════════════════════════════════════
# Драйвер дальномера
# ════════════════════════════════════════════════════════════════

class RangefinderDriver:
    """
    Драйвер лазерного дальномера Meskernel TS1224.

    Поддерживает два режима:
    1. Одиночный замер (fire) — по команде с джойстика
    2. Непрерывный опрос (continuous) — в фоновом потоке
    """

    # ── Команды протокола TS1224 ──
    # Стандартные команды (могут отличаться в зависимости от прошивки)
    CMD_SINGLE_MEASURE = bytes([0x55, 0x55, 0x01, 0x00, 0xAA])
    CMD_CONTINUOUS_ON = bytes([0x55, 0x55, 0x02, 0x00, 0xAA])
    CMD_CONTINUOUS_OFF = bytes([0x55, 0x55, 0x03, 0x00, 0xAA])
    CMD_LASER_ON = bytes([0x55, 0x55, 0x04, 0x00, 0xAA])
    CMD_LASER_OFF = bytes([0x55, 0x55, 0x05, 0x00, 0xAA])

    # Заголовок ответа
    RESPONSE_HEADER = bytes([0xAA, 0xAA])

    def __init__(self, port: str = "/dev/ttyAMA0", baudrate: int = 115200,
                 timeout_sec: float = 0.1,
                 min_range_m: float = 3.0, max_range_m: float = 1200.0,
                 poll_rate_hz: int = 10):
        """
        Args:
            port: UART-порт (/dev/ttyAMA0 на RPi5)
            baudrate: Скорость порта
            timeout_sec: Таймаут чтения
            min_range_m: Минимальная дальность (м)
            max_range_m: Максимальная дальность (м)
            poll_rate_hz: Частота опроса в непрерывном режиме
        """
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout_sec
        self._min_range = min_range_m
        self._max_range = max_range_m
        self._poll_interval = 1.0 / poll_rate_hz

        self._serial: Optional[serial.Serial] = None
        self._lock = threading.Lock()

        # Фоновый поток непрерывного опроса
        self._continuous_thread: Optional[threading.Thread] = None
        self._continuous_running = False

        # Публичное состояние
        self.state = RangefinderState()

        logger.info(
            f"RangefinderDriver: порт={port}, скорость={baudrate}, "
            f"дальность={min_range_m}-{max_range_m} м"
        )

    # ── Инициализация / завершение ──────────────────────────

    def open(self) -> bool:
        """
        Открыть UART-порт. Возвращает True при успехе.
        """
        if serial is None:
            logger.error("Библиотека pyserial не установлена!")
            return False

        try:
            self._serial = serial.Serial(
                port=self._port,
                baudrate=self._baudrate,
                timeout=self._timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )

            # Очистка буферов
            self._serial.reset_input_buffer()
            self._serial.reset_output_buffer()

            self.state.is_connected = True
            logger.info(f"Дальномер подключён: {self._port}")
            return True

        except Exception as e:
            logger.error(f"Ошибка подключения дальномера: {e}")
            return False

    def close(self):
        """Остановить опрос и закрыть порт."""
        self.stop_continuous()
        if self._serial and self._serial.is_open:
            try:
                # Выключить лазер перед закрытием
                self._send_command(self.CMD_LASER_OFF)
            except Exception:
                pass
            self._serial.close()
        self.state.is_connected = False
        logger.info("Дальномер отключён")

    # ── Одиночный замер ─────────────────────────────────────

    def fire(self) -> Optional[float]:
        """
        Выполнить одиночный замер дистанции.

        Returns:
            Дистанция в метрах или None при ошибке
        """
        if not self._serial or not self._serial.is_open:
            logger.warning("Дальномер: порт не открыт")
            return None

        with self._lock:
            try:
                # Очистка входного буфера
                self._serial.reset_input_buffer()

                # Отправка команды замера
                self._send_command(self.CMD_SINGLE_MEASURE)

                # Чтение ответа (с увеличенным таймаутом для одиночного замера)
                response = self._read_response(timeout=2.0)

                if response is not None:
                    distance = self._parse_response(response)
                    if distance is not None and self._min_range <= distance <= self._max_range:
                        self.state.distance_m = distance
                        self.state.is_valid = True
                        self.state.timestamp = time.time()
                        self.state.measurements_count += 1
                        logger.info(f"Дальномер: {distance:.1f} м")
                        return distance
                    else:
                        logger.warning(f"Дальномер: невалидное значение {distance}")
                        self.state.is_valid = False
                        self.state.errors_count += 1
                else:
                    logger.warning("Дальномер: нет ответа")
                    self.state.is_valid = False
                    self.state.errors_count += 1

            except Exception as e:
                logger.error(f"Дальномер: ошибка замера — {e}")
                self.state.errors_count += 1

        return None

    # ── Непрерывный опрос ───────────────────────────────────

    def start_continuous(self):
        """Запустить непрерывный опрос дальномера в фоновом потоке."""
        if self._continuous_running:
            return

        self._continuous_running = True
        self._continuous_thread = threading.Thread(
            target=self._continuous_loop,
            name="RangefinderPoll",
            daemon=True
        )
        self._continuous_thread.start()
        logger.info(f"Непрерывный опрос дальномера запущен ({1/self._poll_interval:.0f} Гц)")

    def stop_continuous(self):
        """Остановить непрерывный опрос."""
        self._continuous_running = False
        if self._continuous_thread and self._continuous_thread.is_alive():
            self._continuous_thread.join(timeout=2.0)
        logger.info("Непрерывный опрос дальномера остановлен")

    def _continuous_loop(self):
        """Фоновый поток непрерывного опроса."""
        while self._continuous_running:
            self.fire()
            time.sleep(self._poll_interval)

    # ── Управление лазером ──────────────────────────────────

    def laser_on(self):
        """Включить лазерный указатель."""
        self._send_command(self.CMD_LASER_ON)
        logger.debug("Лазер: ВКЛ")

    def laser_off(self):
        """Выключить лазерный указатель."""
        self._send_command(self.CMD_LASER_OFF)
        logger.debug("Лазер: ВЫКЛ")

    # ── Низкоуровневый UART ─────────────────────────────────

    def _send_command(self, cmd: bytes):
        """Отправить команду по UART."""
        if self._serial and self._serial.is_open:
            self._serial.write(cmd)
            self._serial.flush()

    def _read_response(self, timeout: float = 1.0) -> Optional[bytes]:
        """
        Прочитать ответ дальномера.

        Ищет заголовок 0xAA 0xAA, затем читает тело ответа.

        Args:
            timeout: Максимальное время ожидания (сек)

        Returns:
            Байты ответа или None
        """
        if not self._serial or not self._serial.is_open:
            return None

        start_time = time.time()
        buffer = bytearray()

        while (time.time() - start_time) < timeout:
            if self._serial.in_waiting > 0:
                data = self._serial.read(self._serial.in_waiting)
                buffer.extend(data)

                # Ищем заголовок ответа
                header_pos = buffer.find(self.RESPONSE_HEADER)
                if header_pos >= 0:
                    # Ждём минимум 7 байт после заголовка (типичная длина ответа)
                    if len(buffer) >= header_pos + 7:
                        return bytes(buffer[header_pos:header_pos + 7])
            else:
                time.sleep(0.005)

        return None

    def _parse_response(self, data: bytes) -> Optional[float]:
        """
        Разобрать ответ дальномера и извлечь дистанцию.

        Типичный формат ответа TS1224:
            Байт 0-1: Заголовок (0xAA 0xAA)
            Байт 2:   Тип ответа
            Байт 3-4: Дистанция (старший + младший байт, в 0.1 м)
            Байт 5:   Статус (0x00 = OK)
            Байт 6:   Контрольная сумма

        ВНИМАНИЕ: Формат может отличаться! Проверьте документацию.
        """
        if data is None or len(data) < 7:
            return None

        try:
            # Проверка заголовка
            if data[0] != 0xAA or data[1] != 0xAA:
                return None

            # Проверка статуса
            status = data[5]
            if status != 0x00:
                logger.debug(f"Дальномер: статус ошибки 0x{status:02X}")
                return None

            # Извлечение дистанции (в 0.1 метра)
            distance_raw = (data[3] << 8) | data[4]
            distance_m = distance_raw * 0.1

            return distance_m

        except Exception as e:
            logger.error(f"Дальномер: ошибка разбора ответа — {e}")
            return None

    # ── Утилиты ─────────────────────────────────────────────

    @property
    def last_distance(self) -> float:
        """Последняя измеренная дистанция (метры)."""
        return self.state.distance_m

    @property
    def is_valid(self) -> bool:
        """Последнее измерение валидно."""
        return self.state.is_valid

    @property
    def age(self) -> float:
        """Возраст последнего измерения (секунды)."""
        if self.state.timestamp == 0:
            return float('inf')
        return time.time() - self.state.timestamp
