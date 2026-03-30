#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║          TRACKING PLATFORM — Платформа слежения             ║
║                                                              ║
║  Роботизированная поворотная платформа для автоматического   ║
║  слежения и предсказания траектории воздушных объектов       ║
║                                                              ║
║  Raspberry Pi 5 + Hailo-10H + Arducam PTZ + Cubemars AK64  ║
╚══════════════════════════════════════════════════════════════╝

Точка входа в приложение.

Использование:
    python3 main.py                       # Запуск с конфигом по умолчанию
    python3 main.py -c custom.yaml        # Запуск с другим конфигом
    python3 main.py --test-joystick       # Тест джойстика
    python3 main.py --test-motors         # Тест моторов
"""

import sys
import os
import logging
import argparse

# Добавляем корень проекта в PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.app_config import load_config
from core.platform_controller import PlatformController


def setup_logging(log_level: str = "INFO", log_to_file: bool = True,
                  log_file: str = "logs/platform.log"):
    """
    Настроить систему логирования.

    Args:
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Писать лог в файл
        log_file: Путь к файлу лога
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_to_file:
        # Создаём директорию для логов
        log_dir = os.path.join(PROJECT_ROOT, os.path.dirname(log_file))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(PROJECT_ROOT, log_file)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-5s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )


def test_joystick():
    """Режим тестирования джойстика — показывает все оси и кнопки."""
    from drivers.joystick_driver import JoystickDriver
    import time

    print("=" * 60)
    print("  ТЕСТ ДЖОЙСТИКА")
    print("  Двигайте стик и нажимайте кнопки")
    print("  Ctrl+C для выхода")
    print("=" * 60)

    joy = JoystickDriver(device_index=0)
    if not joy.initialize():
        print("ОШИБКА: джойстик не найден!")
        return

    print(f"\nПодключён: {joy.state.device_name}")
    print(f"Осей: {len(joy.state.axes)}, Кнопок: {len(joy.state.buttons)}\n")

    try:
        while True:
            joy.update()

            axes = " | ".join([f"A{i}:{v:+.3f}" for i, v in enumerate(joy.state.axes)])
            pressed = [f"B{i}" for i, v in enumerate(joy.state.buttons) if v]
            buttons = ", ".join(pressed) if pressed else "—"
            hats = " | ".join([f"H{i}:{h}" for i, h in enumerate(joy.state.hats)])

            print(f"\rОси: [{axes}]  Кнопки: [{buttons}]  Hat: [{hats}]    ",
                  end="", flush=True)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nТест завершён.")
    finally:
        joy.shutdown()


def test_motors(config_path: str):
    """Режим тестирования моторов — малые движения для проверки связи."""
    import time
    import math
    from drivers.cubemars_driver import CANBusManager

    config = load_config(config_path)
    setup_logging(config.system.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("  ТЕСТ МОТОРОВ")
    logger.info("=" * 60)

    can_cfg = config.can
    mot_cfg = config.motors

    manager = CANBusManager(
        interface=can_cfg.interface,
        channel=can_cfg.channel,
        bitrate=can_cfg.bitrate
    )

    if not manager.connect():
        logger.error("Не удалось подключиться к CAN-шине!")
        return

    try:
        motor_h = manager.add_motor(mot_cfg.horizontal.can_id)
        motor_v = manager.add_motor(mot_cfg.vertical.can_id)

        logger.info("Включение моторов...")
        motor_h.enable()
        motor_v.enable()
        time.sleep(0.5)

        logger.info("Установка нуля...")
        motor_h.set_zero()
        motor_v.set_zero()
        time.sleep(0.5)

        # Тест горизонтального мотора
        logger.info("Горизонтальный мотор: +5°...")
        target = math.radians(5.0)
        for _ in range(100):
            motor_h.send_position(target, kp=30.0, kd=3.0)
            time.sleep(0.01)
        time.sleep(1.0)

        logger.info("Горизонтальный мотор: 0°...")
        for _ in range(100):
            motor_h.send_position(0.0, kp=30.0, kd=3.0)
            time.sleep(0.01)
        time.sleep(1.0)

        # Тест вертикального мотора
        logger.info("Вертикальный мотор: +5°...")
        target = math.radians(5.0)
        for _ in range(100):
            motor_v.send_position(target, kp=40.0, kd=4.0)
            time.sleep(0.01)
        time.sleep(1.0)

        logger.info("Вертикальный мотор: 0°...")
        for _ in range(100):
            motor_v.send_position(0.0, kp=40.0, kd=4.0)
            time.sleep(0.01)
        time.sleep(1.0)

        logger.info("Тест завершён!")
        logger.info(
            f"  Горизонт: {motor_h.state.position_deg:.2f}°, "
            f"ток={motor_h.state.current_a:.2f}А"
        )
        logger.info(
            f"  Вертикаль: {motor_v.state.position_deg:.2f}°, "
            f"ток={motor_v.state.current_a:.2f}А"
        )

    except KeyboardInterrupt:
        logger.info("Тест прерван")
    finally:
        manager.disconnect()


def main():
    """Главная точка входа."""
    parser = argparse.ArgumentParser(
        description="Платформа слежения — роботизированная система наведения",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python3 main.py                    Запуск с настройками по умолчанию
  python3 main.py -c custom.yaml     Запуск с другим конфигом
  python3 main.py --test-joystick    Тест джойстика (оси, кнопки)
  python3 main.py --test-motors      Тест моторов (малые движения)
        """
    )
    parser.add_argument(
        "-c", "--config",
        default="config/settings.yaml",
        help="Путь к файлу конфигурации (по умолчанию: config/settings.yaml)"
    )
    parser.add_argument(
        "--test-joystick",
        action="store_true",
        help="Режим тестирования джойстика"
    )
    parser.add_argument(
        "--test-motors",
        action="store_true",
        help="Режим тестирования моторов"
    )

    args = parser.parse_args()

    # ── Тест джойстика ──
    if args.test_joystick:
        test_joystick()
        return

    # ── Тест моторов ──
    if args.test_motors:
        test_motors(args.config)
        return

    # ── Нормальный запуск ──
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"ОШИБКА: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ОШИБКА загрузки конфигурации: {e}")
        sys.exit(1)

    # Настройка логирования
    setup_logging(
        log_level=config.system.log_level,
        log_to_file=config.system.log_to_file,
        log_file=config.system.log_file,
    )

    logger = logging.getLogger(__name__)
    logger.info("╔══════════════════════════════════════════════╗")
    logger.info("║     TRACKING PLATFORM — Платформа слежения  ║")
    logger.info("╚══════════════════════════════════════════════╝")

    # Создание и запуск контроллера
    controller = PlatformController(config)

    if controller.initialize():
        controller.run()
    else:
        logger.error("Инициализация не удалась. Завершение.")
        controller.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    main()
