# -*- coding: utf-8 -*-
"""
Главный контроллер платформы слежения.

Связывает все подсистемы воедино:
- Камера → Детектор → Трекер → Автослежение → Моторы
- Джойстик → Управление / переключение режимов
- Дальномер → Баллистика → Точка перехвата
- Все данные → HUD → Дисплей

Архитектура потоков:
  [Камера]  → фоновый поток захвата
  [Моторы]  → главный цикл (синхронно, 100 Гц)
  [Детекция] → в главном цикле (каждый N-й кадр)
  [HUD]     → в главном цикле
  [Дальномер] → фоновый поток опроса
"""

import time
import math
import signal
import logging
import threading
import sys
import os

import numpy as np

from config.app_config import AppConfig
from core.state_machine import (
    StateMachine, SharedState, PlatformState, TargetLockState
)
from core.autotracker import AutoTracker
from core.ballistic_calculator import BallisticCalculator
from core.trajectory_predictor import TrajectoryPredictor
from drivers.cubemars_driver import CANBusManager, CubemarsMotor
from drivers.joystick_driver import JoystickDriver
from drivers.camera_driver import CameraDriver
from drivers.rangefinder_driver import RangefinderDriver
from perception.detector import YOLODetector
from perception.tracker import TargetManager
from ui.hud_renderer import HUDRenderer

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


class PlatformController:
    """
    Главный контроллер роботизированной платформы слежения.

    Инициализирует все подсистемы, запускает главный цикл,
    обрабатывает переключение режимов и завершение работы.
    """

    def __init__(self, config: AppConfig):
        """
        Args:
            config: Полная конфигурация приложения
        """
        self._cfg = config
        self._running = False
        self._start_time = 0.0

        # ── Общее состояние (single source of truth) ──
        self._state = SharedState()

        # ── State Machine ──
        self._sm = StateMachine(self._state)

        # ── Подсистемы (создаются, но не инициализируются) ──
        self._can_manager: CANBusManager = None
        self._motor_h: CubemarsMotor = None
        self._motor_v: CubemarsMotor = None
        self._joystick: JoystickDriver = None
        self._camera: CameraDriver = None
        self._rangefinder: RangefinderDriver = None
        self._detector: YOLODetector = None
        self._target_mgr: TargetManager = None
        self._autotracker: AutoTracker = None
        self._ballistics: BallisticCalculator = None
        self._predictor: TrajectoryPredictor = None
        self._hud: HUDRenderer = None

        # ── Управление ──
        self._target_yaw_deg = 0.0
        self._target_pitch_deg = 0.0
        self._smooth_yaw = 0.0
        self._smooth_pitch = 0.0
        self._smoothing = 0.3

        # ── Состояние стрельбы ──
        self._firing = False           # Триггер удерживается
        self._firing_boost_until = 0.0 # Время окончания усиленных gains

        # ── Потоки ──
        # Поток детекции YOLO26
        self._detection_thread: threading.Thread = None
        self._latest_detections = []
        self._detection_lock = threading.Lock()
        self._detection_frame = None
        self._new_frame_event = threading.Event()

        # Поток управления моторами (100 Гц, высший приоритет)
        self._motor_thread: threading.Thread = None
        self._motor_lock = threading.Lock()

        # Последний кадр для HUD (потокобезопасно)
        self._hud_frame = None
        self._hud_frame_lock = threading.Lock()

        # Обработчики сигналов
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("PlatformController создан")

    def _signal_handler(self, signum, frame):
        """Обработка сигналов завершения."""
        logger.info(f"Сигнал {signum} получен, завершаю работу...")
        self._running = False

    # ════════════════════════════════════════════════════════
    # Инициализация
    # ════════════════════════════════════════════════════════

    def initialize(self) -> bool:
        """
        Инициализировать все подсистемы.
        Возвращает True если критические системы готовы.
        """
        logger.info("=" * 60)
        logger.info("  ПЛАТФОРМА СЛЕЖЕНИЯ — ИНИЦИАЛИЗАЦИЯ")
        logger.info("=" * 60)

        cfg = self._cfg
        ok = True

        # ── 1. Джойстик ──
        logger.info("  [1/7] Джойстик...")
        joy_cfg = cfg.joystick
        self._joystick = JoystickDriver(
            device_index=joy_cfg.device_index,
            axis_yaw=joy_cfg.axis_yaw,
            axis_pitch=joy_cfg.axis_pitch,
            axis_zoom=joy_cfg.axis_zoom,
            invert_yaw=joy_cfg.invert_yaw,
            invert_pitch=joy_cfg.invert_pitch,
            deadzone=joy_cfg.deadzone,
            sensitivity_exponent=joy_cfg.sensitivity_exponent,
            speed_multiplier=joy_cfg.speed_multiplier,
        )
        if self._joystick.initialize():
            logger.info(f"    ОК: {self._joystick.state.device_name}")
            # Назначаем callback-и на кнопки
            self._joystick.set_button_callback(
                joy_cfg.button_mode_toggle, self._on_mode_toggle
            )
            self._joystick.set_button_callback(
                joy_cfg.button_estop, self._on_estop
            )
            self._joystick.set_button_callback(
                joy_cfg.button_center, self._on_center
            )
            self._joystick.set_button_callback(
                joy_cfg.button_fire_rangefinder, self._on_fire_rangefinder
            )
            self._joystick.set_button_callback(
                joy_cfg.button_lock_target, self._on_lock_target
            )
            self._joystick.set_button_callback(
                joy_cfg.button_switch_weapon, self._on_switch_weapon
            )
            self._joystick.set_button_callback(
                joy_cfg.button_fire, self._on_fire
            )
        else:
            logger.error("    ОШИБКА: джойстик не найден!")
            ok = False

        # ── 2. CAN-шина и моторы ──
        logger.info("  [2/7] CAN-шина и моторы...")
        can_cfg = cfg.can
        mot_cfg = cfg.motors
        self._can_manager = CANBusManager(
            interface=can_cfg.interface,
            channel=can_cfg.channel,
            bitrate=can_cfg.bitrate
        )
        if self._can_manager.connect():
            self._motor_h = self._can_manager.add_motor(mot_cfg.horizontal.can_id)
            self._motor_v = self._can_manager.add_motor(mot_cfg.vertical.can_id)

            if self._motor_h.enable() and self._motor_v.enable():
                self._motor_h.set_zero()
                self._motor_v.set_zero()
                logger.info("    ОК: моторы включены, ноль установлен")
            else:
                logger.error("    ОШИБКА: не удалось включить моторы")
                ok = False
        else:
            logger.error("    ОШИБКА: CAN-шина недоступна")
            ok = False

        # ── 3. Камера ──
        logger.info("  [3/7] Камера...")
        cam_cfg = cfg.camera
        self._camera = CameraDriver(
            device_index=cam_cfg.device_index,
            width=cam_cfg.width, height=cam_cfg.height, fps=cam_cfg.fps,
            zoom_min=cam_cfg.zoom_min, zoom_max=cam_cfg.zoom_max,
            zoom_step=cam_cfg.zoom_step, zoom_default=cam_cfg.zoom_default,
            fov_h_deg=cam_cfg.fov_h_deg, fov_v_deg=cam_cfg.fov_v_deg,
        )
        if self._camera.open():
            logger.info(f"    ОК: {cam_cfg.width}x{cam_cfg.height}@{cam_cfg.fps}")
        else:
            logger.error("    ОШИБКА: камера недоступна")
            ok = False

        # ── 4. Дальномер ──
        logger.info("  [4/7] Дальномер...")
        if cfg.rangefinder.enabled:
            rf_cfg = cfg.rangefinder
            self._rangefinder = RangefinderDriver(
                port=rf_cfg.port, baudrate=rf_cfg.baudrate,
                timeout_sec=rf_cfg.timeout_sec,
                min_range_m=rf_cfg.min_range_m, max_range_m=rf_cfg.max_range_m,
                poll_rate_hz=rf_cfg.poll_rate_hz,
            )
            if self._rangefinder.open():
                logger.info(f"    ОК: {rf_cfg.port}")
            else:
                logger.warning("    ВНИМАНИЕ: дальномер недоступен")
        else:
            logger.info("    Пропущен (отключён в конфигурации)")

        # ── 5. Детектор YOLO ──
        logger.info("  [5/7] Детектор YOLO...")
        det_cfg = cfg.detection
        self._detector = YOLODetector(
            model_path=det_cfg.model_path,
            confidence_threshold=det_cfg.confidence_threshold,
            nms_threshold=det_cfg.nms_threshold,
            target_classes=det_cfg.target_classes,
            input_size=det_cfg.input_size,
            use_hailo=det_cfg.use_hailo,
        )
        if self._detector.initialize():
            logger.info("    ОК: модель загружена")
        else:
            logger.warning("    ВНИМАНИЕ: детектор недоступен (работа без детекции)")

        # ── 6. Трекер + Автослежение + Баллистика ──
        logger.info("  [6/7] Трекер, автослежение, баллистика...")
        self._target_mgr = TargetManager()

        at_cfg = cfg.autotrack
        self._autotracker = AutoTracker(
            pid_yaw_cfg=at_cfg.pid_yaw,
            pid_pitch_cfg=at_cfg.pid_pitch,
            center_deadzone_px=at_cfg.center_deadzone_px,
            max_speed_dps=at_cfg.max_auto_speed_dps,
            frame_width=cam_cfg.width, frame_height=cam_cfg.height,
            fov_h_deg=cam_cfg.fov_h_deg, fov_v_deg=cam_cfg.fov_v_deg,
        )

        bal_cfg = cfg.ballistics
        self._ballistics = BallisticCalculator(
            atmosphere_cfg=bal_cfg.atmosphere,
            profiles=bal_cfg.profiles,
            active_profile=bal_cfg.active_profile,
        )
        # Обновить weapon_name в shared state
        wp = self._ballistics.active_profile
        self._state.weapon_name = wp.short_name
        self._state.weapon_caliber = wp.cartridge
        self._state.effective_range_m = float(wp.effective_range_m)

        self._predictor = TrajectoryPredictor(
            history_size=bal_cfg.prediction.history_points,
            prediction_horizon_sec=bal_cfg.prediction.horizon_sec,
        )
        logger.info("    ОК")

        # ── 7. HUD ──
        logger.info("  [7/7] HUD-прицел...")
        self._hud = HUDRenderer(
            config=cfg.hud,
            frame_width=cam_cfg.width,
            frame_height=cam_cfg.height,
        )
        logger.info("    ОК")

        # ── Результат ──
        if ok:
            logger.info("=" * 60)
            logger.info("  ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО")
            logger.info("=" * 60)
            self._sm.transition_to(
                PlatformState.MANUAL if cfg.system.default_mode == "manual"
                else PlatformState.AUTO,
                "инициализация завершена"
            )
        else:
            logger.error("ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА С ОШИБКАМИ")
            self._sm.report_error("ошибка инициализации оборудования")

        return ok

    # ════════════════════════════════════════════════════════
    # Главный цикл
    # ════════════════════════════════════════════════════════

    def run(self):
        """
        Запустить систему с полной многопоточностью.

        Архитектура потоков:
          1. Camera Thread    — захват кадров (уже в camera_driver)
          2. Detection Thread — YOLO26 + BoT-SORT трекинг
          3. LRF Thread       — опрос дальномера (уже в rangefinder_driver)
          4. Motor Thread     — управление моторами MIT Mode (100 Гц)
          5. Main Thread      — джойстик + баллистика + HUD (30 Гц)

        Motor Thread работает на 100 Гц независимо от HUD,
        обеспечивая стабильное управление даже при тяжёлом рендере.
        """
        if not self._sm.is_operational:
            logger.error("Платформа не в рабочем состоянии, запуск невозможен")
            return

        self._running = True
        self._start_time = time.time()

        # Запуск фоновых потоков
        self._start_detection_thread()
        self._start_motor_thread()

        logger.info("Многопоточная система запущена:")
        logger.info("  [1] Camera Thread    — захват кадров (30 FPS)")
        logger.info("  [2] Detection Thread — YOLO26 + BoT-SORT")
        logger.info("  [3] LRF Thread       — дальномер (10 Гц)")
        logger.info(f"  [4] Motor Thread     — MIT Mode ({self._cfg.system.motor_control_hz} Гц)")
        logger.info(f"  [5] Main Thread      — джойстик + баллистика + HUD ({self._cfg.system.hud_fps} FPS)")

        hud_dt = 1.0 / self._cfg.system.hud_fps

        # ── Main Thread: джойстик + детекция результаты + баллистика + HUD ──
        while self._running:
            loop_start = time.time()

            # ── 1. Чтение джойстика ──
            if self._joystick:
                self._joystick.update()

            # ── 2. Обработка E-STOP ──
            if self._sm.is_estop:
                if self._joystick and not self._joystick.is_button_pressed(
                    self._cfg.joystick.button_estop
                ):
                    self._sm.recover_from_estop()
                    if self._can_manager:
                        self._can_manager.enable_all()
                else:
                    self._sleep_until(loop_start, hud_dt)
                    continue

            # ── 3. Захват кадра ──
            frame = None
            if self._camera and self._camera.is_opened:
                frame = self._camera.get_frame()
                self._state.camera_fps = self._camera.fps_actual
                self._state.zoom_level = self._camera.zoom

            # ── 4. Передача кадра в поток детекции ──
            if frame is not None and self._detector and self._detector.is_initialized:
                self._detection_frame = frame
                self._new_frame_event.set()

                # Забираем результаты детекции
                with self._detection_lock:
                    if self._latest_detections:
                        self._target_mgr.update(self._latest_detections)
                        self._latest_detections = []
                        self._update_tracking_state()

            # ── 5. Управление платформой (вычисление целевых углов) ──
            motor_dt = 1.0 / self._cfg.system.motor_control_hz
            if self._sm.is_manual:
                self._process_manual_control(motor_dt)
            elif self._sm.is_auto:
                self._process_auto_control(motor_dt)

            # ── 6. Управление зумом ──
            self._process_zoom()

            # ── 7. Баллистика и предсказание ──
            if self._state.target_lock == TargetLockState.LOCKED:
                self._update_ballistics()

            # ── 7.5. Обновление боевой готовности ──
            self._update_combat_readiness()

            # ── 8. HUD + отображение (30 FPS) ──
            if frame is not None and self._hud:
                self._state.timestamp = time.time()
                self._state.uptime_sec = time.time() - self._start_time
                display_frame = self._hud.render(frame, self._state)

                if cv2 is not None:
                    cv2.imshow("ПЛАТФОРМА СЛЕЖЕНИЯ", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        self._running = False

            # ── 9. Тайминг (30 FPS для HUD) ──
            elapsed = time.time() - loop_start
            self._state.loop_time_ms = elapsed * 1000
            self._sleep_until(loop_start, hud_dt)

        # Завершение
        self.shutdown()

    # ════════════════════════════════════════════════════════
    # Обработка режимов
    # ════════════════════════════════════════════════════════

    def _process_manual_control(self, dt: float):
        """Ручное управление джойстиком."""
        if not self._joystick:
            return

        joy_yaw, joy_pitch = self._joystick.get_control_input()

        # Сглаживание
        self._smooth_yaw += self._smoothing * (joy_yaw - self._smooth_yaw)
        self._smooth_pitch += self._smoothing * (joy_pitch - self._smooth_pitch)

        # Применение направления
        h_dir = self._cfg.motors.horizontal.direction
        v_dir = self._cfg.motors.vertical.direction

        # Rate mode: джойстик управляет скоростью
        h_speed = self._cfg.motors.horizontal.max_speed_dps
        v_speed = self._cfg.motors.vertical.max_speed_dps

        self._target_yaw_deg += self._smooth_yaw * h_dir * h_speed * dt
        self._target_pitch_deg += self._smooth_pitch * v_dir * v_speed * dt

        # Ограничение углов
        h_cfg = self._cfg.motors.horizontal
        v_cfg = self._cfg.motors.vertical
        self._target_yaw_deg = max(h_cfg.min_angle_deg,
                                    min(h_cfg.max_angle_deg, self._target_yaw_deg))
        self._target_pitch_deg = max(v_cfg.min_angle_deg,
                                      min(v_cfg.max_angle_deg, self._target_pitch_deg))

        # Обновление состояния
        self._state.target_yaw_deg = self._target_yaw_deg
        self._state.target_pitch_deg = self._target_pitch_deg

    def _process_auto_control(self, dt: float):
        """Автоматическое слежение за целью."""
        locked = self._target_mgr.locked_track if self._target_mgr else None

        if locked is None:
            # Нет захваченной цели — переход в поиск
            self._state.target_lock = TargetLockState.SEARCHING
            self._autotracker.reset()
            return

        # Обновляем параметры камеры (зум мог измениться)
        if self._camera:
            self._autotracker.update_camera_params(
                *self._camera.resolution,
                self._camera.fov_h, self._camera.fov_v
            )

        # Вычисляем скорости автослежения
        tcx, tcy = locked.center
        yaw_speed, pitch_speed = self._autotracker.compute(tcx, tcy, dt)

        # Применяем скорости
        self._target_yaw_deg += yaw_speed * dt
        self._target_pitch_deg += pitch_speed * dt

        # Ограничение
        h_cfg = self._cfg.motors.horizontal
        v_cfg = self._cfg.motors.vertical
        self._target_yaw_deg = max(h_cfg.min_angle_deg,
                                    min(h_cfg.max_angle_deg, self._target_yaw_deg))
        self._target_pitch_deg = max(v_cfg.min_angle_deg,
                                      min(v_cfg.max_angle_deg, self._target_pitch_deg))

        self._state.target_yaw_deg = self._target_yaw_deg
        self._state.target_pitch_deg = self._target_pitch_deg
        self._state.target_lock = TargetLockState.LOCKED

    def _send_motor_commands(self):
        """
        Отправить MIT Mode команды моторам.

        Формула мотора AK80-64:
            τ_out = kp*(θ_cmd - θ_act) + kd*(ω_cmd - ω_act) + τ_ff

        Мы используем все 5 параметров MIT Mode:
        - position: целевой угол
        - velocity: feedforward по скорости (компенсация инерции при слежении)
        - kp/kd: жёсткость и демпфирование (динамически меняются)
        - torque: feedforward момент (гравитационная компенсация для pitch)
        """
        if not self._sm.is_operational:
            return

        h_cfg = self._cfg.motors.horizontal
        v_cfg = self._cfg.motors.vertical

        h_rad = math.radians(self._target_yaw_deg)
        v_rad = math.radians(self._target_pitch_deg)

        # ── Velocity feedforward ──
        # Компенсация инерции: подаём скорость как feedforward
        # чтобы мотор начинал двигаться до накопления ошибки позиции
        h_vel_ff = self._smooth_yaw * math.radians(h_cfg.max_speed_dps) * h_cfg.velocity_ff_gain
        v_vel_ff = self._smooth_pitch * math.radians(v_cfg.max_speed_dps) * v_cfg.velocity_ff_gain

        # ── Gravity compensation (только pitch) ──
        # Момент для удержания веса изделия: τ_grav = m*g*L*cos(θ)
        # При θ=0 (горизонт) момент максимален, при ±90° = 0
        grav_torque = v_cfg.gravity_compensation_nm * math.cos(v_rad)

        # ── Динамические gains ──
        # При стрельбе переключаемся на усиленные gains для гашения отдачи
        if self._firing or time.time() < self._firing_boost_until:
            h_kp = h_cfg.kp_firing
            h_kd = h_cfg.kd_firing
            v_kp = v_cfg.kp_firing
            v_kd = v_cfg.kd_firing
        else:
            h_kp = h_cfg.kp
            h_kd = h_cfg.kd
            v_kp = v_cfg.kp
            v_kd = v_cfg.kd

        # ── Отправка MIT Mode команд ──
        if self._motor_h and self._motor_h.is_enabled:
            self._motor_h.send_mit_command(
                position_rad=h_rad,
                velocity_rads=h_vel_ff,
                kp=h_kp,
                kd=h_kd,
                torque_nm=0.0  # Yaw: ЦМ на оси, гравитация не влияет
            )
            self._state.yaw_deg = self._motor_h.state.position_deg
            self._state.yaw_current_a = self._motor_h.state.current_a

        if self._motor_v and self._motor_v.is_enabled:
            self._motor_v.send_mit_command(
                position_rad=v_rad,
                velocity_rads=v_vel_ff,
                kp=v_kp,
                kd=v_kd,
                torque_nm=grav_torque  # Pitch: компенсация веса изделия
            )
            self._state.pitch_deg = self._motor_v.state.position_deg
            self._state.pitch_current_a = self._motor_v.state.current_a

    def _process_zoom(self):
        """Управление зумом через колёсико джойстика."""
        if not self._joystick or not self._camera:
            return

        zoom_input = self._joystick.get_zoom_input()
        if zoom_input > 0.1:
            self._camera.zoom_in()
        elif zoom_input < -0.1:
            self._camera.zoom_out()

    def _update_tracking_state(self):
        """Обновить состояние трекинга в SharedState."""
        from perception.detector import get_class_name, get_threat_priority

        locked = self._target_mgr.locked_track

        if locked:
            self._state.target_bbox = locked.bbox
            self._state.target_center_px = locked.center
            self._state.target_class_id = locked.class_id
            self._state.target_confidence = locked.confidence
            self._state.track_id = locked.track_id

            # Тип цели и уровень угрозы
            self._state.target_type_name = get_class_name(locked.class_id)
            self._state.threat_level = get_threat_priority(locked.class_id)
        else:
            self._state.target_bbox = None
            self._state.target_center_px = None
            self._state.track_id = -1
            self._state.target_type_name = ""
            self._state.threat_level = 0

            if self._state.target_lock == TargetLockState.LOCKED:
                self._state.target_lock = TargetLockState.LOST

            # Обновить track_id из всех треков (для HUD)
            all_tracks = self._target_mgr.tracks
            if all_tracks:
                self._state.target_lock = TargetLockState.DETECTED

    def _update_ballistics(self):
        """Обновить баллистические расчёты."""
        if not self._rangefinder or not self._rangefinder.is_valid:
            return

        distance = self._rangefinder.last_distance
        self._state.distance_m = distance
        self._state.distance_valid = True

        # Добавить наблюдение в предсказатель
        self._predictor.add_observation(
            self._state.yaw_deg, self._state.pitch_deg, distance,
            self._state.target_center_px[0] if self._state.target_center_px else 0,
            self._state.target_center_px[1] if self._state.target_center_px else 0,
        )

        est = self._predictor.estimate
        self._state.target_speed_mps = est.speed_mps
        self._state.target_heading_deg = est.heading_deg

        # Баллистический расчёт
        if est.is_valid and distance > 0:
            solution = self._ballistics.compute_lead(
                distance_m=distance,
                target_speed_mps=est.speed_mps,
                target_heading_rad=math.radians(est.heading_deg),
                platform_yaw_rad=math.radians(self._state.yaw_deg),
                platform_pitch_rad=math.radians(self._state.pitch_deg),
            )

            self._state.time_of_flight_sec = solution.time_of_flight
            self._state.lead_yaw_deg = solution.lead_yaw_deg
            self._state.lead_pitch_deg = solution.lead_pitch_deg
            self._state.intercept_possible = solution.is_valid
            self._state.bullet_energy_j = solution.energy_at_target
            self._state.mach_at_target = solution.mach_at_target

            # ── Коррекция параллакса камера-LRF ──
            par_yaw, par_pitch = self._apply_parallax_correction(distance)

            # ── Коррекция boresight (пристрелка) ──
            bs_yaw, bs_pitch = self._interpolate_boresight(distance)

            # Суммарные углы упреждения с коррекциями
            total_lead_yaw = solution.lead_yaw_deg + par_yaw + bs_yaw
            total_lead_pitch = solution.lead_pitch_deg + par_pitch + bs_pitch

            self._state.lead_yaw_deg = total_lead_yaw
            self._state.lead_pitch_deg = total_lead_pitch

            # Точка перехвата в пикселях (с коррекциями)
            if solution.is_valid and self._camera:
                lx, ly = self._camera.angle_to_pixel(
                    total_lead_yaw, total_lead_pitch
                )
                self._state.lead_point_px = (lx, ly)

    # ════════════════════════════════════════════════════════
    # Callback-и кнопок джойстика
    # ════════════════════════════════════════════════════════

    def _on_mode_toggle(self):
        """Переключение Manual ↔ Auto."""
        self._sm.toggle_mode()
        if self._sm.is_manual:
            self._autotracker.reset()
        elif self._sm.is_auto:
            # Автоматически захватить ближайшую цель
            if self._target_mgr and not self._target_mgr.is_target_locked:
                self._target_mgr.lock_target()

    def _on_estop(self):
        """Аварийная остановка."""
        logger.warning("КНОПКА E-STOP НАЖАТА!")
        self._sm.emergency_stop()
        if self._can_manager:
            self._can_manager.emergency_stop_all()

    def _on_center(self):
        """Возврат в центр."""
        logger.info("Возврат в центр...")
        self._target_yaw_deg = 0.0
        self._target_pitch_deg = 0.0

    def _on_fire_rangefinder(self):
        """Замер дальности."""
        if self._rangefinder:
            logger.info("Замер дальности...")
            self._rangefinder.fire()

    def _on_lock_target(self):
        """Захват / сброс цели."""
        if self._target_mgr:
            self._target_mgr.toggle_lock()

    def _on_fire(self):
        """
        ОГОНЬ — триггер нажат.

        При нажатии триггера:
        1. Переключаемся на усиленные MIT Mode gains (kp_firing/kd_firing)
           для мгновенного гашения отдачи
        2. Gains остаются усиленными на boost_duration_sec после отпускания
        3. Логирование события стрельбы
        """
        self._firing = True
        self._state.firing_active = True
        self._state.burst_start_time = time.time()
        recoil_cfg = self._cfg.motors.recoil
        self._firing_boost_until = time.time() + recoil_cfg.boost_duration_sec
        logger.info("ОГОНЬ!")

    def _on_switch_weapon(self):
        """Переключение профиля вооружения (КОРД ↔ ПКТ)."""
        if self._ballistics:
            self._ballistics.switch_profile()
            wp = self._ballistics.active_profile
            self._state.weapon_name = wp.short_name
            self._state.weapon_caliber = wp.cartridge
            self._state.effective_range_m = float(wp.effective_range_m)
            self._state.shots_fired = 0
            logger.info(f"Вооружение: {wp.short_name} ({wp.cartridge})")

    # ════════════════════════════════════════════════════════
    # Боевая готовность
    # ════════════════════════════════════════════════════════

    def _update_combat_readiness(self):
        """
        Обновить индикаторы боевой готовности в SharedState.

        ГОТОВ К СТРЕЛЬБЕ = все 3 условия:
          1. Цель захвачена (LOCKED)
          2. Дистанция замерена (distance_valid)
          3. Перехват возможен (intercept_possible)

        В ЗОНЕ ПОРАЖЕНИЯ = дистанция < effective_range текущего оружия
        """
        s = self._state
        now = time.time()

        # Зона поражения
        s.in_effective_range = (
            s.distance_valid
            and 0 < s.distance_m <= s.effective_range_m
        )

        # Готовность к стрельбе
        s.ready_to_fire = (
            s.target_lock == TargetLockState.LOCKED
            and s.distance_valid
            and s.intercept_possible
            and s.in_effective_range
        )

        # Состояние стрельбы и отдачи
        is_firing_now = (
            self._joystick
            and self._joystick.is_button_pressed(self._cfg.joystick.button_fire)
        )
        s.firing_active = bool(is_firing_now)
        s.recoil_boost_active = now < self._firing_boost_until

        # Если триггер отпущен — сбросить firing
        if not is_firing_now:
            self._firing = False

        # Счётчик выстрелов (приблизительный по rate_of_fire)
        if s.firing_active and s.burst_start_time > 0:
            burst_duration = now - s.burst_start_time
            wp = self._ballistics.active_profile if self._ballistics else None
            if wp and wp.rate_of_fire_rpm > 0:
                s.shots_fired = int(burst_duration * wp.rate_of_fire_rpm / 60.0)

        # Вектор движения цели в пикселях (для стрелки на HUD)
        if s.target_center_px and self._predictor and self._predictor.has_enough_data:
            est = self._predictor.estimate
            if est.is_valid and self._camera and s.distance_m > 0:
                # Преобразуем скорость цели в пиксельное смещение
                # Используем FOV камеры для масштабирования
                fov_h = self._camera.fov_h
                fov_v = self._camera.fov_v
                w, h = self._camera.resolution
                if fov_h > 0 and fov_v > 0:
                    # Угловая скорость цели (°/с) → пиксели/с
                    # Грубая оценка: speed * sin(heading) / distance → угловая скорость
                    heading_rad = math.radians(est.heading_deg)
                    angular_h = (est.speed_mps * math.sin(heading_rad)) / s.distance_m
                    angular_v = (est.speed_mps * math.cos(heading_rad) * math.sin(math.radians(s.pitch_deg))) / s.distance_m
                    # Рад/с → пиксели (масштаб: FOV = ширина кадра)
                    dx = math.degrees(angular_h) / fov_h * w
                    dy = math.degrees(angular_v) / fov_v * h
                    # Ограничиваем длину вектора
                    max_len = 80.0
                    vec_len = math.sqrt(dx * dx + dy * dy)
                    if vec_len > max_len:
                        scale = max_len / vec_len
                        dx *= scale
                        dy *= scale
                    s.target_velocity_px = (dx, dy)
                else:
                    s.target_velocity_px = None
            else:
                s.target_velocity_px = None
        else:
            s.target_velocity_px = None

    # ════════════════════════════════════════════════════════
    # Завершение
    # ════════════════════════════════════════════════════════

    def shutdown(self):
        """Корректное завершение работы всех подсистем."""
        logger.info("=" * 60)
        logger.info("  ЗАВЕРШЕНИЕ РАБОТЫ")
        logger.info("=" * 60)

        self._running = False
        self._sm.transition_to(PlatformState.SHUTDOWN, "завершение работы")

        # Камера
        if self._camera:
            self._camera.close()

        # Дальномер
        if self._rangefinder:
            self._rangefinder.close()

        # Детектор
        if self._detector:
            self._detector.shutdown()

        # Моторы и CAN
        if self._can_manager:
            self._can_manager.disconnect()

        # Джойстик
        if self._joystick:
            self._joystick.shutdown()

        # OpenCV окна
        if cv2 is not None:
            cv2.destroyAllWindows()

        logger.info("Все подсистемы остановлены")

    # ════════════════════════════════════════════════════════
    # Утилиты
    # ════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════
    # Поток детекции YOLO26
    # ════════════════════════════════════════════════════════

    def _start_motor_thread(self):
        """
        Запустить фоновый поток управления моторами (100 Гц).

        Этот поток работает с ВЫСШИМ приоритетом и стабильной частотой,
        независимо от HUD-рендера и детекции. Обеспечивает плавное
        управление платформой даже при тяжёлой нагрузке на CPU.
        """
        self._motor_thread = threading.Thread(
            target=self._motor_control_loop,
            name="MotorControlThread",
            daemon=True
        )
        self._motor_thread.start()
        logger.info(f"Поток управления моторами запущен ({self._cfg.system.motor_control_hz} Гц)")

    def _motor_control_loop(self):
        """
        Фоновый поток: отправка MIT Mode команд моторам на 100 Гц.

        Читает целевые углы из self._target_yaw_deg / self._target_pitch_deg
        (обновляются из Main Thread) и отправляет CAN-команды.
        """
        motor_dt = 1.0 / self._cfg.system.motor_control_hz

        while self._running:
            loop_start = time.time()

            if self._sm.is_operational:
                self._send_motor_commands()

            self._sleep_until(loop_start, motor_dt)

    def _start_detection_thread(self):
        """Запустить фоновый поток детекции."""
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            name="DetectionThread",
            daemon=True
        )
        self._detection_thread.start()
        logger.info("Поток детекции YOLO26 запущен")

    def _detection_loop(self):
        """
        Фоновый поток: детекция + трекинг YOLO26 BoT-SORT.
        Работает параллельно с главным циклом, не блокирует моторы.
        """
        while self._running:
            # Ждём нового кадра от главного цикла
            self._new_frame_event.wait(timeout=0.1)
            self._new_frame_event.clear()

            frame = self._detection_frame
            if frame is None:
                continue

            try:
                detections = self._detector.detect_and_track(frame)
                with self._detection_lock:
                    self._latest_detections = detections
            except Exception as e:
                logger.error(f"Ошибка в потоке детекции: {e}")

    # ════════════════════════════════════════════════════════
    # Коррекция параллакса и boresight
    # ════════════════════════════════════════════════════════

    def _apply_parallax_correction(self, distance_m: float) -> tuple:
        """
        Коррекция параллакса камера-LRF.

        Камера и дальномер смещены на baseline мм.
        На ближних дистанциях это даёт угловую ошибку:
            parallax_angle = atan(baseline / distance)

        Args:
            distance_m: Дистанция до цели (м)

        Returns:
            (yaw_correction_deg, pitch_correction_deg)
        """
        pcfg = self._cfg.motors.parallax
        baseline_m = pcfg.baseline_mm / 1000.0

        if distance_m <= 0:
            return (0.0, 0.0)

        angle_rad = math.atan(baseline_m / distance_m)
        angle_deg = math.degrees(angle_rad)

        # Направление коррекции
        if pcfg.direction == "right":
            return (-angle_deg, 0.0)  # LRF правее → камера смотрит левее
        elif pcfg.direction == "left":
            return (angle_deg, 0.0)
        elif pcfg.direction == "above":
            return (0.0, -angle_deg)
        elif pcfg.direction == "below":
            return (0.0, angle_deg)
        return (0.0, 0.0)

    def _interpolate_boresight(self, distance_m: float) -> tuple:
        """
        Линейная интерполяция boresight offsets по дистанции.

        Таблица пристрелки из конфига: [[dist, yaw_mrad, pitch_mrad], ...]
        Между точками — линейная интерполяция.
        За пределами — экстраполяция по крайним точкам.

        Args:
            distance_m: Дистанция до цели (м)

        Returns:
            (yaw_offset_deg, pitch_offset_deg)
        """
        table = self._cfg.motors.boresight
        if not table or len(table) < 2:
            return (0.0, 0.0)

        # Сортировка по дистанции
        table = sorted(table, key=lambda x: x[0])

        # За пределами таблицы — экстраполяция крайними значениями
        if distance_m <= table[0][0]:
            yaw_mrad = table[0][1]
            pitch_mrad = table[0][2]
        elif distance_m >= table[-1][0]:
            yaw_mrad = table[-1][1]
            pitch_mrad = table[-1][2]
        else:
            # Линейная интерполяция
            for i in range(len(table) - 1):
                d0, y0, p0 = table[i]
                d1, y1, p1 = table[i + 1]
                if d0 <= distance_m <= d1:
                    t = (distance_m - d0) / (d1 - d0) if d1 != d0 else 0
                    yaw_mrad = y0 + t * (y1 - y0)
                    pitch_mrad = p0 + t * (p1 - p0)
                    break
            else:
                yaw_mrad = 0.0
                pitch_mrad = 0.0

        # мрад → градусы
        return (yaw_mrad * 0.001 * 180.0 / math.pi,
                pitch_mrad * 0.001 * 180.0 / math.pi)

    # ════════════════════════════════════════════════════════
    # Утилиты
    # ════════════════════════════════════════════════════════

    @staticmethod
    def _sleep_until(start: float, dt: float):
        """Спать до конца временного слота."""
        elapsed = time.time() - start
        remaining = dt - elapsed
        if remaining > 0:
            time.sleep(remaining)
