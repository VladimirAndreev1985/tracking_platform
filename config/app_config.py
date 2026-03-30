# -*- coding: utf-8 -*-
"""
Модуль конфигурации приложения.

Загружает настройки из YAML-файла и предоставляет
типизированные dataclass-объекты для каждой подсистемы.
Все параметры доступны через единый объект AppConfig.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import yaml

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Dataclass-ы для каждой секции конфигурации
# ════════════════════════════════════════════════════════════════

@dataclass
class SystemConfig:
    """Общие системные параметры."""
    main_loop_hz: int = 60
    motor_control_hz: int = 100
    hud_fps: int = 30
    default_mode: str = "manual"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/platform.log"


@dataclass
class CANConfig:
    """Параметры CAN-шины."""
    interface: str = "socketcan"
    channel: str = "can0"
    bitrate: int = 1_000_000


@dataclass
class MotorConfig:
    """Параметры одного мотора Cubemars AK80-64 (MIT Mode)."""
    can_id: int = 1
    name: str = ""
    min_angle_deg: float = -180.0
    max_angle_deg: float = 180.0
    max_speed_dps: float = 80.0
    # MIT Mode gains (слежение)
    kp: float = 100.0
    kd: float = 4.0
    # MIT Mode gains (стрельба — усиленные для гашения отдачи)
    kp_firing: float = 150.0
    kd_firing: float = 5.0
    # Feedforward по скорости (Нм·с/рад)
    velocity_ff_gain: float = 0.3
    # Гравитационная компенсация (Нм)
    gravity_compensation_nm: float = 0.0
    direction: int = 1


@dataclass
class ParallaxConfig:
    """Параметры параллакса камера-LRF."""
    baseline_mm: float = 80.0
    direction: str = "right"


@dataclass
class RecoilConfig:
    """Параметры компенсации отдачи."""
    enabled: bool = True
    boost_duration_sec: float = 0.12
    pitch_arm_m: float = 0.0186
    mass_kord_kg: float = 40.0
    mass_pkt_kg: float = 25.0


@dataclass
class MotorsConfig:
    """Конфигурация обоих моторов платформы."""
    horizontal: MotorConfig = field(default_factory=MotorConfig)
    vertical: MotorConfig = field(default_factory=MotorConfig)
    parallax: ParallaxConfig = field(default_factory=ParallaxConfig)
    boresight: list = field(default_factory=lambda: [[100, 0.0, 0.0], [300, 0.0, 0.0], [500, 0.0, 0.0]])
    recoil: RecoilConfig = field(default_factory=RecoilConfig)


@dataclass
class CameraConfig:
    """Параметры камеры Arducam PTZ."""
    device_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    zoom_min: float = 1.0
    zoom_max: float = 5.0
    zoom_step: float = 0.1
    zoom_default: float = 1.0
    fov_h_deg: float = 62.2
    fov_v_deg: float = 48.8


@dataclass
class RangefinderConfig:
    """Параметры лазерного дальномера."""
    enabled: bool = True
    port: str = "/dev/ttyAMA0"
    baudrate: int = 115200
    timeout_sec: float = 0.1
    min_range_m: float = 3.0
    max_range_m: float = 1200.0
    poll_rate_hz: int = 10


@dataclass
class DetectionConfig:
    """Параметры детекции YOLO."""
    model_path: str = "models/yolo26n.pt"
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.5
    target_classes: List[int] = field(default_factory=lambda: [4])
    input_size: int = 640
    use_hailo: bool = True


@dataclass
class TrackerConfig:
    """Параметры трекера BoT-SORT."""
    type: str = "botsort"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass
class JoystickConfig:
    """Параметры джойстика Logitech X56."""
    device_index: int = 0
    axis_yaw: int = 0
    axis_pitch: int = 1
    axis_zoom: int = 5
    invert_yaw: bool = False
    invert_pitch: bool = True
    deadzone: float = 0.08
    sensitivity_exponent: float = 2.0
    speed_multiplier: float = 1.0
    button_mode_toggle: int = 5
    button_estop: int = 1
    button_center: int = 2
    button_fire_rangefinder: int = 0
    button_lock_target: int = 3
    button_switch_weapon: int = 4


@dataclass
class PIDConfig:
    """Коэффициенты PID-регулятора."""
    kp: float = 0.5
    ki: float = 0.05
    kd: float = 0.15


@dataclass
class AutotrackConfig:
    """Параметры автоматического слежения."""
    pid_yaw: PIDConfig = field(default_factory=PIDConfig)
    pid_pitch: PIDConfig = field(default_factory=PIDConfig)
    center_deadzone_px: int = 15
    max_auto_speed_dps: float = 80.0


@dataclass
class WeaponProfile:
    """Профиль вооружения с баллистическими данными пули."""
    name: str = ""
    short_name: str = ""
    caliber_mm: float = 0.0
    cartridge: str = ""
    muzzle_velocity_mps: float = 800.0
    bullet_mass_g: float = 10.0
    bullet_diameter_mm: float = 7.62
    bullet_length_mm: float = 28.0
    ballistic_coefficient_g1: float = 0.4
    drag_model: str = "g1"
    rate_of_fire_rpm: int = 600
    effective_range_m: int = 1500
    max_range_m: int = 4000


@dataclass
class AtmosphereConfig:
    """Атмосферные условия."""
    gravity_mps2: float = 9.80665
    altitude_m: float = 150.0
    temperature_c: float = 15.0
    pressure_hpa: float = 1013.25
    humidity_pct: float = 50.0


@dataclass
class PredictionConfig:
    """Параметры предсказания траектории."""
    horizon_sec: float = 3.0
    history_points: int = 30


@dataclass
class BallisticsConfig:
    """Параметры баллистического калькулятора."""
    active_profile: str = "kord"
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    profiles: dict = field(default_factory=dict)


@dataclass
class HUDConfig:
    """Параметры HUD-прицела."""
    color_reticle: Tuple[int, int, int] = (0, 255, 0)
    color_locked: Tuple[int, int, int] = (0, 0, 255)
    color_lead: Tuple[int, int, int] = (0, 255, 255)
    color_info: Tuple[int, int, int] = (0, 255, 0)
    line_thickness: int = 1
    mil_dot_size: int = 3
    mil_dots_count: int = 5
    mil_dot_spacing: int = 40
    font_scale: float = 0.5
    show_info_panel: bool = True
    show_compass: bool = True
    overlay_alpha: float = 0.7


# ════════════════════════════════════════════════════════════════
# Главный объект конфигурации
# ════════════════════════════════════════════════════════════════

@dataclass
class AppConfig:
    """
    Корневой объект конфигурации всей платформы.
    Содержит типизированные секции для каждой подсистемы.
    """
    system: SystemConfig = field(default_factory=SystemConfig)
    can: CANConfig = field(default_factory=CANConfig)
    motors: MotorsConfig = field(default_factory=MotorsConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    rangefinder: RangefinderConfig = field(default_factory=RangefinderConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    joystick: JoystickConfig = field(default_factory=JoystickConfig)
    autotrack: AutotrackConfig = field(default_factory=AutotrackConfig)
    ballistics: BallisticsConfig = field(default_factory=BallisticsConfig)
    hud: HUDConfig = field(default_factory=HUDConfig)


# ════════════════════════════════════════════════════════════════
# Функции загрузки
# ════════════════════════════════════════════════════════════════

def _dict_to_dataclass(cls, data: dict):
    """
    Рекурсивно преобразует словарь в dataclass.
    Игнорирует неизвестные ключи, использует значения по умолчанию
    для отсутствующих.
    """
    if data is None:
        return cls()

    import dataclasses
    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}

    for key, value in data.items():
        if key not in fieldtypes:
            logger.warning(f"Неизвестный параметр '{key}' в конфигурации, пропускаю")
            continue

        field_type = fieldtypes[key]

        # Обработка вложенных dataclass-ов
        if dataclasses.is_dataclass(field_type):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        # Обработка Tuple из списка
        elif hasattr(field_type, '__origin__') and field_type.__origin__ is tuple:
            kwargs[key] = tuple(value) if isinstance(value, list) else value
        # Обработка строковых аннотаций типов (для Tuple)
        elif isinstance(field_type, str) and 'Tuple' in field_type:
            kwargs[key] = tuple(value) if isinstance(value, list) else value
        else:
            kwargs[key] = value

    return cls(**kwargs)


def _resolve_field_type(cls, field_name: str):
    """Получить реальный тип поля dataclass по имени."""
    import dataclasses
    for f in dataclasses.fields(cls):
        if f.name == field_name:
            return f.type
    return None


def _parse_ballistics(raw: dict) -> BallisticsConfig:
    """
    Разобрать секцию ballistics с профилями вооружения.
    Профили хранятся как dict[str, WeaponProfile].
    """
    if not raw:
        return BallisticsConfig()

    # Атмосфера
    atmosphere = _dict_to_dataclass(
        AtmosphereConfig, raw.get("atmosphere")
    )

    # Предсказание
    prediction = _dict_to_dataclass(
        PredictionConfig, raw.get("prediction")
    )

    # Профили вооружения
    profiles_raw = raw.get("profiles", {})
    profiles = {}
    for key, profile_data in profiles_raw.items():
        profiles[key] = _dict_to_dataclass(WeaponProfile, profile_data)

    active = raw.get("active_profile", "kord")

    if profiles:
        active_name = profiles.get(active, next(iter(profiles.values()))).short_name
        logger.info(f"  Профили вооружения: {list(profiles.keys())}")
        logger.info(f"  Активный профиль: {active} ({active_name})")

    return BallisticsConfig(
        active_profile=active,
        atmosphere=atmosphere,
        prediction=prediction,
        profiles=profiles,
    )


def load_config(config_path: str = "config/settings.yaml") -> AppConfig:
    """
    Загружает конфигурацию из YAML-файла.

    Args:
        config_path: Путь к YAML-файлу настроек

    Returns:
        AppConfig — полностью инициализированный объект конфигурации

    Raises:
        FileNotFoundError: Если файл не найден
        yaml.YAMLError: Если файл содержит невалидный YAML
    """
    # Определяем абсолютный путь относительно корня проекта
    if not os.path.isabs(config_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        logger.warning("Конфигурационный файл пуст, используются значения по умолчанию")
        return AppConfig()

    # Собираем AppConfig из словаря
    config = AppConfig(
        system=_dict_to_dataclass(SystemConfig, raw.get("system")),
        can=_dict_to_dataclass(CANConfig, raw.get("can")),
        motors=MotorsConfig(
            horizontal=_dict_to_dataclass(MotorConfig, raw.get("motors", {}).get("horizontal")),
            vertical=_dict_to_dataclass(MotorConfig, raw.get("motors", {}).get("vertical")),
            parallax=_dict_to_dataclass(ParallaxConfig, raw.get("motors", {}).get("parallax")),
            boresight=raw.get("motors", {}).get("boresight", [[100, 0.0, 0.0], [300, 0.0, 0.0], [500, 0.0, 0.0]]),
            recoil=_dict_to_dataclass(RecoilConfig, raw.get("motors", {}).get("recoil")),
        ),
        camera=_dict_to_dataclass(CameraConfig, raw.get("camera")),
        rangefinder=_dict_to_dataclass(RangefinderConfig, raw.get("rangefinder")),
        detection=_dict_to_dataclass(DetectionConfig, raw.get("detection")),
        tracker=_dict_to_dataclass(TrackerConfig, raw.get("tracker")),
        joystick=_dict_to_dataclass(JoystickConfig, raw.get("joystick")),
        autotrack=AutotrackConfig(
            pid_yaw=_dict_to_dataclass(PIDConfig, raw.get("autotrack", {}).get("pid_yaw")),
            pid_pitch=_dict_to_dataclass(PIDConfig, raw.get("autotrack", {}).get("pid_pitch")),
            center_deadzone_px=raw.get("autotrack", {}).get("center_deadzone_px", 15),
            max_auto_speed_dps=raw.get("autotrack", {}).get("max_auto_speed_dps", 80.0),
        ),
        ballistics=_parse_ballistics(raw.get("ballistics", {})),
        hud=_dict_to_dataclass(HUDConfig, raw.get("hud")),
    )

    logger.info(f"Конфигурация загружена из {config_path}")
    logger.info(f"  Режим по умолчанию: {config.system.default_mode}")
    logger.info(f"  Главный цикл: {config.system.main_loop_hz} Гц")
    logger.info(f"  Камера: {config.camera.width}x{config.camera.height}@{config.camera.fps}")
    logger.info(f"  Дальномер: {'ВКЛ' if config.rangefinder.enabled else 'ВЫКЛ'}")

    return config
