# -*- coding: utf-8 -*-
"""
Конечный автомат (State Machine) режимов платформы.

Определяет все возможные состояния системы и правила
переходов между ними. Обеспечивает безопасную логику
переключения режимов.

Диаграмма состояний:

    ┌──────────┐  инициализация   ┌──────────┐
    │  INIT    │─────────────────▶│  MANUAL  │◀─────┐
    └──────────┘                  └────┬─────┘      │
         │                             │            │
         │ ошибка                кнопка│       кнопка│
         ▼                             ▼            │
    ┌──────────┐                 ┌──────────┐       │
    │  ERROR   │◀────────────────│   AUTO   │───────┘
    └──────────┘    ошибка       └──────────┘
         ▲                             │
         │                      потеря цели
    ┌────┴─────┐                       │
    │  ESTOP   │◀──────────────────────┘
    └──────────┘   аварийная остановка (из любого)
"""

import time
import logging
from enum import Enum, auto
from typing import Optional, Callable, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Состояния системы
# ════════════════════════════════════════════════════════════════

class PlatformState(Enum):
    """Все возможные состояния платформы."""
    INIT = auto()       # Инициализация оборудования
    MANUAL = auto()     # Ручное управление джойстиком
    AUTO = auto()       # Автоматическое слежение за целью
    ESTOP = auto()      # Аварийная остановка
    ERROR = auto()      # Ошибка (требуется вмешательство)
    SHUTDOWN = auto()   # Завершение работы


# ════════════════════════════════════════════════════════════════
# Статус захвата цели
# ════════════════════════════════════════════════════════════════

class TargetLockState(Enum):
    """Статус захвата цели в автоматическом режиме."""
    NO_TARGET = auto()      # Цель не обнаружена
    SEARCHING = auto()      # Поиск цели (детекция работает)
    DETECTED = auto()       # Цель обнаружена, трек не подтверждён
    LOCKED = auto()         # Цель захвачена и отслеживается
    LOST = auto()           # Цель потеряна (была захвачена, пропала)


# ════════════════════════════════════════════════════════════════
# Общее состояние системы (shared state)
# ════════════════════════════════════════════════════════════════

@dataclass
class SharedState:
    """
    Общее состояние системы, доступное всем подсистемам.
    Является единственным источником истины (single source of truth).
    """

    # ── Режим работы ──
    platform_state: PlatformState = PlatformState.INIT
    target_lock: TargetLockState = TargetLockState.NO_TARGET

    # ── Позиция платформы (градусы) ──
    yaw_deg: float = 0.0           # Текущий угол горизонта
    pitch_deg: float = 0.0         # Текущий угол вертикали
    target_yaw_deg: float = 0.0    # Целевой угол горизонта
    target_pitch_deg: float = 0.0  # Целевой угол вертикали

    # ── Токи моторов ──
    yaw_current_a: float = 0.0
    pitch_current_a: float = 0.0

    # ── Дальномер ──
    distance_m: float = 0.0        # Дистанция до цели (м)
    distance_valid: bool = False   # Дистанция валидна

    # ── Камера ──
    zoom_level: float = 1.0       # Текущий зум
    camera_fps: float = 0.0       # FPS камеры

    # ── Детекция / трекинг ──
    target_bbox: Optional[tuple] = None    # (x1, y1, x2, y2) bounding box
    target_center_px: Optional[tuple] = None  # (cx, cy) центр цели в пикселях
    target_class_id: int = -1              # ID класса COCO
    target_confidence: float = 0.0         # Уверенность детекции
    track_id: int = -1                     # ID трека

    # ── Баллистика ──
    target_speed_mps: float = 0.0          # Скорость цели (м/с)
    target_heading_deg: float = 0.0        # Курс цели (градусы)
    time_of_flight_sec: float = 0.0        # Время полёта снаряда (сек)
    lead_yaw_deg: float = 0.0             # Угол упреждения по горизонту
    lead_pitch_deg: float = 0.0           # Угол упреждения по вертикали
    lead_point_px: Optional[tuple] = None  # Точка перехвата в пикселях (x, y)
    intercept_possible: bool = False       # Перехват возможен

    # ── Предсказание ──
    predicted_positions: List[tuple] = field(default_factory=list)  # [(x,y,z,t), ...]

    # ── Вооружение ──
    weapon_name: str = ""          # Название активного профиля (КОРД / ПКТ)
    weapon_caliber: str = ""       # Калибр (12.7×108 / 7.62×54R)
    bullet_energy_j: float = 0.0   # Энергия пули на дистанции (Дж)
    mach_at_target: float = 0.0    # Число Маха на дистанции
    effective_range_m: float = 2000.0  # Эффективная дальность текущего оружия (м)

    # ── Боевая готовность ──
    ready_to_fire: bool = False    # ГОТОВ К СТРЕЛЬБЕ (цель захвачена + дист. замерена + перехват возможен)
    in_effective_range: bool = False  # Цель в зоне поражения
    firing_active: bool = False    # Триггер удерживается прямо сейчас
    recoil_boost_active: bool = False  # Усиленные gains активны (гашение отдачи)
    shots_fired: int = 0           # Счётчик выстрелов (приблизительный, по rate_of_fire)
    burst_start_time: float = 0.0  # Время начала текущей очереди

    # ── Тип цели ──
    target_type_name: str = ""     # Русское название типа цели ("FPV-ДРОН", "БПЛА" и т.д.)
    threat_level: int = 0          # Уровень угрозы (0-10, FPV=10, птица=1)

    # ── Вектор движения цели ──
    target_velocity_px: Optional[tuple] = None  # Вектор скорости цели в пикселях (dx, dy) для стрелки HUD

    # ── Системное ──
    loop_time_ms: float = 0.0     # Время итерации главного цикла
    uptime_sec: float = 0.0       # Время работы
    error_message: str = ""       # Последнее сообщение об ошибке
    timestamp: float = 0.0        # Метка времени последнего обновления


# ════════════════════════════════════════════════════════════════
# Конечный автомат
# ════════════════════════════════════════════════════════════════

class StateMachine:
    """
    Конечный автомат управления режимами платформы.

    Определяет допустимые переходы между состояниями
    и вызывает callback-и при входе/выходе из состояний.
    """

    # Таблица допустимых переходов: из_состояния → [в_состояния]
    TRANSITIONS = {
        PlatformState.INIT: [
            PlatformState.MANUAL,
            PlatformState.ERROR,
            PlatformState.ESTOP,
            PlatformState.SHUTDOWN,
        ],
        PlatformState.MANUAL: [
            PlatformState.AUTO,
            PlatformState.ESTOP,
            PlatformState.ERROR,
            PlatformState.SHUTDOWN,
        ],
        PlatformState.AUTO: [
            PlatformState.MANUAL,
            PlatformState.ESTOP,
            PlatformState.ERROR,
            PlatformState.SHUTDOWN,
        ],
        PlatformState.ESTOP: [
            PlatformState.MANUAL,
            PlatformState.SHUTDOWN,
        ],
        PlatformState.ERROR: [
            PlatformState.MANUAL,
            PlatformState.SHUTDOWN,
        ],
        PlatformState.SHUTDOWN: [],  # Терминальное состояние
    }

    def __init__(self, shared_state: SharedState):
        """
        Args:
            shared_state: Общее состояние системы
        """
        self._state = shared_state
        self._current = PlatformState.INIT

        # Callback-и: on_enter_STATE, on_exit_STATE
        self._on_enter: Dict[PlatformState, List[Callable]] = {s: [] for s in PlatformState}
        self._on_exit: Dict[PlatformState, List[Callable]] = {s: [] for s in PlatformState}

        # Время входа в текущее состояние
        self._state_enter_time = time.time()

        logger.info(f"StateMachine: начальное состояние = {self._current.name}")

    @property
    def current(self) -> PlatformState:
        """Текущее состояние."""
        return self._current

    @property
    def state_duration(self) -> float:
        """Время в текущем состоянии (секунды)."""
        return time.time() - self._state_enter_time

    @property
    def is_manual(self) -> bool:
        return self._current == PlatformState.MANUAL

    @property
    def is_auto(self) -> bool:
        return self._current == PlatformState.AUTO

    @property
    def is_estop(self) -> bool:
        return self._current == PlatformState.ESTOP

    @property
    def is_operational(self) -> bool:
        """Платформа в рабочем режиме (Manual или Auto)."""
        return self._current in (PlatformState.MANUAL, PlatformState.AUTO)

    # ── Регистрация callback-ов ─────────────────────────────

    def on_enter(self, state: PlatformState, callback: Callable):
        """Зарегистрировать callback при ВХОДЕ в состояние."""
        self._on_enter[state].append(callback)

    def on_exit(self, state: PlatformState, callback: Callable):
        """Зарегистрировать callback при ВЫХОДЕ из состояния."""
        self._on_exit[state].append(callback)

    # ── Переходы ────────────────────────────────────────────

    def transition_to(self, new_state: PlatformState, reason: str = "") -> bool:
        """
        Выполнить переход в новое состояние.

        Args:
            new_state: Целевое состояние
            reason: Причина перехода (для логирования)

        Returns:
            True если переход выполнен, False если недопустим
        """
        if new_state == self._current:
            return True  # Уже в этом состоянии

        # Проверка допустимости перехода
        allowed = self.TRANSITIONS.get(self._current, [])
        if new_state not in allowed:
            logger.warning(
                f"StateMachine: переход {self._current.name} → {new_state.name} "
                f"НЕДОПУСТИМ (причина: {reason})"
            )
            return False

        old_state = self._current

        # Callback-и выхода из старого состояния
        for cb in self._on_exit[old_state]:
            try:
                cb()
            except Exception as e:
                logger.error(f"Ошибка в on_exit({old_state.name}): {e}")

        # Переход
        self._current = new_state
        self._state_enter_time = time.time()
        self._state.platform_state = new_state

        # Callback-и входа в новое состояние
        for cb in self._on_enter[new_state]:
            try:
                cb()
            except Exception as e:
                logger.error(f"Ошибка в on_enter({new_state.name}): {e}")

        logger.info(
            f"StateMachine: {old_state.name} → {new_state.name}"
            f"{f' ({reason})' if reason else ''}"
        )

        return True

    def toggle_mode(self):
        """
        Переключить между Manual и Auto режимами.
        Вызывается по нажатию кнопки на джойстике.
        """
        if self._current == PlatformState.MANUAL:
            self.transition_to(PlatformState.AUTO, "кнопка переключения режима")
        elif self._current == PlatformState.AUTO:
            self.transition_to(PlatformState.MANUAL, "кнопка переключения режима")
        else:
            logger.warning(
                f"StateMachine: переключение режима невозможно "
                f"из состояния {self._current.name}"
            )

    def emergency_stop(self):
        """Аварийная остановка — переход в ESTOP из любого состояния."""
        if self._current != PlatformState.ESTOP:
            self.transition_to(PlatformState.ESTOP, "АВАРИЙНАЯ ОСТАНОВКА")

    def recover_from_estop(self):
        """Восстановление после аварийной остановки → Manual."""
        if self._current == PlatformState.ESTOP:
            self.transition_to(PlatformState.MANUAL, "восстановление после E-STOP")

    def report_error(self, message: str):
        """Сообщить об ошибке и перейти в ERROR."""
        self._state.error_message = message
        self.transition_to(PlatformState.ERROR, f"ошибка: {message}")

    # ── Отображение ─────────────────────────────────────────

    def get_mode_display_name(self) -> str:
        """Получить отображаемое имя текущего режима (на русском)."""
        names = {
            PlatformState.INIT: "ИНИЦИАЛИЗАЦИЯ",
            PlatformState.MANUAL: "РУЧНОЙ",
            PlatformState.AUTO: "АВТО",
            PlatformState.ESTOP: "АВАРИЙНАЯ ОСТАНОВКА",
            PlatformState.ERROR: "ОШИБКА",
            PlatformState.SHUTDOWN: "ЗАВЕРШЕНИЕ",
        }
        return names.get(self._current, "НЕИЗВЕСТНО")

    def get_lock_display_name(self) -> str:
        """Получить отображаемое имя статуса захвата (на русском)."""
        names = {
            TargetLockState.NO_TARGET: "НЕТ ЦЕЛИ",
            TargetLockState.SEARCHING: "ПОИСК",
            TargetLockState.DETECTED: "ОБНАРУЖЕНА",
            TargetLockState.LOCKED: "ЗАХВАТ",
            TargetLockState.LOST: "ПОТЕРЯНА",
        }
        return names.get(self._state.target_lock, "—")
