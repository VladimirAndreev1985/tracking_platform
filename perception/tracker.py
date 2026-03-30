# -*- coding: utf-8 -*-
"""
Менеджер отслеживания целей.

Использует встроенный BoT-SORT трекер YOLO26 (Ultralytics)
через model.track() — полноценный трекинг из коробки:
  - Фильтр Калмана для предсказания позиции между кадрами
  - Re-ID (повторная идентификация после потери)
  - Camera Motion Compensation (CMC)
  - Автоматическое присвоение track_id

Этот модуль НЕ реализует собственный трекинг.
Он управляет логикой захвата/сброса цели и предоставляет
удобный API для platform_controller.

Детекция + трекинг выполняется в detector.py через:
    detections = detector.detect_and_track(frame)
    # Каждая Detection содержит track_id из BoT-SORT
"""

import time
import logging
from dataclasses import dataclass
from typing import List, Optional

from perception.detector import Detection

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Трек (отслеживаемый объект) — view над Detection
# ════════════════════════════════════════════════════════════════

@dataclass
class Track:
    """Один отслеживаемый объект (из YOLO26 BoT-SORT)."""
    track_id: int                  # Уникальный ID трека (из YOLO26)
    bbox: tuple = (0, 0, 0, 0)    # (x1, y1, x2, y2)
    confidence: float = 0.0       # Уверенность детекции
    class_id: int = -1            # ID класса COCO
    is_confirmed: bool = True     # Треки из YOLO26 всегда подтверждены

    @property
    def center(self) -> tuple:
        """Центр bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


# ════════════════════════════════════════════════════════════════
# Менеджер целей
# ════════════════════════════════════════════════════════════════

class TargetManager:
    """
    Менеджер отслеживания и захвата целей.

    Принимает детекции с track_id из YOLO26 BoT-SORT
    и управляет логикой захвата/сброса/выбора цели.

    Использование:
        # В главном цикле:
        detections = detector.detect_and_track(frame)
        tracks = target_manager.update(detections)
        locked = target_manager.locked_track  # Захваченная цель
    """

    def __init__(self):
        # Текущие треки (из последнего кадра)
        self._tracks: List[Track] = []
        # ID захваченной цели
        self._locked_track_id: int = -1
        # Счётчик кадров без захваченной цели
        self._lost_frames: int = 0
        # Макс. кадров потери до сброса захвата
        self._max_lost_frames: int = 30

        logger.info("TargetManager: используется YOLO26 встроенный BoT-SORT")

    @property
    def tracks(self) -> List[Track]:
        """Все активные треки текущего кадра."""
        return self._tracks

    @property
    def locked_track(self) -> Optional[Track]:
        """Захваченная (выбранная) цель."""
        if self._locked_track_id < 0:
            return None
        for t in self._tracks:
            if t.track_id == self._locked_track_id:
                self._lost_frames = 0
                return t
        # Трек не найден в текущем кадре
        self._lost_frames += 1
        if self._lost_frames > self._max_lost_frames:
            logger.warning(
                f"Цель track_id={self._locked_track_id} потеряна "
                f"({self._lost_frames} кадров)"
            )
            self._locked_track_id = -1
        return None

    @property
    def is_target_locked(self) -> bool:
        """Есть ли захваченная цель."""
        return self.locked_track is not None

    # ── Обновление ──────────────────────────────────────────

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Обновить список треков из детекций YOLO26 BoT-SORT.

        Args:
            detections: Детекции с track_id из detector.detect_and_track()

        Returns:
            Список треков
        """
        self._tracks = []

        for det in detections:
            tid = getattr(det, 'track_id', -1)
            if tid < 0:
                continue  # Пропускаем детекции без track_id

            self._tracks.append(Track(
                track_id=tid,
                bbox=det.to_xyxy(),
                confidence=det.confidence,
                class_id=det.class_id,
                is_confirmed=True,
            ))

        return self._tracks

    # ── Захват цели ─────────────────────────────────────────

    def lock_target(self, track_id: int = None):
        """
        Захватить цель для автослежения.

        Args:
            track_id: ID трека. Если None — автоматически выбрать
                      ближайший к центру или с наибольшей уверенностью.
        """
        if track_id is not None:
            self._locked_track_id = track_id
            self._lost_frames = 0
            logger.info(f"Цель захвачена: track_id={track_id}")
            return

        # Автовыбор — трек с наибольшей уверенностью
        if not self._tracks:
            logger.warning("Нет треков для захвата")
            return

        best = max(self._tracks, key=lambda t: t.confidence)
        self._locked_track_id = best.track_id
        self._lost_frames = 0
        logger.info(
            f"Цель захвачена автоматически: track_id={best.track_id}, "
            f"класс={best.class_id}, уверенность={best.confidence:.2f}"
        )

    def release_target(self):
        """Сбросить захват цели."""
        if self._locked_track_id >= 0:
            logger.info(f"Захват сброшен: track_id={self._locked_track_id}")
        self._locked_track_id = -1
        self._lost_frames = 0

    def toggle_lock(self):
        """Переключить захват: если есть — сбросить, если нет — захватить."""
        if self.is_target_locked:
            self.release_target()
        else:
            self.lock_target()

    def reset(self):
        """Сбросить все треки и захват."""
        self._tracks.clear()
        self._locked_track_id = -1
        self._lost_frames = 0
        logger.debug("TargetManager: сброс")
