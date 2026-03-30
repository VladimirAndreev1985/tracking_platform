# -*- coding: utf-8 -*-
"""
Трекер объектов BoT-SORT.

Связывает детекции между кадрами, присваивая каждому
объекту уникальный track_id. Обеспечивает устойчивое
отслеживание даже при кратковременной потере детекции.

Реализация: обёртка над BoT-SORT из библиотеки
boxmot (или упрощённая реализация на базе IoU + Kalman).
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from perception.detector import Detection

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Трек (отслеживаемый объект)
# ════════════════════════════════════════════════════════════════

@dataclass
class Track:
    """Один отслеживаемый объект."""
    track_id: int                  # Уникальный ID трека
    bbox: tuple = (0, 0, 0, 0)    # (x1, y1, x2, y2)
    confidence: float = 0.0       # Уверенность последней детекции
    class_id: int = -1            # ID класса COCO
    age: int = 0                  # Количество кадров с момента создания
    hits: int = 0                 # Количество успешных ассоциаций
    time_since_update: int = 0    # Кадров с последнего обновления
    is_confirmed: bool = False    # Трек подтверждён (hits >= min_hits)

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
# Утилиты IoU
# ════════════════════════════════════════════════════════════════

def _iou(box_a: tuple, box_b: tuple) -> float:
    """
    Вычислить IoU (Intersection over Union) двух bounding box.

    Args:
        box_a: (x1, y1, x2, y2)
        box_b: (x1, y1, x2, y2)

    Returns:
        IoU от 0.0 до 1.0
    """
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def _iou_matrix(tracks: List[Track], detections: List[Detection]) -> np.ndarray:
    """Матрица IoU между треками и детекциями."""
    n_tracks = len(tracks)
    n_dets = len(detections)
    matrix = np.zeros((n_tracks, n_dets))

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            matrix[i, j] = _iou(track.bbox, det.to_xyxy())

    return matrix


# ════════════════════════════════════════════════════════════════
# Трекер
# ════════════════════════════════════════════════════════════════

class ObjectTracker:
    """
    Трекер объектов на базе IoU-ассоциации.

    Упрощённая реализация BoT-SORT:
    1. Ассоциация детекций с существующими треками по IoU
    2. Создание новых треков для неассоциированных детекций
    3. Удаление треков, потерянных на max_age кадров
    4. Подтверждение треков после min_hits детекций

    Для полноценного BoT-SORT с Re-ID и CMC используйте
    библиотеку boxmot (pip install boxmot).
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        """
        Args:
            max_age: Макс. кадров без детекции до удаления трека
            min_hits: Мин. детекций для подтверждения трека
            iou_threshold: Порог IoU для ассоциации
        """
        self._max_age = max_age
        self._min_hits = min_hits
        self._iou_threshold = iou_threshold

        # Активные треки
        self._tracks: List[Track] = []
        # Счётчик ID
        self._next_id = 1

        # Выбранная (захваченная) цель
        self._locked_track_id: int = -1

        logger.info(
            f"ObjectTracker: max_age={max_age}, min_hits={min_hits}, "
            f"iou_threshold={iou_threshold}"
        )

    @property
    def tracks(self) -> List[Track]:
        """Все активные треки."""
        return self._tracks

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Только подтверждённые треки."""
        return [t for t in self._tracks if t.is_confirmed]

    @property
    def locked_track(self) -> Optional[Track]:
        """Захваченный (выбранный) трек."""
        if self._locked_track_id < 0:
            return None
        for t in self._tracks:
            if t.track_id == self._locked_track_id:
                return t
        # Трек потерян
        self._locked_track_id = -1
        return None

    @property
    def is_target_locked(self) -> bool:
        """Есть ли захваченная цель."""
        return self.locked_track is not None

    # ── Обновление ──────────────────────────────────────────

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Обновить треки новыми детекциями.

        Args:
            detections: Список детекций текущего кадра

        Returns:
            Список подтверждённых треков
        """
        # Шаг 1: Ассоциация детекций с треками по IoU
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self._tracks, detections
        )

        # Шаг 2: Обновить ассоциированные треки
        for track_idx, det_idx in matched:
            track = self._tracks[track_idx]
            det = detections[det_idx]
            track.bbox = det.to_xyxy()
            track.confidence = det.confidence
            track.class_id = det.class_id
            track.hits += 1
            track.time_since_update = 0
            if track.hits >= self._min_hits:
                track.is_confirmed = True

        # Шаг 3: Увеличить возраст неассоциированных треков
        for track_idx in unmatched_tracks:
            self._tracks[track_idx].time_since_update += 1

        # Шаг 4: Создать новые треки для неассоциированных детекций
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = Track(
                track_id=self._next_id,
                bbox=det.to_xyxy(),
                confidence=det.confidence,
                class_id=det.class_id,
                hits=1,
                age=0,
                time_since_update=0,
            )
            self._tracks.append(new_track)
            self._next_id += 1

        # Шаг 5: Удалить потерянные треки
        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self._max_age
        ]

        # Шаг 6: Увеличить возраст всех треков
        for track in self._tracks:
            track.age += 1

        return self.confirmed_tracks

    def _associate(self, tracks: List[Track],
                   detections: List[Detection]):
        """
        Ассоциировать детекции с треками по IoU (жадный алгоритм).

        Returns:
            (matched, unmatched_tracks, unmatched_detections)
        """
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        iou_mat = _iou_matrix(tracks, detections)

        matched = []
        used_tracks = set()
        used_dets = set()

        # Жадная ассоциация: берём пары с максимальным IoU
        while True:
            if iou_mat.size == 0:
                break

            # Найти максимальный IoU
            max_iou = iou_mat.max()
            if max_iou < self._iou_threshold:
                break

            # Индексы максимума
            t_idx, d_idx = np.unravel_index(iou_mat.argmax(), iou_mat.shape)

            matched.append((t_idx, d_idx))
            used_tracks.add(t_idx)
            used_dets.add(d_idx)

            # Обнулить строку и столбец
            iou_mat[t_idx, :] = 0
            iou_mat[:, d_idx] = 0

        unmatched_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]

        return matched, unmatched_tracks, unmatched_dets

    # ── Захват цели ─────────────────────────────────────────

    def lock_target(self, track_id: int = None):
        """
        Захватить цель (выбрать трек для автослежения).

        Args:
            track_id: ID трека. Если None — захватить ближайший
                      к центру кадра подтверждённый трек.
        """
        if track_id is not None:
            self._locked_track_id = track_id
            logger.info(f"Цель захвачена: track_id={track_id}")
            return

        # Автовыбор — ближайший к центру
        confirmed = self.confirmed_tracks
        if not confirmed:
            logger.warning("Нет подтверждённых треков для захвата")
            return

        # Берём трек с наибольшей уверенностью
        best = max(confirmed, key=lambda t: t.confidence)
        self._locked_track_id = best.track_id
        logger.info(
            f"Цель захвачена автоматически: track_id={best.track_id}, "
            f"класс={best.class_id}, уверенность={best.confidence:.2f}"
        )

    def release_target(self):
        """Сбросить захват цели."""
        if self._locked_track_id >= 0:
            logger.info(f"Захват сброшен: track_id={self._locked_track_id}")
        self._locked_track_id = -1

    def toggle_lock(self):
        """Переключить захват: если есть — сбросить, если нет — захватить."""
        if self.is_target_locked:
            self.release_target()
        else:
            self.lock_target()

    # ── Сброс ───────────────────────────────────────────────

    def reset(self):
        """Сбросить все треки."""
        self._tracks.clear()
        self._locked_track_id = -1
        self._next_id = 1
        logger.debug("ObjectTracker: сброс")
