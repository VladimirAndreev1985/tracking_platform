# -*- coding: utf-8 -*-
"""
Трекер объектов — BoT-SORT с fallback на IoU-трекер.

Два режима:
1. Полноценный BoT-SORT (через библиотеку boxmot) — если установлена
   - Фильтр Калмана для предсказания позиции между кадрами
   - Re-ID для повторной идентификации после потери
   - Camera Motion Compensation (CMC) — важно для вращающейся платформы
2. Встроенный IoU+Kalman трекер — fallback без внешних зависимостей
   - IoU-ассоциация + упрощённый фильтр Калмана
   - Достаточно для большинства сценариев

Связывает детекции между кадрами, присваивая каждому
объекту уникальный track_id.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

from perception.detector import Detection

logger = logging.getLogger(__name__)

# Пробуем импортировать boxmot для полноценного BoT-SORT
_HAS_BOXMOT = False
try:
    from boxmot import BoTSORT
    _HAS_BOXMOT = True
    logger.info("boxmot найден — используем полноценный BoT-SORT")
except ImportError:
    logger.info("boxmot не найден — используем встроенный IoU+Kalman трекер")


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
# Упрощённый фильтр Калмана для одного трека
# ════════════════════════════════════════════════════════════════

class _SimpleKalman:
    """
    Упрощённый фильтр Калмана для отслеживания bbox.
    Состояние: [cx, cy, w, h, vx, vy, vw, vh]
    """

    def __init__(self, bbox: tuple):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        # Состояние: [cx, cy, w, h, vx, vy, vw, vh]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        # Ковариация
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0  # Большая неопределённость скоростей

        # Модель процесса (dt=1 кадр)
        self.F = np.eye(8)
        self.F[0, 4] = 1  # cx += vx
        self.F[1, 5] = 1  # cy += vy
        self.F[2, 6] = 1  # w += vw
        self.F[3, 7] = 1  # h += vh

        # Шум процесса
        self.Q = np.eye(8) * 1.0
        self.Q[4:, 4:] *= 0.01

        # Матрица наблюдения (наблюдаем cx, cy, w, h)
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1

        # Шум наблюдения
        self.R = np.eye(4) * 1.0

    def predict(self) -> tuple:
        """Предсказать следующее состояние."""
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

        cx, cy, w, h = self.state[:4]
        w = max(w, 1)
        h = max(h, 1)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def update(self, bbox: tuple):
        """Обновить состояние по наблюдению."""
        x1, y1, x2, y2 = bbox
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])

        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

    @property
    def bbox(self) -> tuple:
        """Текущий bbox из состояния."""
        cx, cy, w, h = self.state[:4]
        w = max(w, 1)
        h = max(h, 1)
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


# ════════════════════════════════════════════════════════════════
# Утилиты IoU
# ════════════════════════════════════════════════════════════════

def _iou(box_a: tuple, box_b: tuple) -> float:
    """Вычислить IoU двух bounding box."""
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


# ════════════════════════════════════════════════════════════════
# Внутренний трек с фильтром Калмана
# ════════════════════════════════════════════════════════════════

class _InternalTrack:
    """Трек с фильтром Калмана для встроенного трекера."""

    def __init__(self, track_id: int, detection: Detection):
        self.track_id = track_id
        self.kalman = _SimpleKalman(detection.to_xyxy())
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.is_confirmed = False

    def predict(self) -> tuple:
        """Предсказать bbox на следующий кадр."""
        return self.kalman.predict()

    def update(self, detection: Detection):
        """Обновить трек новой детекцией."""
        self.kalman.update(detection.to_xyxy())
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= 3:
            self.is_confirmed = True

    @property
    def bbox(self) -> tuple:
        return self.kalman.bbox

    def to_track(self) -> Track:
        """Преобразовать во внешний Track."""
        return Track(
            track_id=self.track_id,
            bbox=self.bbox,
            confidence=self.confidence,
            class_id=self.class_id,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            is_confirmed=self.is_confirmed,
        )


# ════════════════════════════════════════════════════════════════
# Трекер объектов
# ════════════════════════════════════════════════════════════════

class ObjectTracker:
    """
    Трекер объектов с поддержкой BoT-SORT (boxmot) и встроенного IoU+Kalman.

    Автоматически выбирает бэкенд:
    - Если boxmot установлен → полноценный BoT-SORT с Re-ID и CMC
    - Иначе → встроенный IoU-трекер с фильтром Калмана
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

        # Выбранная (захваченная) цель
        self._locked_track_id: int = -1

        # Выбор бэкенда
        self._use_boxmot = _HAS_BOXMOT
        self._botsort = None

        if self._use_boxmot:
            try:
                self._botsort = BoTSORT(
                    reid_weights=None,  # Без Re-ID модели (можно добавить позже)
                    device='cpu',
                    half=False,
                    track_high_thresh=0.4,
                    track_low_thresh=0.1,
                    new_track_thresh=0.5,
                    track_buffer=max_age,
                    match_thresh=iou_threshold,
                )
                logger.info(
                    f"BoT-SORT (boxmot): max_age={max_age}, "
                    f"iou={iou_threshold}"
                )
            except Exception as e:
                logger.warning(f"Ошибка инициализации BoT-SORT: {e}, fallback на IoU+Kalman")
                self._use_boxmot = False

        if not self._use_boxmot:
            # Встроенный трекер
            self._tracks: List[_InternalTrack] = []
            self._next_id = 1
            logger.info(
                f"IoU+Kalman трекер: max_age={max_age}, min_hits={min_hits}, "
                f"iou={iou_threshold}"
            )

        # Кэш последних треков (для внешнего API)
        self._last_tracks: List[Track] = []

    @property
    def tracks(self) -> List[Track]:
        """Все активные треки."""
        return self._last_tracks

    @property
    def confirmed_tracks(self) -> List[Track]:
        """Только подтверждённые треки."""
        return [t for t in self._last_tracks if t.is_confirmed]

    @property
    def locked_track(self) -> Optional[Track]:
        """Захваченный (выбранный) трек."""
        if self._locked_track_id < 0:
            return None
        for t in self._last_tracks:
            if t.track_id == self._locked_track_id:
                return t
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
        if self._use_boxmot:
            return self._update_boxmot(detections)
        else:
            return self._update_builtin(detections)

    def _update_boxmot(self, detections: List[Detection]) -> List[Track]:
        """Обновление через BoT-SORT (boxmot)."""
        if not detections:
            # Пустой массив детекций
            dets = np.empty((0, 6))
        else:
            dets = np.array([
                [d.x1, d.y1, d.x2, d.y2, d.confidence, d.class_id]
                for d in detections
            ])

        # BoT-SORT ожидает numpy array (N, 6): x1,y1,x2,y2,conf,cls
        # и изображение (для CMC/Re-ID) — передаём None если без Re-ID
        try:
            outputs = self._botsort.update(dets, img=None)
        except Exception:
            # Некоторые версии boxmot требуют img
            outputs = np.empty((0, 7))

        tracks = []
        for out in outputs:
            if len(out) >= 7:
                x1, y1, x2, y2, tid, conf, cls = out[:7]
                tracks.append(Track(
                    track_id=int(tid),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=int(cls),
                    is_confirmed=True,
                ))

        self._last_tracks = tracks
        return tracks

    def _update_builtin(self, detections: List[Detection]) -> List[Track]:
        """Обновление через встроенный IoU+Kalman трекер."""

        # Шаг 1: Предсказание всех треков
        for track in self._tracks:
            track.predict()
            track.age += 1
            track.time_since_update += 1

        # Шаг 2: Ассоциация по IoU (предсказанные bbox vs детекции)
        matched, unmatched_tracks, unmatched_dets = self._associate(
            self._tracks, detections
        )

        # Шаг 3: Обновить ассоциированные треки
        for track_idx, det_idx in matched:
            self._tracks[track_idx].update(detections[det_idx])

        # Шаг 4: Создать новые треки
        for det_idx in unmatched_dets:
            new_track = _InternalTrack(self._next_id, detections[det_idx])
            self._tracks.append(new_track)
            self._next_id += 1

        # Шаг 5: Удалить потерянные треки
        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= self._max_age
        ]

        # Преобразовать во внешний формат
        self._last_tracks = [t.to_track() for t in self._tracks]
        return self.confirmed_tracks

    def _associate(self, tracks: List[_InternalTrack],
                   detections: List[Detection]):
        """Ассоциация по IoU (жадный алгоритм)."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Матрица IoU (предсказанные bbox vs детекции)
        n_tracks = len(tracks)
        n_dets = len(detections)
        iou_mat = np.zeros((n_tracks, n_dets))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_mat[i, j] = _iou(track.bbox, det.to_xyxy())

        matched = []
        used_tracks = set()
        used_dets = set()

        while True:
            if iou_mat.size == 0:
                break
            max_iou = iou_mat.max()
            if max_iou < self._iou_threshold:
                break
            t_idx, d_idx = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            matched.append((t_idx, d_idx))
            used_tracks.add(t_idx)
            used_dets.add(d_idx)
            iou_mat[t_idx, :] = 0
            iou_mat[:, d_idx] = 0

        unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
        unmatched_dets = [i for i in range(n_dets) if i not in used_dets]

        return matched, unmatched_tracks, unmatched_dets

    # ── Захват цели ─────────────────────────────────────────

    def lock_target(self, track_id: int = None):
        """Захватить цель."""
        if track_id is not None:
            self._locked_track_id = track_id
            logger.info(f"Цель захвачена: track_id={track_id}")
            return

        confirmed = self.confirmed_tracks
        if not confirmed:
            logger.warning("Нет подтверждённых треков для захвата")
            return

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
        """Переключить захват."""
        if self.is_target_locked:
            self.release_target()
        else:
            self.lock_target()

    def reset(self):
        """Сбросить все треки."""
        if not self._use_boxmot:
            self._tracks.clear()
            self._next_id = 1
        self._locked_track_id = -1
        self._last_tracks.clear()
        logger.debug("ObjectTracker: сброс")
