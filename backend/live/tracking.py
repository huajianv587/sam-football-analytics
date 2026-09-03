from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionSample:
    timestamp: float
    point: tuple[float, float]
    speed_px_s: float


class MotionHistory:
    def __init__(
        self, history_size: int = 90, alpha: float = 0.35, max_missing_frames: int = 15
    ) -> None:
        self.history_size = history_size
        self.alpha = alpha
        self.max_missing_frames = max_missing_frames
        self.samples: dict[int, deque[MotionSample]] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self.missing_frames: dict[int, int] = {}

    def update(
        self, track_id: int, timestamp: float, bbox: tuple[float, float, float, float]
    ) -> MotionSample:
        point = ((bbox[0] + bbox[2]) / 2, bbox[3])
        history = self.samples[track_id]
        speed = 0.0
        if history:
            previous = history[-1]
            elapsed = timestamp - previous.timestamp
            if elapsed > 1e-3:
                raw = float(np.linalg.norm(np.subtract(point, previous.point)) / elapsed)
                speed = self.alpha * raw + (1 - self.alpha) * previous.speed_px_s
        sample = MotionSample(timestamp=timestamp, point=point, speed_px_s=speed)
        history.append(sample)
        self.missing_frames[track_id] = 0
        return sample

    def trail(self, track_id: int, limit: int = 45) -> list[tuple[float, float]]:
        return [sample.point for sample in list(self.samples[track_id])[-limit:]]

    def retain(self, active_ids: set[int]) -> None:
        for track_id in set(self.samples) - active_ids:
            missing = self.missing_frames.get(track_id, 0) + 1
            self.missing_frames[track_id] = missing
            if missing > self.max_missing_frames:
                del self.samples[track_id]
                del self.missing_frames[track_id]

    def clear(self) -> None:
        self.samples.clear()
        self.missing_frames.clear()


def largest_polygon(mask: np.ndarray, epsilon_ratio: float = 0.004) -> list[tuple[float, float]]:
    binary = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [(round(float(x), 1), round(float(y), 1)) for x, y in simplified]


def simplify_polygon(
    polygon: np.ndarray, epsilon_ratio: float = 0.004
) -> list[tuple[float, float]]:
    if not len(polygon):
        return []
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    return [(round(float(x), 1), round(float(y), 1)) for x, y in points]
