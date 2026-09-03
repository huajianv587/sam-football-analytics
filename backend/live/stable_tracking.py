"""Stable public IDs and short-gap box/mask prediction for live streams."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


BBox = tuple[float, float, float, float]
Point = tuple[float, float]


@dataclass(frozen=True)
class RawDetection:
    raw_id: int
    bbox: BBox
    confidence: float
    class_name: str
    polygon: list[Point] = field(default_factory=list)


@dataclass(frozen=True)
class StableDetection:
    track_id: int
    raw_tracker_id: int | None
    bbox: BBox
    confidence: float
    class_name: str
    polygon: list[Point]
    state: str
    first_frame: int
    last_seen_frame: int
    reassociation_count: int


@dataclass
class _Track:
    track_id: int
    bbox: BBox
    confidence: float
    class_name: str
    polygon: list[Point]
    first_frame: int
    last_seen_frame: int
    raw_tracker_id: int | None
    missing_frames: int = 0
    reassociation_count: int = 0
    kalman: cv2.KalmanFilter | None = None
    predicted_bbox: BBox | None = None


def _center(box: BBox) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _iou(first: BBox, second: BBox) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    overlap = _area((x1, y1, x2, y2))
    return overlap / max(1e-6, _area(first) + _area(second) - overlap)


def _new_filter(box: BBox) -> cv2.KalmanFilter:
    kalman = cv2.KalmanFilter(8, 4)
    kalman.transitionMatrix = np.eye(8, dtype=np.float32)
    kalman.transitionMatrix[:4, 4:] = np.eye(4, dtype=np.float32)
    kalman.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
    kalman.measurementMatrix[:, :4] = np.eye(4, dtype=np.float32)
    kalman.processNoiseCov = np.eye(8, dtype=np.float32) * 0.02
    kalman.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.08
    kalman.errorCovPost = np.eye(8, dtype=np.float32)
    kalman.statePost[:4, 0] = np.asarray(box, dtype=np.float32)
    return kalman


def _filter_box(kalman: cv2.KalmanFilter) -> BBox:
    values = kalman.statePost[:4, 0].tolist()
    x1, y1, x2, y2 = (float(value) for value in values)
    if x2 <= x1 or y2 <= y1:
        return (x1, y1, x1 + 1.0, y1 + 1.0)
    return (x1, y1, x2, y2)


class StableTrackRegistry:
    """Map transient detector IDs to monotonic IDs and predict short gaps."""

    def __init__(self, max_missing_frames: int = 45) -> None:
        self.max_missing_frames = max_missing_frames
        self.next_track_id = 1
        self.tracks: dict[int, _Track] = {}
        self.raw_to_public: dict[int, int] = {}

    def reset(self) -> None:
        self.next_track_id = 1
        self.tracks.clear()
        self.raw_to_public.clear()

    def _predict(self, track: _Track) -> BBox:
        if track.kalman is None:
            track.kalman = _new_filter(track.bbox)
        predicted = track.kalman.predict()
        values = predicted[:4, 0].tolist()
        x1, y1, x2, y2 = (float(value) for value in values)
        if x2 <= x1 or y2 <= y1:
            return track.bbox
        track.predicted_bbox = (x1, y1, x2, y2)
        return track.predicted_bbox

    def _correct(self, track: _Track, bbox: BBox) -> None:
        if track.kalman is None:
            track.kalman = _new_filter(bbox)
        track.kalman.correct(np.asarray(bbox, dtype=np.float32).reshape(4, 1))
        track.predicted_bbox = bbox

    def _candidate_score(self, track: _Track, detection: RawDetection) -> float | None:
        predicted = track.predicted_bbox or track.bbox
        overlap = _iou(predicted, detection.bbox)
        px, py = _center(predicted)
        dx, dy = _center(detection.bbox)
        distance = float(np.hypot(px - dx, py - dy))
        height = max(4.0, predicted[3] - predicted[1], detection.bbox[3] - detection.bbox[1])
        if overlap < 0.05 and distance > max(80.0, height * 2.5):
            return None
        if track.class_name != detection.class_name:
            return None
        return overlap + max(0.0, 1.0 - distance / (height * 3.0)) * 0.25

    def _make_new(self, detection: RawDetection, frame_id: int) -> _Track:
        track = _Track(
            track_id=self.next_track_id,
            bbox=detection.bbox,
            confidence=detection.confidence,
            class_name=detection.class_name,
            polygon=list(detection.polygon),
            first_frame=frame_id,
            last_seen_frame=frame_id,
            raw_tracker_id=detection.raw_id,
            kalman=_new_filter(detection.bbox),
        )
        self.next_track_id += 1
        self.tracks[track.track_id] = track
        self.raw_to_public[detection.raw_id] = track.track_id
        return track

    def update(self, frame_id: int, detections: list[RawDetection]) -> list[StableDetection]:
        for track in self.tracks.values():
            self._predict(track)

        assigned: dict[int, tuple[RawDetection, _Track]] = {}
        used: set[int] = set()
        unmatched: list[RawDetection] = []
        for detection in detections:
            public_id = self.raw_to_public.get(detection.raw_id)
            track = self.tracks.get(public_id) if public_id is not None else None
            if track is not None and track.track_id not in used:
                assigned[track.track_id] = (detection, track)
                used.add(track.track_id)
            else:
                unmatched.append(detection)

        for detection in unmatched:
            candidates = (
                track
                for track in self.tracks.values()
                if track.track_id not in used
            )
            scored = sorted(
                (
                    (self._candidate_score(track, detection), track)
                    for track in candidates
                ),
                key=lambda item: item[0] if item[0] is not None else -1,
                reverse=True,
            )
            best_score, best_track = scored[0] if scored else (None, None)
            if best_track is not None and best_score is not None:
                assigned[best_track.track_id] = (detection, best_track)
                used.add(best_track.track_id)
                self.raw_to_public[detection.raw_id] = best_track.track_id
            else:
                new_track = self._make_new(detection, frame_id)
                assigned[new_track.track_id] = (detection, new_track)
                used.add(new_track.track_id)

        output: list[StableDetection] = []
        for track_id, (detection, track) in assigned.items():
            previous_raw_id = track.raw_tracker_id
            if previous_raw_id != detection.raw_id and track.last_seen_frame < frame_id:
                # Count each raw-id change once, at the point of assignment.
                track.reassociation_count += 1
            track.bbox = detection.bbox
            track.confidence = detection.confidence
            track.class_name = detection.class_name
            track.polygon = list(detection.polygon)
            track.raw_tracker_id = detection.raw_id
            track.last_seen_frame = frame_id
            track.missing_frames = 0
            self._correct(track, detection.bbox)
            self.raw_to_public[detection.raw_id] = track.track_id
            output.append(StableDetection(
                track_id=track.track_id,
                raw_tracker_id=detection.raw_id,
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_name=detection.class_name,
                polygon=list(detection.polygon),
                state="detected",
                first_frame=track.first_frame,
                last_seen_frame=track.last_seen_frame,
                reassociation_count=track.reassociation_count,
            ))

        observed_ids = {item.track_id for item in output}
        for track_id, track in list(self.tracks.items()):
            if track_id in observed_ids:
                continue
            track.missing_frames += 1
            if track.missing_frames > self.max_missing_frames:
                self.tracks.pop(track_id, None)
                for raw_id, public_id in list(self.raw_to_public.items()):
                    if public_id == track_id:
                        self.raw_to_public.pop(raw_id, None)
                continue
            previous = track.bbox
            predicted = track.predicted_bbox or previous
            dx = predicted[0] - previous[0]
            dy = predicted[1] - previous[1]
            polygon = [(round(x + dx, 1), round(y + dy, 1)) for x, y in track.polygon]
            track.bbox = predicted
            output.append(StableDetection(
                track_id=track.track_id,
                raw_tracker_id=track.raw_tracker_id,
                bbox=predicted,
                confidence=track.confidence,
                class_name=track.class_name,
                polygon=polygon,
                state="predicted",
                first_frame=track.first_frame,
                last_seen_frame=track.last_seen_frame,
                reassociation_count=track.reassociation_count,
            ))
        return sorted(output, key=lambda item: item.track_id)
