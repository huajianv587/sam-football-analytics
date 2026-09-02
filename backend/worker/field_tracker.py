"""Lightweight field-registered association for broadcast football footage.

The detector intentionally keeps low-confidence people. Association happens in
two passes: reliable detections first, then low-score detections that can recover
an existing track through short occlusions. Expensive person ReID is replaced by
a 5 FPS colour descriptor and carries only five percent of the match score.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .analytics import project_point
from .game_state import bbox_iou


FIELD_WIDTH_M = 105.0
FIELD_HEIGHT_M = 68.0


def _normalise_homography(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.size != 9:
        return None
    matrix = matrix.reshape(3, 3)
    if not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-9:
        return None
    return matrix / matrix[2, 2]


def interpolate_calibrations(
    state: dict[str, Any], target_frame_count: int, target_fps: float
) -> list[dict[str, Any]]:
    """Interpolate low-rate camera estimates and smooth their coefficients."""
    source_fps = float(state.get("fps") or target_fps)
    observations: list[tuple[float, np.ndarray, float]] = []
    for frame in state.get("frames", []):
        matrix = _normalise_homography(frame.get("homography"))
        confidence = float(frame.get("calibration_confidence") or 0)
        if matrix is not None and confidence >= 0.25:
            observations.append((int(frame["index"]) / source_fps, matrix, confidence))
    if not observations:
        return [
            {"index": index, "homography": None, "calibration_confidence": 0.0}
            for index in range(target_frame_count)
        ]

    times = np.asarray([item[0] for item in observations], dtype=np.float64)
    matrices = np.stack([item[1].reshape(-1) for item in observations])
    confidences = np.asarray([item[2] for item in observations], dtype=np.float64)
    target_times = np.arange(target_frame_count, dtype=np.float64) / target_fps
    interpolated = np.column_stack(
        [np.interp(target_times, times, matrices[:, column]) for column in range(9)]
    )
    interpolated_confidence = np.interp(target_times, times, confidences)

    # A three-sample temporal median removes isolated PnLCalib spikes while
    # preserving genuine broadcast pans and zooms.
    smoothed = interpolated.copy()
    for index in range(target_frame_count):
        window = interpolated[max(0, index - 1) : min(target_frame_count, index + 2)]
        smoothed[index] = np.median(window, axis=0)

    output = []
    for index, (flat, confidence) in enumerate(zip(smoothed, interpolated_confidence)):
        matrix = _normalise_homography(flat)
        output.append(
            {
                "index": index,
                "homography": matrix.reshape(-1).tolist() if matrix is not None else None,
                "calibration_confidence": round(float(confidence), 3),
            }
        )
    return output


def _bbox_center(box: list[float]) -> np.ndarray:
    return np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], dtype=float)


def _bbox_bottom(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, box[3]


def _appearance(frame: np.ndarray, box: list[float]) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    y2 = min(y2, y1 + max(1, int((y2 - y1) * 0.65)))
    crop = frame[y1:y2, x1:x2]
    if crop.size < 64:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    descriptor = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256]).reshape(-1)
    norm = float(np.linalg.norm(descriptor))
    return descriptor / norm if norm else None


def _appearance_score(first: np.ndarray | None, second: np.ndarray | None) -> float:
    if first is None or second is None:
        return 0.5
    return max(0.0, min(1.0, float(np.dot(first, second))))


def _field_point(box: list[float], calibration: dict[str, Any]) -> np.ndarray | None:
    if float(calibration.get("calibration_confidence") or 0) < 0.5:
        return None
    matrix = _normalise_homography(calibration.get("homography"))
    if matrix is None:
        return None
    point = np.asarray(project_point(_bbox_bottom(box), matrix), dtype=float)
    if -2 <= point[0] <= FIELD_WIDTH_M + 2 and -2 <= point[1] <= FIELD_HEIGHT_M + 2:
        return point
    return None


@dataclass
class _Track:
    track_id: int
    last_frame: int
    bbox: list[float]
    confidence: float
    role: str
    field_point: np.ndarray | None
    appearance: np.ndarray | None
    image_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    field_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=float))
    hits: int = 1

    def predicted_center(self, frame_index: int) -> np.ndarray:
        return _bbox_center(self.bbox) + self.image_velocity * (frame_index - self.last_frame)

    def predicted_field(self, frame_index: int) -> np.ndarray | None:
        if self.field_point is None:
            return None
        return self.field_point + self.field_velocity * (frame_index - self.last_frame)


def _association_score(
    track: _Track,
    detection: dict[str, Any],
    frame_index: int,
    detector_fps: float,
) -> float | None:
    box = detection["bbox"]
    gap = frame_index - track.last_frame
    seconds = gap / detector_fps
    center = _bbox_center(box)
    predicted_center = track.predicted_center(frame_index)
    height = max(1.0, box[3] - box[1], track.bbox[3] - track.bbox[1])
    # Camera pan and a noisy previous update can make a constant-velocity
    # prediction overshoot. The last observed position remains a valid fallback
    # for a short football-tracking gap.
    image_distance = min(
        float(np.linalg.norm(center - predicted_center)),
        float(np.linalg.norm(center - _bbox_center(track.bbox))),
    )
    if image_distance > max(90.0, height * (2.5 + gap)):
        return None

    field_score = 0.5
    predicted_field = track.predicted_field(frame_index)
    current_field = detection.get("_field")
    if predicted_field is not None and current_field is not None:
        last_field_distance = float(np.linalg.norm(current_field - track.field_point))
        # PnLCalib can move a projected foot point by a few metres during a
        # broadcast zoom. Keep that calibration tolerance separate from the
        # 14 m/s physical motion allowance; half-pitch jumps still fail closed.
        if last_field_distance > 5.0 + 14.0 * seconds:
            return None
        field_distance = min(
            last_field_distance,
            float(np.linalg.norm(current_field - predicted_field)),
        )
        field_score = math.exp(-field_distance / max(1.5, 5.0 * seconds + 0.5))

    if track.role != detection.get("role", "player"):
        return None
    motion_score = math.exp(-image_distance / max(20.0, 1.5 * height))
    overlap_score = bbox_iou(track.bbox, box)
    team_score = 0.5  # Team is resolved after masks; neutral cannot force a merge.
    appearance_score = _appearance_score(track.appearance, detection.get("_appearance"))
    return (
        0.45 * field_score
        + 0.25 * motion_score
        + 0.15 * overlap_score
        + 0.10 * team_score
        + 0.05 * appearance_score
    )


def _match(
    tracks: list[_Track],
    detections: list[dict[str, Any]],
    frame_index: int,
    detector_fps: float,
    minimum_score: float,
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    if not tracks or not detections:
        return [], set(range(len(tracks))), set(range(len(detections)))
    costs = np.full((len(tracks), len(detections)), 1e6, dtype=float)
    for track_index, track in enumerate(tracks):
        for detection_index, detection in enumerate(detections):
            score = _association_score(track, detection, frame_index, detector_fps)
            if score is not None and score >= minimum_score:
                costs[track_index, detection_index] = 1.0 - score
    rows, columns = linear_sum_assignment(costs)
    matches = [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if costs[row, column] < 1e5
    ]
    used_tracks = {row for row, _ in matches}
    used_detections = {column for _, column in matches}
    return (
        matches,
        set(range(len(tracks))) - used_tracks,
        set(range(len(detections))) - used_detections,
    )


def _update_track(
    track: _Track,
    detection: dict[str, Any],
    frame_index: int,
    detector_fps: float,
) -> None:
    gap = max(1, frame_index - track.last_frame)
    center_velocity = (_bbox_center(detection["bbox"]) - _bbox_center(track.bbox)) / gap
    maximum_image_velocity = max(
        20.0,
        0.75 * max(1.0, detection["bbox"][3] - detection["bbox"][1]),
    )
    center_speed = float(np.linalg.norm(center_velocity))
    if center_speed > maximum_image_velocity:
        center_velocity *= maximum_image_velocity / center_speed
    track.image_velocity = 0.35 * center_velocity + 0.65 * track.image_velocity
    current_field = detection.get("_field")
    if current_field is not None and track.field_point is not None:
        field_velocity = (current_field - track.field_point) / gap
        maximum_field_velocity = 14.0 / detector_fps
        field_speed = float(np.linalg.norm(field_velocity))
        if field_speed > maximum_field_velocity:
            field_velocity *= maximum_field_velocity / field_speed
        track.field_velocity = 0.35 * field_velocity + 0.65 * track.field_velocity
    if detection.get("_appearance") is not None:
        track.appearance = (
            detection["_appearance"]
            if track.appearance is None
            else 0.2 * detection["_appearance"] + 0.8 * track.appearance
        )
        norm = float(np.linalg.norm(track.appearance))
        if norm:
            track.appearance /= norm
    track.last_frame = frame_index
    track.bbox = detection["bbox"]
    track.confidence = float(detection.get("confidence") or 0)
    track.field_point = current_field
    track.hits += 1


def _scene_cut(previous: np.ndarray | None, current: np.ndarray) -> bool:
    if previous is None:
        return False
    first = cv2.calcHist([previous], [0], None, [32], [0, 256])
    second = cv2.calcHist([current], [0], None, [32], [0, 256])
    cv2.normalize(first, first)
    cv2.normalize(second, second)
    return cv2.compareHist(first, second, cv2.HISTCMP_CORREL) < 0.30


def field_space_track(
    detection_state: dict[str, Any],
    calibrations: list[dict[str, Any]],
    video_path: Path,
    *,
    high_confidence: float = 0.65,
    low_confidence: float = 0.05,
    max_lost_seconds: float = 1.5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    detector_fps = float(detection_state.get("fps") or 10)
    appearance_fps = min(
        detector_fps,
        max(0.1, float(os.getenv("REID_APPEARANCE_FPS", "5"))),
    )
    max_lost_frames = max(1, round(max_lost_seconds * detector_fps))
    capture = cv2.VideoCapture(str(video_path))
    active: dict[int, _Track] = {}
    output_by_track: dict[int, list[dict[str, Any]]] = {}
    next_id = 1
    previous_gray: np.ndarray | None = None
    cut_frames: list[int] = []
    low_confidence_recoveries = 0

    output_frames: list[dict[str, Any]] = []
    for frame_index, source_frame in enumerate(detection_state.get("frames", [])):
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90))
        if _scene_cut(previous_gray, gray):
            active.clear()
            cut_frames.append(frame_index)
        previous_gray = gray
        active = {
            track_id: track
            for track_id, track in active.items()
            if frame_index - track.last_frame <= max_lost_frames
        }
        calibration = calibrations[min(frame_index, len(calibrations) - 1)]
        prepared = []
        calculate_appearance = (
            frame_index % max(1, round(detector_fps / appearance_fps)) == 0
        )
        for source in source_frame.get("tracks", []):
            confidence = float(source.get("confidence") or 0)
            if confidence < low_confidence:
                continue
            box = [float(value) for value in source["bbox"]]
            prepared.append(
                {
                    **source,
                    "bbox": box,
                    "role": str(source.get("role") or "player"),
                    "_field": _field_point(box, calibration),
                    "_appearance": _appearance(frame, box) if calculate_appearance else None,
                }
            )
        high = [item for item in prepared if float(item["confidence"]) >= high_confidence]
        low = [item for item in prepared if float(item["confidence"]) < high_confidence]
        live = list(active.values())
        high_matches, unmatched_tracks, unmatched_high = _match(
            live, high, frame_index, detector_fps, minimum_score=0.34
        )
        assigned: list[tuple[_Track, dict[str, Any]]] = []
        for track_index, detection_index in high_matches:
            track = live[track_index]
            detection = high[detection_index]
            _update_track(track, detection, frame_index, detector_fps)
            assigned.append((track, detection))

        remaining_tracks = [live[index] for index in sorted(unmatched_tracks)]
        low_matches, _, _ = _match(
            remaining_tracks, low, frame_index, detector_fps, minimum_score=0.25
        )
        for track_index, detection_index in low_matches:
            track = remaining_tracks[track_index]
            detection = low[detection_index]
            _update_track(track, detection, frame_index, detector_fps)
            assigned.append((track, detection))
            low_confidence_recoveries += 1

        for detection_index in sorted(unmatched_high):
            detection = high[detection_index]
            track = _Track(
                track_id=next_id,
                last_frame=frame_index,
                bbox=detection["bbox"],
                confidence=float(detection["confidence"]),
                role=detection["role"],
                field_point=detection.get("_field"),
                appearance=detection.get("_appearance"),
            )
            active[next_id] = track
            next_id += 1
            assigned.append((track, detection))

        frame_tracks = []
        for track, detection in assigned:
            public_detection = {
                key: value
                for key, value in detection.items()
                if not key.startswith("_") and key != "track_id"
            }
            public_detection["track_id"] = track.track_id
            if detection.get("_field") is not None:
                public_detection["pitch"] = {
                    "x_bottom_middle": round(float(detection["_field"][0] - 52.5), 3),
                    "y_bottom_middle": round(float(detection["_field"][1] - 34.0), 3),
                }
            frame_tracks.append(public_detection)
            output_by_track.setdefault(track.track_id, []).append(
                {**public_detection, "frame": frame_index}
            )
        output_frames.append(
            {
                "index": frame_index,
                "homography": calibration.get("homography"),
                "calibration_confidence": calibration.get("calibration_confidence", 0),
                "tracks": frame_tracks,
            }
        )
    capture.release()

    # Noise detections never become public tracks. Require about 0.7 seconds;
    # the later on-pitch filter performs the final field gate.
    minimum_hits = max(3, round(detector_fps * 0.7))
    retained_ids = {
        track_id for track_id, detections in output_by_track.items() if len(detections) >= minimum_hits
    }
    for frame in output_frames:
        frame["tracks"] = [
            item for item in frame["tracks"] if int(item["track_id"]) in retained_ids
        ]
    metrics = {
        "raw_detections": sum(len(frame.get("tracks", [])) for frame in detection_state["frames"]),
        "created_tracks": next_id - 1,
        "retained_tracks": len(retained_ids),
        "low_confidence_recoveries": low_confidence_recoveries,
        "scene_cuts": cut_frames,
        "appearance_fps": appearance_fps,
    }
    return (
        {
            "fps": detector_fps,
            "width": detection_state.get("width"),
            "height": detection_state.get("height"),
            "frames": output_frames,
        },
        metrics,
    )


def resample_tracks(
    state: dict[str, Any],
    calibrations: list[dict[str, Any]],
    target_frame_count: int,
    target_fps: float,
    *,
    maximum_gap_seconds: float = 1.2,
) -> dict[str, Any]:
    """Linearly interpolate boxes to the 15 FPS SAM/output timeline."""
    source_fps = float(state.get("fps") or target_fps)
    by_track: dict[int, list[dict[str, Any]]] = {}
    for frame in state.get("frames", []):
        for detection in frame.get("tracks", []):
            by_track.setdefault(int(detection["track_id"]), []).append(
                {**detection, "_time": int(frame["index"]) / source_fps}
            )
    output_frames = []
    for target_index in range(target_frame_count):
        target_time = target_index / target_fps
        calibration = calibrations[target_index]
        frame_tracks = []
        for track_id, detections in by_track.items():
            times = [float(item["_time"]) for item in detections]
            position = int(np.searchsorted(times, target_time))
            if position == 0:
                before = after = detections[0]
            elif position >= len(detections):
                before = after = detections[-1]
            else:
                before, after = detections[position - 1], detections[position]
            if target_time < times[0] - 1e-6 or target_time > times[-1] + 1e-6:
                continue
            if float(after["_time"]) - float(before["_time"]) > maximum_gap_seconds:
                continue
            span = float(after["_time"]) - float(before["_time"])
            fraction = 0.0 if span <= 1e-9 else (target_time - float(before["_time"])) / span
            bbox = [
                float(first) + fraction * (float(second) - float(first))
                for first, second in zip(before["bbox"], after["bbox"])
            ]
            confidence = float(before.get("confidence") or 0) + fraction * (
                float(after.get("confidence") or 0) - float(before.get("confidence") or 0)
            )
            item = {
                **{key: value for key, value in before.items() if not key.startswith("_")},
                "track_id": track_id,
                "bbox": bbox,
                "confidence": confidence,
            }
            point = _field_point(bbox, calibration)
            item["pitch"] = (
                {
                    "x_bottom_middle": round(float(point[0] - 52.5), 3),
                    "y_bottom_middle": round(float(point[1] - 34.0), 3),
                }
                if point is not None
                else None
            )
            frame_tracks.append(item)
        output_frames.append(
            {
                "index": target_index,
                "homography": calibration.get("homography"),
                "calibration_confidence": calibration.get("calibration_confidence", 0),
                "tracks": frame_tracks,
            }
        )
    return {
        "fps": target_fps,
        "width": state.get("width"),
        "height": state.get("height"),
        "frames": output_frames,
    }
