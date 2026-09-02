from collections import Counter
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

MAX_PLAYER_SPEED_KMH = 50.0
MAX_PLAYER_ACCELERATION_MPS2 = 12.0


def compute_homography(
    pairs: Sequence[dict[str, Sequence[float]]], width: int, height: int
) -> np.ndarray:
    source = np.asarray([[pair["video"][0] * width, pair["video"][1] * height] for pair in pairs], dtype=np.float32)
    target = np.asarray([pair["pitch"] for pair in pairs], dtype=np.float32)
    matrix, _ = cv2.findHomography(source, target, method=0)
    if matrix is None:
        raise ValueError("calibration points do not define a valid homography")
    return matrix


def project_point(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    source = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
    target = cv2.perspectiveTransform(source, homography)[0, 0]
    return round(float(target[0]), 3), round(float(target[1]), 3)


def smooth_metric_positions(
    points: Sequence[tuple[float, float] | None], fps: float
) -> list[tuple[float, float] | None]:
    """Robust field-space state estimate used by both speed and distance.

    A five-frame median suppresses foot-point noise, then a constant-velocity
    alpha-beta update rejects physically impossible camera/calibration jumps.
    Missing calibration remains missing instead of being silently predicted.
    """
    positions = np.full((len(points), 2), np.nan, dtype=np.float64)
    for index, point in enumerate(points):
        if point is not None:
            positions[index] = point
    median = positions.copy()
    if len(points) >= 10:
        median[:] = np.nan
        for index, point in enumerate(points):
            if point is None:
                continue
            window = positions[max(0, index - 2) : min(len(points), index + 3)]
            valid = window[~np.isnan(window).any(axis=1)]
            if len(valid):
                median[index] = np.median(valid, axis=0)

    output = np.full_like(median, np.nan)
    estimate: np.ndarray | None = None
    velocity = np.zeros(2, dtype=np.float64)
    dt = 1.0 / fps
    missing = 0
    for index, measurement in enumerate(median):
        if np.isnan(measurement).any():
            missing += 1
            if missing > max(2, round(fps * 0.5)):
                estimate = None
                velocity[:] = 0
            continue
        if estimate is None:
            estimate = measurement.copy()
            output[index] = estimate
            missing = 0
            continue
        if missing:
            estimate = measurement.copy()
            velocity[:] = 0
            output[index] = estimate
            missing = 0
            continue
        predicted = estimate + velocity * dt * (missing + 1)
        residual = measurement - predicted
        implied_speed = float(np.linalg.norm(measurement - estimate)) * fps / (missing + 1)
        if implied_speed * 3.6 > MAX_PLAYER_SPEED_KMH:
            # Begin a new physically plausible segment without publishing the
            # discontinuity itself as a position or speed observation.
            estimate = measurement.copy()
            velocity[:] = 0
            missing += 1
            continue
        if not np.any(velocity):
            velocity = (measurement - estimate) / dt
            estimate = measurement.copy()
        else:
            correction = 0.45 * residual
            estimate = predicted + correction
            velocity += 0.10 * residual / max(dt * (missing + 1), 1e-6)
        speed = float(np.linalg.norm(velocity))
        if speed * 3.6 > MAX_PLAYER_SPEED_KMH:
            velocity *= (MAX_PLAYER_SPEED_KMH / 3.6) / speed
        output[index] = estimate
        missing = 0
    return [
        None if np.isnan(value).any() else (round(float(value[0]), 3), round(float(value[1]), 3))
        for value in output
    ]


def speed_series(points: Sequence[tuple[float, float] | None], fps: float) -> list[float | None]:
    filtered_points = smooth_metric_positions(points, fps)
    positions = np.full((len(filtered_points), 2), np.nan, dtype=np.float64)
    for index, point in enumerate(filtered_points):
        if point is not None:
            positions[index] = point

    lag = 5 if len(points) >= 10 else 1

    raw = np.full(len(points), np.nan, dtype=np.float64)
    for index in range(lag, len(points)):
        if any(point is None for point in filtered_points[index - lag : index + 1]):
            continue
        speed = np.linalg.norm(positions[index] - positions[index - lag]) * fps / lag * 3.6
        if speed <= MAX_PLAYER_SPEED_KMH:
            raw[index] = speed

    median = np.full_like(raw, np.nan)
    for index in range(len(raw)):
        window = raw[max(0, index - 2) : min(len(raw), index + 3)]
        valid = window[~np.isnan(window)]
        if not np.isnan(raw[index]) and len(valid):
            median[index] = np.median(valid)

    smoothed = np.full_like(median, np.nan)
    alpha = 0.35
    previous: float | None = None
    for index, value in enumerate(median):
        if np.isnan(value):
            previous = None
            continue
        candidate = value if previous is None else alpha * value + (1 - alpha) * previous
        if previous is not None:
            acceleration = abs(candidate - previous) / 3.6 * fps
            if acceleration > MAX_PLAYER_ACCELERATION_MPS2:
                continue
        smoothed[index] = candidate
        previous = float(smoothed[index])
    return [None if np.isnan(value) else round(float(value), 2) for value in smoothed]


def traveled_distance(points: Sequence[tuple[float, float] | None], fps: float | None = None) -> float:
    if fps is not None:
        filtered = smooth_metric_positions(points, fps)
        total = 0.0
        for previous, current in zip(filtered, filtered[1:]):
            if previous is None or current is None:
                continue
            distance = float(np.linalg.norm(np.subtract(current, previous)))
            if distance * fps * 3.6 <= MAX_PLAYER_SPEED_KMH:
                total += distance
        if not total:
            for previous, current in zip(points, points[1:]):
                if previous is None or current is None:
                    continue
                distance = float(np.linalg.norm(np.subtract(current, previous)))
                if distance * fps * 3.6 <= MAX_PLAYER_SPEED_KMH:
                    total += distance
        return round(total, 2)
    total = 0.0
    for previous, current in zip(points, points[1:]):
        if previous is not None and current is not None:
            distance = float(np.linalg.norm(np.subtract(current, previous)))
            if fps is None or distance * fps * 3.6 <= MAX_PLAYER_SPEED_KMH:
                total += distance
    return round(total, 2)


def dominant_jersey_bgr(frame: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int] | None:
    x1, y1, x2, y2 = bbox
    torso_y2 = y1 + max(1, int((y2 - y1) * 0.55))
    crop = frame[y1:torso_y2, x1:x2]
    crop_mask = mask[y1:torso_y2, x1:x2]
    if crop.size == 0 or crop_mask.sum() < 16:
        return None
    pixels = crop[crop_mask]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    keep = ~((hsv[:, 0] >= 35) & (hsv[:, 0] <= 90) & (hsv[:, 1] > 45))
    pixels = pixels[keep]
    if not len(pixels):
        return None
    return tuple(int(value) for value in np.median(pixels, axis=0))


def classify_team(
    bgr: tuple[int, int, int] | None,
    references_rgb: dict[str, Sequence[int]],
    team_a: str,
    team_b: str,
    max_lab_distance: float = 45.0,
) -> str:
    if bgr is None:
        return "unknown"
    sample = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    labels = {"team_a": team_a, "team_b": team_b, "referee": "Referee"}
    distances: dict[str, float] = {}
    for key, rgb in references_rgb.items():
        ref_bgr = tuple(reversed(rgb))
        ref = cv2.cvtColor(np.uint8([[ref_bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(float)
        distances[key] = float(np.linalg.norm(sample - ref))
    closest = min(distances, key=distances.get)
    return labels[closest] if distances[closest] <= max_lab_distance else "unknown"


def ocr_vote(candidates: Sequence[tuple[str, float]]) -> tuple[int | None, float]:
    scores: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    for text, confidence in candidates:
        digits = "".join(character for character in text if character.isdigit())
        if digits and 0 < int(digits) <= 99:
            scores[digits] += float(confidence)
            counts[digits] += 1
    if not scores:
        return None, 0.0
    number, score = scores.most_common(1)[0]
    return int(number), round(min(1.0, score / counts[number]), 3)


def bbox_iou(first: Sequence[int], second: Sequence[int]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def bbox_touches_edge(
    bbox: Sequence[int], width: int, height: int, margin_ratio: float = 0.02
) -> bool:
    x1, y1, x2, y2 = bbox
    margin_x = max(2, int(width * margin_ratio))
    margin_y = max(2, int(height * margin_ratio))
    return x1 <= margin_x or y1 <= margin_y or x2 >= width - margin_x or y2 >= height - margin_y


def occlusion_metrics(
    samples: dict[int, list[dict[str, Any]]],
    frame_size: tuple[int, int] | None = None,
) -> dict[int, dict[str, Any]]:
    events: dict[int, list[int]] = {object_id: [] for object_id in samples}
    frame_maps = {object_id: {sample["frame"]: sample for sample in track} for object_id, track in samples.items()}
    all_frames = sorted({frame for frames in frame_maps.values() for frame in frames})
    for frame in all_frames:
        visible = [(object_id, frames[frame]) for object_id, frames in frame_maps.items() if frame in frames]
        for index, (first_id, first) in enumerate(visible):
            for second_id, second in visible[index + 1 :]:
                if bbox_iou(first["bbox"], second["bbox"]) >= 0.15:
                    events[first_id].append(frame)
                    events[second_id].append(frame)

    output: dict[int, dict[str, Any]] = {}
    final_frame = max(all_frames, default=0)
    for object_id, track in samples.items():
        areas = [sample["area"] for sample in track]
        baseline = float(np.median(areas)) if areas else 1.0
        area_drop_frames = [sample["frame"] for sample in track if sample["area"] < baseline * 0.6]
        occluded = sorted(set(events[object_id]) | set(area_drop_frames))
        groups = sum(index == 0 or frame > occluded[index - 1] + 1 for index, frame in enumerate(occluded))
        recovery_frames: int | None = None
        recovery_ratio = 0.0
        if occluded:
            after = [sample for sample in track if sample["frame"] > occluded[-1]]
            if after:
                recovery_ratio = max(sample["area"] for sample in after[:5]) / max(baseline, 1)
                recovered = next((sample for sample in after if sample["area"] >= baseline * 0.8), None)
                if recovered:
                    recovery_frames = recovered["frame"] - occluded[-1]
        elif areas:
            recovery_ratio = float(np.median(areas[-3:])) / max(baseline, 1)
        centroids = [sample.get("centroid", sample["foot"]) for sample in track]
        centroid_jumps = [
            float(np.linalg.norm(np.subtract(current, previous)))
            for previous, current in zip(centroids, centroids[1:])
        ]
        exited_frame = None
        if track and frame_size and track[-1]["frame"] < final_frame:
            width, height = frame_size
            if bbox_touches_edge(track[-1]["bbox"], width, height):
                exited_frame = track[-1]["frame"]
        id_retained = bool(track) and (
            exited_frame is not None or not occluded or recovery_frames is not None
        )
        output[object_id] = {
            "occlusion_frames": occluded,
            "occlusion_count": groups,
            "area_recovery_ratio": round(recovery_ratio, 3),
            "recovery_frames": recovery_frames,
            "exited_frame": exited_frame,
            "max_centroid_jump_px": round(max(centroid_jumps, default=0.0), 2),
            "id_retained": id_retained,
        }
    return output
