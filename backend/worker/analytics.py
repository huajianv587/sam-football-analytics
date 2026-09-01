from collections import Counter
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np


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


def speed_series(points: Sequence[tuple[float, float] | None], fps: float) -> list[float]:
    raw = np.zeros(len(points), dtype=np.float64)
    for index in range(1, len(points)):
        if points[index] is None or points[index - 1] is None:
            raw[index] = raw[index - 1]
            continue
        raw[index] = np.linalg.norm(np.subtract(points[index], points[index - 1])) * fps * 3.6

    median = raw.copy()
    for index in range(len(raw)):
        window = raw[max(0, index - 2) : min(len(raw), index + 3)]
        median[index] = np.median(window)

    smoothed = np.zeros_like(median)
    alpha = 0.35
    for index, value in enumerate(median):
        smoothed[index] = value if index == 0 else alpha * value + (1 - alpha) * smoothed[index - 1]
    return [round(float(value), 2) for value in smoothed]


def traveled_distance(points: Sequence[tuple[float, float] | None]) -> float:
    total = 0.0
    for previous, current in zip(points, points[1:]):
        if previous is not None and current is not None:
            total += float(np.linalg.norm(np.subtract(current, previous)))
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
    return labels[min(distances, key=distances.get)]


def ocr_vote(candidates: Sequence[tuple[str, float]]) -> tuple[int | None, float]:
    scores: Counter[str] = Counter()
    for text, confidence in candidates:
        digits = "".join(character for character in text if character.isdigit())
        if digits and 0 < int(digits) <= 99:
            scores[digits] += float(confidence)
    if not scores:
        return None, 0.0
    number, score = scores.most_common(1)[0]
    return int(number), round(score, 3)


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
        baseline = float(np.median(areas[: min(5, len(areas))])) if areas else 1.0
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
