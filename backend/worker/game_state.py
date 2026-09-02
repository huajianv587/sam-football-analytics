import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VALID_ROLES = {"player", "goalkeeper", "referee"}


def normalize_role(value: Any) -> str | None:
    role = str(value).lower() if value is not None else ""
    # The pinned SoccerNet step-one pipeline deliberately omits the large
    # vision-language role model. Every on-pitch person detector result starts
    # as a player; referee and goalkeeper roles are refined from appearance and
    # roster evidence after SAM segmentation.
    if not role:
        return "player"
    return role if role in VALID_ROLES else None


def detections_by_track(state: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    tracks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in state["frames"]:
        for detection in frame.get("tracks", []):
            role = normalize_role(detection.get("role"))
            if role:
                tracks[int(detection["track_id"])].append(
                    {**detection, "role": role, "frame": int(frame["index"])}
                )
    for detections in tracks.values():
        detections.sort(key=lambda item: item["frame"])
    return dict(tracks)


def on_pitch_tracks(
    tracks: dict[int, list[dict[str, Any]]],
    minimum_ratio: float = 0.5,
    minimum_detections: int = 30,
    pitch_margin_m: float = 1.0,
) -> dict[int, list[dict[str, Any]]]:
    """Remove people whose calibrated positions are mostly outside the field.

    Calibration is allowed to fail. Tracks without any metric observations are
    retained so a failed camera estimate never hides otherwise valid players.
    """
    retained: dict[int, list[dict[str, Any]]] = {}
    for track_id, detections in tracks.items():
        if len(detections) < minimum_detections:
            continue
        positions = [item.get("pitch") for item in detections if isinstance(item.get("pitch"), dict)]
        if not positions:
            retained[track_id] = detections
            continue
        inside = sum(
            -52.5 - pitch_margin_m
            <= float(position.get("x_bottom_middle", float("inf")))
            <= 52.5 + pitch_margin_m
            and -34.0 - pitch_margin_m
            <= float(position.get("y_bottom_middle", float("inf")))
            <= 34.0 + pitch_margin_m
            for position in positions
        )
        if inside / len(positions) >= minimum_ratio:
            retained[track_id] = detections
    return retained


def track_continuity_metrics(
    tracks: dict[int, list[dict[str, Any]]],
    *,
    long_gap_frames: int = 45,
) -> dict[str, Any]:
    long_gaps = 0
    impossible_steps = 0
    visible: dict[int, int] = defaultdict(int)
    for detections in tracks.values():
        ordered = sorted(detections, key=lambda item: int(item["frame"]))
        for detection in ordered:
            visible[int(detection["frame"])] += 1
        for first, second in zip(ordered, ordered[1:], strict=False):
            gap = int(second["frame"]) - int(first["frame"])
            if gap > long_gap_frames:
                long_gaps += 1
            ax1, ay1, ax2, ay2 = (float(value) for value in first["bbox"])
            bx1, by1, bx2, by2 = (float(value) for value in second["bbox"])
            displacement = math.dist(
                ((ax1 + ax2) / 2, (ay1 + ay2) / 2),
                ((bx1 + bx2) / 2, (by1 + by2) / 2),
            )
            height = max(ay2 - ay1, by2 - by1)
            impossible_steps += int(
                gap <= 6 and displacement > max(120.0, 4.0 * height)
            )
    counts = list(visible.values())
    return {
        "long_gap_events_over_3s": long_gaps,
        "impossible_short_step_events": impossible_steps,
        "visible_people_min": min(counts, default=0),
        "visible_people_median": round(float(np.median(counts)), 2) if counts else 0,
        "visible_people_max": max(counts, default=0),
    }


def bbox_iou(first: Iterable[float], second: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def select_prompt_detection(
    detections: list[dict[str, Any]],
    frames: list[Path] | None = None,
    preferred_frame: int | None = None,
    temporal_radius: int | None = None,
) -> dict[str, Any]:
    if not detections:
        raise ValueError("track has no detections")
    candidates = detections
    if preferred_frame is not None and temporal_radius is not None:
        centered = [
            item
            for item in detections
            if abs(int(item["frame"]) - preferred_frame) <= temporal_radius
        ]
        if centered:
            candidates = centered
    ranked: list[tuple[float, dict[str, Any]]] = []
    for detection in candidates:
        x1, y1, x2, y2 = detection["bbox"]
        confidence = float(detection.get("confidence") or 0)
        area = max(1.0, (x2 - x1) * (y2 - y1))
        clarity = 1.0
        if frames is not None:
            frame = cv2.imread(str(frames[int(detection["frame"])]))
            crop = frame[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)]
            if crop.size:
                clarity = min(
                    4.0,
                    1.0 + cv2.Laplacian(
                        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                    ).var()
                    / 300.0,
                )
        ranked.append((confidence * np.sqrt(area) * clarity, detection))
    return max(ranked, key=lambda item: item[0])[1]


def select_prompt_detections(
    detections: list[dict[str, Any]],
    frames: list[Path] | None = None,
    count: int = 3,
) -> list[dict[str, Any]]:
    """Choose temporally distinct, high-quality anchors for one track window."""
    if not detections:
        raise ValueError("track has no detections")
    if count <= 1 or len(detections) == 1:
        return [select_prompt_detection(detections, frames)]

    first_frame = int(detections[0]["frame"])
    last_frame = int(detections[-1]["frame"])
    span = last_frame - first_frame
    fractions = (0.15, 0.5, 0.85) if count == 3 else np.linspace(0.1, 0.9, count)
    radius = max(4, span // (count * 3))
    selected: dict[int, dict[str, Any]] = {}
    for fraction in fractions:
        preferred = round(first_frame + span * float(fraction))
        prompt = select_prompt_detection(
            detections,
            frames,
            preferred_frame=preferred,
            temporal_radius=radius,
        )
        selected[int(prompt["frame"])] = prompt
    return [selected[frame] for frame in sorted(selected)]


def track_windows(
    first_frame: int,
    last_frame: int,
    window_size: int = 180,
    overlap: int = 30,
) -> list[tuple[int, int]]:
    if first_frame > last_frame:
        return []
    windows: list[tuple[int, int]] = []
    start = first_frame
    while start <= last_frame:
        end = min(last_frame, start + window_size - 1)
        windows.append((start, end))
        if end == last_frame:
            break
        start = end - overlap + 1
    return windows


def manual_game_state(
    payload: dict[str, Any],
    frame_count: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    from .analytics import compute_homography

    homography = compute_homography(payload["calibration"], width, height).reshape(-1).tolist()
    frames = []
    for frame_index in range(frame_count):
        tracks = []
        for prompt in payload["prompts"]:
            x1, y1, x2, y2 = prompt["box"]
            tracks.append(
                {
                    "track_id": int(prompt["object_id"]),
                    "bbox": [x1 * width, y1 * height, x2 * width, y2 * height],
                    "confidence": 1.0,
                    "role": "player",
                }
            )
        frames.append(
            {
                "index": frame_index,
                "homography": homography,
                "calibration_confidence": 1.0,
                "tracks": tracks,
            }
        )
    return {"fps": 15, "width": width, "height": height, "frames": frames}


def load_game_state(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text())
    if not state.get("frames"):
        raise RuntimeError("game-state output contains no frames")
    if not detections_by_track(state):
        raise RuntimeError("game-state output contains no valid person tracks")
    return state


def resample_game_state(
    state: dict[str, Any], target_frame_count: int, target_fps: float
) -> dict[str, Any]:
    """Repeat low-rate GSR observations into the full SAM frame timeline."""
    source_frames = state["frames"]
    if not source_frames or target_frame_count <= 0:
        raise ValueError("game state needs source and target frames")
    frames = []
    for target_index in range(target_frame_count):
        source_index = min(
            len(source_frames) - 1,
            int(target_index * len(source_frames) / target_frame_count),
        )
        source = source_frames[source_index]
        frames.append(
            {
                **source,
                "index": target_index,
                "tracks": [{**track} for track in source.get("tracks", [])],
            }
        )
    return {**state, "fps": target_fps, "frames": frames}


def frame_calibrations(state: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(frame["index"]): {
            "homography": frame.get("homography"),
            "confidence": float(frame.get("calibration_confidence") or 0),
            "valid": frame.get("homography") is not None
            and float(frame.get("calibration_confidence") or 0) >= 0.5,
        }
        for frame in state["frames"]
    }
