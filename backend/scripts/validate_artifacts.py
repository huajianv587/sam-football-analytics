#!/usr/bin/env python3
"""Validate a PitchVision result bundle and report reproducible quality metrics."""

import argparse
import gzip
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from worker.game_state import bbox_iou
from worker.rle import decode_mask, mask_bbox


def load_json(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def video_report(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    while capture.read()[0]:
        decoded += 1
    capture.release()
    return {
        "declared_frames": declared,
        "decoded_frames": decoded,
        "fps": round(fps, 3),
        "width": width,
        "height": height,
        "complete": declared > 0 and declared == decoded,
    }


def percentile(values: list[float], value: int) -> float | None:
    return round(float(np.percentile(values, value)), 4) if values else None


def continuity_report(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    long_gaps: list[dict[str, int]] = []
    impossible_jumps: list[dict[str, float | int]] = []
    for track in tracks:
        track_id = int(track["object_id"])
        detections = sorted(track.get("detections", []), key=lambda item: int(item["frame"]))
        for first, second in zip(detections, detections[1:], strict=False):
            first_frame = int(first["frame"])
            second_frame = int(second["frame"])
            gap = second_frame - first_frame
            if gap > 45:
                long_gaps.append({"track_id": track_id, "frames": gap})
            ax1, ay1, ax2, ay2 = (float(value) for value in first["bbox"])
            bx1, by1, bx2, by2 = (float(value) for value in second["bbox"])
            displacement = math.dist(
                ((ax1 + ax2) / 2, (ay1 + ay2) / 2),
                ((bx1 + bx2) / 2, (by1 + by2) / 2),
            )
            height = max(ay2 - ay1, by2 - by1)
            if gap <= 6 and displacement > max(120.0, 4.0 * height):
                impossible_jumps.append(
                    {
                        "track_id": track_id,
                        "from_frame": first_frame,
                        "to_frame": second_frame,
                        "pixels": round(displacement, 2),
                    }
                )
    return {
        "long_gap_events_over_3s": len(long_gaps),
        "impossible_short_step_events": len(impossible_jumps),
        "examples": (long_gaps + impossible_jumps)[:10],
    }


def validate(directory: Path) -> dict[str, Any]:
    tracks = load_json(directory / "tracks.json")
    calibration = load_json(directory / "calibration.json.gz")
    mask_paths = sorted(
        path for path in (directory / "masks").glob("*.json.gz")
        if not path.name.startswith("._")
    )
    track_ids = {int(track["object_id"]) for track in tracks}
    mask_ids = {int(path.name.split(".", 1)[0]) for path in mask_paths}

    ious: list[float] = []
    centroids_inside = 0
    comparable_frames = 0
    total_mask_frames = 0
    empty_masks = 0
    per_track_frames: dict[int, int] = {}
    severe_by_track: dict[int, int] = {}

    tracks_by_id = {int(track["object_id"]): track for track in tracks}
    for path in mask_paths:
        payload = load_json(path)
        track_id = int(payload["track_id"])
        track = tracks_by_id[track_id]
        detections = {
            int(sample["frame"]): sample["bbox"]
            for sample in track.get("detections") or track["trajectory"]
        }
        per_track_frames[track_id] = len(payload["frames"])
        for frame in payload["frames"]:
            total_mask_frames += 1
            mask = decode_mask(frame["rle"])
            box = mask_bbox(mask)
            if box is None:
                empty_masks += 1
                continue
            detection = detections.get(int(frame["index"]))
            if detection is None:
                continue
            comparable_frames += 1
            overlap = bbox_iou(box, detection)
            ious.append(overlap)
            if overlap < 0.1:
                severe_by_track[track_id] = severe_by_track.get(track_id, 0) + 1
            ys, xs = np.nonzero(mask)
            cx = float(xs.mean())
            cy = float(ys.mean())
            x1, y1, x2, y2 = detection
            centroids_inside += int(x1 <= cx <= x2 and y1 <= cy <= y2)

    valid_calibrations = sum(bool(frame.get("valid")) for frame in calibration["frames"])
    roles: dict[str, int] = {}
    teams: dict[str, int] = {}
    identified = 0
    metric_tracks = 0
    visible_per_frame: dict[int, int] = {}
    for track in tracks:
        roles[track["role"]] = roles.get(track["role"], 0) + 1
        teams[track["team"]] = teams.get(track["team"], 0) + 1
        identified += int(bool(track.get("player_name")))
        metric_tracks += int(any(point.get("speed_kmh") is not None for point in track["trajectory"]))
        for point in track.get("detections") or track["trajectory"]:
            frame = int(point["frame"])
            visible_per_frame[frame] = visible_per_frame.get(frame, 0) + 1

    return {
        "artifact_consistency": {
            "track_count": len(tracks),
            "mask_file_count": len(mask_paths),
            "track_ids_without_mask": sorted(track_ids - mask_ids),
            "mask_ids_without_track": sorted(mask_ids - track_ids),
            "total_mask_frames": total_mask_frames,
            "empty_masks": empty_masks,
            "mask_frames_min": min(per_track_frames.values(), default=0),
            "mask_frames_max": max(per_track_frames.values(), default=0),
        },
        "mask_detection_consistency": {
            "comparable_frames": comparable_frames,
            "bbox_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
            "bbox_iou_median": percentile(ious, 50),
            "bbox_iou_p10": percentile(ious, 10),
            "bbox_iou_below_0_1": sum(value < 0.1 for value in ious),
            "worst_tracks": [
                {"track_id": track_id, "frames_below_0_1": count}
                for track_id, count in sorted(
                    severe_by_track.items(), key=lambda item: (-item[1], item[0])
                )[:10]
            ],
            "centroid_inside_bbox_rate": round(centroids_inside / comparable_frames, 4)
            if comparable_frames else None,
        },
        "calibration": {
            "frame_count": len(calibration["frames"]),
            "valid_frames": valid_calibrations,
            "valid_rate": round(valid_calibrations / len(calibration["frames"]), 4)
            if calibration["frames"] else 0.0,
        },
        "tracks": {
            "roles": roles,
            "teams": teams,
            "identified": identified,
            "metric_speed_available": metric_tracks,
            "visible_per_frame": {
                "min": min(visible_per_frame.values(), default=0),
                "median": percentile(list(visible_per_frame.values()), 50),
                "max": max(visible_per_frame.values(), default=0),
            },
        },
        "track_continuity": continuity_report(tracks),
        "videos": {
            name: video_report(directory / name)
            for name in ("normalized.mp4", "foreground.mp4")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.result_directory)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
