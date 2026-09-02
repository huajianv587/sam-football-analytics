import argparse
import json
import math
import pickle
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def first(row: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if name in row and clean(row[name]) is not None:
            return clean(row[name])
    return default


def homography_from_parameters(parameters: Any) -> tuple[list[float] | None, float]:
    parameters = clean(parameters)
    if isinstance(parameters, list):
        matrix = np.asarray(parameters, dtype=float)
        if matrix.size == 9:
            return matrix.reshape(-1).tolist(), 1.0
    if isinstance(parameters, dict):
        for key in ("homography", "H", "homography_matrix"):
            if key in parameters:
                matrix = np.asarray(parameters[key], dtype=float)
                if matrix.size == 9:
                    return matrix.reshape(-1).tolist(), float(parameters.get("confidence", 1))
    return None, 0.0


def jersey_value(value: Any) -> int | None:
    value = clean(value)
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if 0 < number <= 99 else None


def calibration_support(row: Any) -> float:
    keypoints = clean(row.get("keypoints")) if "keypoints" in row else None
    lines = clean(row.get("lines_det")) if "lines_det" in row else None
    support = sum(len(item) for item in (keypoints, lines) if isinstance(item, dict))
    return min(1.0, support / 12.0)


def export(pklz: Path, video: Path, output: Path) -> None:
    capture = cv2.VideoCapture(str(video))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 15)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    with zipfile.ZipFile(pklz) as archive:
        detection_name = next(
            name for name in archive.namelist() if name.endswith(".pkl") and not name.endswith("_image.pkl")
        )
        image_name = detection_name.removesuffix(".pkl") + "_image.pkl"
        with archive.open(detection_name) as handle:
            detections = pickle.load(handle)
        with archive.open(image_name) as handle:
            images = pickle.load(handle)

    image_rows = {}
    for index, row in images.iterrows():
        image_id = first(row, "id", default=index)
        frame_index = int(first(row, "frame", default=image_id))
        homography, _ = homography_from_parameters(first(row, "parameters"))
        # A matrix alone is not evidence that the camera estimate is usable.
        # Score it from the pitch keypoints/lines that were actually observed;
        # the RANSAC inlier ratio below then penalizes geometric disagreement.
        confidence = calibration_support(row)
        image_rows[image_id] = {
            "index": frame_index,
            "homography": homography,
            "calibration_confidence": confidence,
            "tracks": [],
        }

    for _, row in detections.iterrows():
        track_id = first(row, "track_id")
        image_id = first(row, "image_id")
        bbox = first(row, "bbox_ltwh")
        if track_id is None or image_id not in image_rows or bbox is None:
            continue
        x, y, w, h = [float(value) for value in bbox]
        role = first(row, "role", "role_detection")
        team = first(row, "team")
        jersey_number = first(row, "jersey_number", "jersey_number_detection")
        pitch = first(row, "bbox_pitch")
        image_rows[image_id]["tracks"].append(
            {
                "track_id": int(track_id),
                "bbox": [x, y, x + w, y + h],
                "confidence": float(first(row, "bbox_conf", "confidence", default=0)),
                "role": role,
                "role_confidence": float(first(row, "role_confidence", default=0)),
                "team": team,
                "jersey_number": jersey_value(jersey_number),
                "identity_confidence": float(first(row, "jn_confidence", "jersey_number_confidence", default=0)),
                "pitch": pitch,
            }
        )

    for frame in image_rows.values():
        source_points = []
        pitch_points = []
        for track in frame["tracks"]:
            pitch = track.get("pitch")
            if not isinstance(pitch, dict):
                continue
            x1, _, x2, y2 = track["bbox"]
            pitch_x = clean(pitch.get("x_bottom_middle"))
            pitch_y = clean(pitch.get("y_bottom_middle"))
            if pitch_x is None or pitch_y is None:
                continue
            source_points.append([(x1 + x2) / 2, y2])
            pitch_points.append([float(pitch_x), float(pitch_y)])
        if len(source_points) >= 4:
            matrix, inliers = cv2.findHomography(
                np.asarray(source_points, dtype=np.float32),
                np.asarray(pitch_points, dtype=np.float32),
                cv2.RANSAC,
                2.0,
            )
            if matrix is not None:
                centered_to_pitch = np.asarray(
                    [[1, 0, 52.5], [0, 1, 34.0], [0, 0, 1]], dtype=float
                )
                frame["homography"] = (centered_to_pitch @ matrix).reshape(-1).tolist()
                inlier_ratio = float(inliers.mean()) if inliers is not None else 1.0
                frame["calibration_confidence"] *= inlier_ratio

    frames_by_index = {row["index"]: row for row in image_rows.values()}
    frames = [
        frames_by_index.get(
            index,
            {"index": index, "homography": None, "calibration_confidence": 0, "tracks": []},
        )
        for index in range(frame_count)
    ]
    output.write_text(
        json.dumps(
            {"fps": fps, "width": width, "height": height, "frames": frames},
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pklz", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    export(arguments.pklz, arguments.video, arguments.output)
