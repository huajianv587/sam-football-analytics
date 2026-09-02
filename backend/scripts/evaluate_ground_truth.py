#!/usr/bin/env python3
"""Evaluate a 30-frame human validation set without substituting proxy metrics."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from worker.game_state import bbox_iou
from worker.rle import decode_mask


def load(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return json.load(handle)


def evaluate(results: Path, annotations: Path) -> dict[str, Any]:
    tracks = load(results / "tracks.json")
    ground_truth = load(annotations)
    predictions_by_frame: dict[int, list[dict[str, Any]]] = {}
    trajectories: dict[tuple[int, int], dict[str, Any]] = {}
    for track in tracks:
        track_id = int(track["object_id"])
        for detection in track.get("detections") or track["trajectory"]:
            frame = int(detection["frame"])
            predictions_by_frame.setdefault(frame, []).append(
                {"track_id": track_id, "bbox": detection["bbox"]}
            )
        for point in track["trajectory"]:
            trajectories[(track_id, int(point["frame"]))] = point

    mask_manifests = {
        int(path.name.removesuffix(".json.gz")): load(path)
        for path in (results / "masks").glob("[0-9]*.json.gz")
    }
    masks = {
        (track_id, int(frame["index"])): frame["rle"]
        for track_id, manifest in mask_manifests.items()
        for frame in manifest["frames"]
    }

    true_positive = false_positive = false_negative = 0
    identity_pairs: Counter[tuple[int, str]] = Counter()
    prediction_count = ground_truth_count = 0
    mask_ious: list[float] = []
    projection_errors: list[float] = []
    for annotated_frame in ground_truth["frames"]:
        frame_index = int(annotated_frame["frame"])
        predicted = predictions_by_frame.get(frame_index, [])
        expected = annotated_frame.get("objects", [])
        prediction_count += len(predicted)
        ground_truth_count += len(expected)
        if predicted and expected:
            costs = np.ones((len(predicted), len(expected)), dtype=float)
            for row, prediction in enumerate(predicted):
                for column, truth in enumerate(expected):
                    costs[row, column] = 1 - bbox_iou(prediction["bbox"], truth["bbox"])
            rows, columns = linear_sum_assignment(costs)
            matches = [
                (int(row), int(column))
                for row, column in zip(rows, columns)
                if costs[row, column] <= 0.5
            ]
        else:
            matches = []
        true_positive += len(matches)
        false_positive += len(predicted) - len(matches)
        false_negative += len(expected) - len(matches)
        for prediction_index, truth_index in matches:
            prediction = predicted[prediction_index]
            truth = expected[truth_index]
            track_id = int(prediction["track_id"])
            identity_pairs[(track_id, str(truth["gt_id"]))] += 1
            predicted_rle = masks.get((track_id, frame_index))
            truth_rle = truth.get("mask_rle")
            if predicted_rle and truth_rle:
                first = decode_mask(predicted_rle).astype(bool)
                second = decode_mask(truth_rle).astype(bool)
                intersection = int(np.logical_and(first, second).sum())
                union = int(np.logical_or(first, second).sum())
                mask_ious.append(intersection / union if union else 1.0)
            truth_pitch = truth.get("pitch")
            point = trajectories.get((track_id, frame_index))
            predicted_pitch = (point or {}).get("smoothed_pitch") or (point or {}).get("pitch")
            if truth_pitch is not None and predicted_pitch is not None:
                projection_errors.append(
                    float(np.linalg.norm(np.subtract(predicted_pitch, truth_pitch)))
                )

    predicted_ids = sorted({pair[0] for pair in identity_pairs})
    truth_ids = sorted({pair[1] for pair in identity_pairs})
    if predicted_ids and truth_ids:
        identity_costs = np.zeros((len(predicted_ids), len(truth_ids)), dtype=float)
        for row, track_id in enumerate(predicted_ids):
            for column, truth_id in enumerate(truth_ids):
                identity_costs[row, column] = -identity_pairs[(track_id, truth_id)]
        rows, columns = linear_sum_assignment(identity_costs)
        id_true_positive = int(-identity_costs[rows, columns].sum())
    else:
        id_true_positive = 0
    id_precision = id_true_positive / prediction_count if prediction_count else 0
    id_recall = id_true_positive / ground_truth_count if ground_truth_count else 0
    idf1 = (
        2 * id_precision * id_recall / (id_precision + id_recall)
        if id_precision + id_recall else 0
    )
    return {
        "annotated_frames": len(ground_truth["frames"]),
        "detection": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": round(true_positive / max(1, true_positive + false_positive), 4),
            "recall": round(true_positive / max(1, true_positive + false_negative), 4),
        },
        "identity": {
            "idf1": round(idf1, 4),
            "id_true_positive": id_true_positive,
            "prediction_observations": prediction_count,
            "ground_truth_observations": ground_truth_count,
        },
        "segmentation": {
            "annotated_masks": len(mask_ious),
            "mean_iou": round(float(np.mean(mask_ious)), 4) if mask_ious else None,
        },
        "calibration": {
            "control_points": len(projection_errors),
            "median_error_m": round(float(np.median(projection_errors)), 3)
            if projection_errors else None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_directory", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = evaluate(args.result_directory, args.annotations)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
