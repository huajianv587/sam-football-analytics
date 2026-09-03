import gzip
import json

import numpy as np

from scripts.evaluate_ground_truth import evaluate
from worker.rle import encode_mask


def test_human_validation_metrics_use_real_matches_and_masks(tmp_path) -> None:
    results = tmp_path / "results"
    masks = results / "masks"
    masks.mkdir(parents=True)
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:12, 3:9] = True
    rle = encode_mask(mask)
    track = {
        "object_id": 7,
        "detections": [
            {"frame": 0, "bbox": [3, 2, 9, 12]},
            {"frame": 1, "bbox": [4, 2, 10, 12]},
        ],
        "trajectory": [
            {"frame": 0, "pitch": [10, 20], "smoothed_pitch": [10, 20]},
            {"frame": 1, "pitch": [11, 20], "smoothed_pitch": [11, 20]},
        ],
    }
    (results / "tracks.json").write_text(json.dumps([track]))
    with gzip.open(masks / "7.json.gz", "wt") as handle:
        json.dump(
            {
                "track_id": 7,
                "frames": [
                    {"index": 0, "rle": rle},
                    {"index": 1, "rle": rle},
                ],
            },
            handle,
        )
    annotations = tmp_path / "truth.json"
    annotations.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame": 0,
                        "objects": [
                            {
                                "gt_id": "p1",
                                "bbox": [3, 2, 9, 12],
                                "mask_rle": rle,
                                "pitch": [10, 20],
                            }
                        ],
                    },
                    {
                        "frame": 1,
                        "objects": [
                            {
                                "gt_id": "p1",
                                "bbox": [4, 2, 10, 12],
                                "mask_rle": rle,
                                "pitch": [11, 20],
                            }
                        ],
                    },
                ]
            }
        )
    )
    report = evaluate(results, annotations)
    assert report["detection"]["precision"] == 1
    assert report["detection"]["recall"] == 1
    assert report["identity"]["idf1"] == 1
    assert report["segmentation"]["mean_iou"] == 1
    assert report["calibration"]["median_error_m"] == 0


def test_live_session_metrics_read_indexed_tracks_and_sam_polygons(tmp_path) -> None:
    results = tmp_path / "live-session"
    results.mkdir()
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:12, 3:9] = True
    rle = encode_mask(mask)
    with gzip.open(results / "frames.json.gz", "wt") as handle:
        json.dump(
            [
                {
                    "frame_id": 0,
                    "width": 20,
                    "height": 20,
                    "tracks": [{"track_id": 7, "bbox": [3, 2, 9, 12]}],
                }
            ],
            handle,
        )
    with gzip.open(results / "sam-7-0-0.json.gz", "wt") as handle:
        json.dump(
            {
                "track_id": 7,
                "frames": [{"frame": 0, "mask": [[3, 2], [8, 2], [8, 11], [3, 11]]}],
            },
            handle,
        )
    annotations = tmp_path / "truth.json"
    annotations.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame": 0, "objects": [{"gt_id": "p1", "bbox": [3, 2, 9, 12], "mask_rle": rle}]}
                ]
            }
        )
    )
    report = evaluate(results, annotations)
    assert report["detection"]["precision"] == 1
    assert report["identity"]["idf1"] == 1
    assert report["segmentation"]["mean_iou"] == 1
