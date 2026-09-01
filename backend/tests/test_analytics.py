import numpy as np

from worker.analytics import (
    bbox_touches_edge,
    classify_team,
    compute_homography,
    occlusion_metrics,
    ocr_vote,
    project_point,
    speed_series,
    traveled_distance,
)


def test_bbox_edge_detection_uses_small_frame_margin() -> None:
    assert bbox_touches_edge([0, 20, 10, 40], 100, 60) is True
    assert bbox_touches_edge([92, 20, 100, 40], 100, 60) is True
    assert bbox_touches_edge([20, 20, 30, 40], 100, 60) is False


def test_homography_maps_video_corners_to_pitch() -> None:
    pairs = [
        {"video": [0, 0], "pitch": [0, 0]},
        {"video": [1, 0], "pitch": [105, 0]},
        {"video": [1, 1], "pitch": [105, 68]},
        {"video": [0, 1], "pitch": [0, 68]},
    ]
    matrix = compute_homography(pairs, 1000, 500)
    assert project_point((500, 250), matrix) == (52.5, 34.0)


def test_speed_and_distance_use_metric_coordinates() -> None:
    points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    speeds = speed_series(points, fps=1)
    assert speeds[-1] == 3.6
    assert traveled_distance(points) == 4.0


def test_ocr_vote_uses_accumulated_confidence() -> None:
    number, confidence = ocr_vote([("10", 0.7), ("10", 0.6), ("18", 0.9), ("A", 1.0)])
    assert number == 10
    assert confidence == 1.3


def test_jersey_color_uses_nearest_lab_reference() -> None:
    references = {"team_a": [220, 30, 40], "team_b": [120, 200, 235], "referee": [30, 32, 36]}
    assert classify_team((40, 30, 220), references, "Spain", "Argentina") == "Spain"


def test_occlusion_reports_recovery_and_centroid_jump() -> None:
    samples = {
        1: [
            {"frame": 0, "bbox": [0, 0, 10, 10], "area": 100, "centroid": [5, 5], "foot": [5, 10]},
            {"frame": 1, "bbox": [0, 0, 10, 10], "area": 40, "centroid": [7, 5], "foot": [7, 10]},
            {"frame": 2, "bbox": [0, 0, 10, 10], "area": 90, "centroid": [8, 5], "foot": [8, 10]},
        ]
    }
    metrics = occlusion_metrics(samples)[1]
    assert metrics["occlusion_frames"] == [1]
    assert metrics["recovery_frames"] == 1
    assert metrics["max_centroid_jump_px"] == 2.0
    assert metrics["id_retained"] is True


def test_track_leaving_at_frame_edge_is_retained() -> None:
    samples = {
        1: [
            {"frame": 0, "bbox": [20, 20, 30, 40], "area": 100, "centroid": [25, 30], "foot": [25, 40]},
            {"frame": 1, "bbox": [92, 20, 100, 40], "area": 40, "centroid": [96, 30], "foot": [96, 40]},
        ],
        2: [
            {"frame": 0, "bbox": [50, 20, 60, 40], "area": 100, "centroid": [55, 30], "foot": [55, 40]},
            {"frame": 1, "bbox": [52, 20, 62, 40], "area": 100, "centroid": [57, 30], "foot": [57, 40]},
            {"frame": 2, "bbox": [54, 20, 64, 40], "area": 100, "centroid": [59, 30], "foot": [59, 40]},
        ],
    }
    metrics = occlusion_metrics(samples, frame_size=(100, 60))[1]
    assert metrics["exited_frame"] == 1
    assert metrics["id_retained"] is True
