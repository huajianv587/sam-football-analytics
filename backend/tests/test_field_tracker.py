import cv2
import numpy as np

from worker.field_tracker import (
    field_space_track,
    interpolate_calibrations,
    resample_tracks,
)


def write_video(path, frame_count: int) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (96, 64))
    for _ in range(frame_count):
        writer.write(np.zeros((64, 96, 3), dtype=np.uint8))
    writer.release()


def detection_state(confidences: list[float]) -> dict:
    return {
        "fps": 10,
        "width": 96,
        "height": 64,
        "frames": [
            {
                "index": index,
                "tracks": [
                    {
                        "track_id": index + 100,
                        "bbox": [10 + index, 10, 20 + index, 30],
                        "confidence": confidence,
                        "role": "player",
                    }
                ],
            }
            for index, confidence in enumerate(confidences)
        ],
    }


def calibrations(frame_count: int) -> list[dict]:
    return [
        {
            "index": index,
            "homography": np.eye(3).reshape(-1).tolist(),
            "calibration_confidence": 1,
        }
        for index in range(frame_count)
    ]


def test_low_confidence_detections_recover_but_never_create_track(tmp_path) -> None:
    video = tmp_path / "input.mp4"
    write_video(video, 8)
    tracked, metrics = field_space_track(
        detection_state([0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.9]),
        calibrations(8),
        video,
    )
    assert metrics["created_tracks"] == 1
    assert metrics["retained_tracks"] == 1
    assert metrics["low_confidence_recoveries"] == 6
    assert {item["track_id"] for frame in tracked["frames"] for item in frame["tracks"]} == {1}

    low_only, low_metrics = field_space_track(
        detection_state([0.2] * 8), calibrations(8), video
    )
    assert low_metrics["created_tracks"] == 0
    assert not any(frame["tracks"] for frame in low_only["frames"])


def test_calibration_and_track_resampling_are_temporal_not_repeated() -> None:
    calibration_state = {
        "fps": 5,
        "frames": [
            {"index": 0, "homography": np.eye(3).reshape(-1).tolist(), "calibration_confidence": 1},
            {
                "index": 1,
                "homography": np.asarray([[1, 0, 10], [0, 1, 0], [0, 0, 1]]).reshape(-1).tolist(),
                "calibration_confidence": 0.8,
            },
        ],
    }
    interpolated = interpolate_calibrations(calibration_state, 3, 10)
    assert np.asarray(interpolated[1]["homography"]).reshape(3, 3)[0, 2] == 5

    tracked = {
        "fps": 10,
        "width": 100,
        "height": 50,
        "frames": [
            {"index": 0, "tracks": [{"track_id": 1, "bbox": [0, 0, 10, 20], "confidence": 0.9}]},
            {"index": 1, "tracks": [{"track_id": 1, "bbox": [2, 0, 12, 20], "confidence": 0.8}]},
        ],
    }
    expanded = resample_tracks(tracked, calibrations(3), 3, 20)
    assert expanded["frames"][1]["tracks"][0]["bbox"][0] == 1


def test_impossible_field_jump_starts_a_new_identity(tmp_path) -> None:
    video = tmp_path / "jump.mp4"
    write_video(video, 14)
    state = detection_state([0.9] * 14)
    for index, frame in enumerate(state["frames"]):
        left = 10 + index if index < 7 else 70 + index - 7
        frame["tracks"][0]["bbox"] = [left, 10, left + 10, 30]
    tracked, metrics = field_space_track(state, calibrations(14), video)
    assert metrics["retained_tracks"] == 2
    assert {
        item["track_id"] for frame in tracked["frames"] for item in frame["tracks"]
    } == {1, 2}


def test_small_calibration_zoom_jitter_does_not_fragment_identity(tmp_path) -> None:
    video = tmp_path / "calibration-jitter.mp4"
    write_video(video, 12)
    jittered = calibrations(12)
    for index, calibration in enumerate(jittered):
        calibration["homography"] = np.asarray(
            [[1, 0, 4 if index % 2 else 0], [0, 1, 0], [0, 0, 1]]
        ).reshape(-1).tolist()
    tracked, metrics = field_space_track(
        detection_state([0.9] * 12), jittered, video
    )
    assert metrics["created_tracks"] == 1
    assert {
        item["track_id"] for frame in tracked["frames"] for item in frame["tracks"]
    } == {1}
