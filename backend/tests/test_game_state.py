import numpy as np
import pytest

from app.schemas import CreateJobRequest
from gsr.export_state import calibration_support, jersey_value
from gsr.role_logic import (
    parse_role,
    representative_detections,
    resolve_role_with_pitch,
    vote_role,
)
from worker.game_state import (
    detections_by_track,
    frame_calibrations,
    on_pitch_tracks,
    resample_game_state,
    select_prompt_detection,
    select_prompt_detections,
    track_continuity_metrics,
    track_windows,
)
from worker.main import jersey_collage, mask_quality, roster_match
from scripts.validate_artifacts import continuity_report


def test_auto_job_does_not_require_boxes_or_calibration() -> None:
    job = CreateJobRequest(
        project_id="00000000-0000-0000-0000-000000000001",
        source_path="owner/project/source.mp4",
    )
    assert job.analysis_mode == "auto_all"
    assert job.prompts == []


def test_manual_job_keeps_legacy_validation() -> None:
    with pytest.raises(ValueError, match="prompt box"):
        CreateJobRequest(
            project_id="00000000-0000-0000-0000-000000000001",
            source_path="owner/project/source.mp4",
            analysis_mode="manual_sam",
        )


def test_only_valid_field_roles_become_tracks() -> None:
    state = {
        "frames": [
            {
                "index": 0,
                "tracks": [
                    {"track_id": 1, "bbox": [0, 0, 10, 20], "role": "player"},
                    {"track_id": 2, "bbox": [0, 0, 8, 8], "role": "ball"},
                    {"track_id": 3, "bbox": [0, 0, 10, 20], "role": "referee"},
                ],
            }
        ]
    }
    assert set(detections_by_track(state)) == {1, 3}


def test_prompt_selection_prefers_confident_large_box() -> None:
    selected = select_prompt_detection(
        [
            {"frame": 0, "bbox": [0, 0, 10, 20], "confidence": 0.9},
            {"frame": 1, "bbox": [0, 0, 30, 60], "confidence": 0.8},
        ]
    )
    assert selected["frame"] == 1


def test_prompt_selection_stays_near_lifecycle_center_when_available() -> None:
    selected = select_prompt_detection(
        [
            {"frame": 0, "bbox": [0, 0, 50, 100], "confidence": 0.99},
            {"frame": 45, "bbox": [0, 0, 30, 60], "confidence": 0.8},
            {"frame": 90, "bbox": [0, 0, 50, 100], "confidence": 0.99},
        ],
        preferred_frame=45,
        temporal_radius=15,
    )
    assert selected["frame"] == 45


def test_prompt_anchors_cover_early_middle_and_late_track_regions() -> None:
    detections = [
        {"frame": frame, "bbox": [0, 0, 20, 40], "confidence": 0.9}
        for frame in range(0, 91, 5)
    ]
    selected = select_prompt_detections(detections)
    frames = [item["frame"] for item in selected]
    assert len(frames) == 3
    assert frames[0] < 30
    assert 30 <= frames[1] <= 60
    assert frames[2] > 60


def test_mask_quality_rejects_a_different_person_box() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:30, 10:20] = True
    assert mask_quality(mask, {"bbox": [10, 10, 20, 30]}) == 1
    assert mask_quality(mask, {"bbox": [70, 70, 90, 95]}) == 0


def test_track_windows_overlap_without_exceeding_lifecycle() -> None:
    assert track_windows(0, 449, window_size=180, overlap=30) == [
        (0, 179),
        (150, 329),
        (300, 449),
    ]


def test_jersey_collage_has_fixed_shape_and_preserves_pixels() -> None:
    first = np.full((40, 20, 3), 120, dtype=np.uint8)
    second = np.full((20, 50, 3), 240, dtype=np.uint8)

    collage = jersey_collage([first, second])

    assert collage.shape == (192, 640, 3)
    assert collage[:, :128].max() == 120
    assert collage[:, 128:256].max() == 240
    assert collage[:, 256:].max() == 0


def test_missing_role_defaults_to_player_but_explicit_ball_is_excluded() -> None:
    state = {
        "frames": [
            {
                "index": 0,
                "tracks": [
                    {"track_id": 1, "bbox": [0, 0, 10, 20]},
                    {"track_id": 2, "bbox": [0, 0, 5, 5], "role": "ball"},
                ],
            }
        ]
    }
    assert set(detections_by_track(state)) == {1}


def test_off_pitch_people_are_removed_without_hiding_uncalibrated_tracks() -> None:
    tracks = {
        1: [{"pitch": {"x_bottom_middle": 10, "y_bottom_middle": 5}}] * 30,
        2: [{"pitch": {"x_bottom_middle": 80, "y_bottom_middle": 50}}] * 30,
        3: [{"pitch": None}] * 30,
        4: [{"pitch": None}] * 29,
    }
    assert set(on_pitch_tracks(tracks)) == {1, 3}


def test_low_confidence_calibration_is_invalid() -> None:
    state = {
        "frames": [
            {"index": 0, "homography": list(range(9)), "calibration_confidence": 0.49},
            {"index": 1, "homography": list(range(9)), "calibration_confidence": 0.8},
        ]
    }
    calibration = frame_calibrations(state)
    assert calibration[0]["valid"] is False
    assert calibration[1]["valid"] is True


def test_low_rate_game_state_repeats_into_full_timeline() -> None:
    state = {
        "fps": 7.5,
        "width": 100,
        "height": 50,
        "frames": [
            {"index": 0, "tracks": [{"track_id": 7, "bbox": [0, 0, 10, 20]}]},
            {"index": 1, "tracks": [{"track_id": 7, "bbox": [2, 0, 12, 20]}]},
        ],
    }
    expanded = resample_game_state(state, target_frame_count=4, target_fps=15)
    assert expanded["fps"] == 15
    assert [frame["index"] for frame in expanded["frames"]] == [0, 1, 2, 3]
    assert [frame["tracks"][0]["bbox"][0] for frame in expanded["frames"]] == [0, 0, 2, 2]


def test_calibration_support_reflects_visible_pitch_evidence() -> None:
    assert calibration_support({"keypoints": {str(index): {} for index in range(6)}}) == 0.5
    assert calibration_support({"keypoints": {}, "lines_det": {}}) == 0
    assert calibration_support({"keypoints": {str(index): {} for index in range(20)}}) == 1


def test_roster_match_requires_high_confidence_team_and_number() -> None:
    roster = [
        {"id": 7, "team": "Spain", "squad_number": 10, "player_name": "Player", "position": "FW"}
    ]
    assert roster_match(roster, "Spain", 10, 0.8) == roster[0]
    assert roster_match(roster, "Spain", 10, 0.4) is None
    assert roster_match(roster, "Argentina", 10, 0.8) is None


def test_jersey_parser_never_guesses_invalid_text() -> None:
    assert jersey_value("10.0") == 10
    assert jersey_value("unreadable") is None
    assert jersey_value(100) is None


def test_role_samples_are_high_quality_and_temporally_distinct() -> None:
    samples = representative_detections(
        [
            {"frame": 0, "bbox": [0, 0, 10, 20], "confidence": 0.9},
            {"frame": 1, "bbox": [0, 0, 20, 40], "confidence": 0.9},
            {"frame": 12, "bbox": [0, 0, 15, 30], "confidence": 0.8},
        ],
        samples_per_track=2,
    )
    assert [sample["frame"] for sample in samples] == [1, 12]


def test_role_output_is_strict_and_multiframe_voted() -> None:
    assert parse_role("referee\n") == "referee"
    assert parse_role("probably referee") is None
    assert vote_role(["referee", "referee", "player"]) == ("referee", 0.667)


def test_midfield_goalkeeper_prediction_resolves_to_referee() -> None:
    midfield = [
        {"pitch": {"x_bottom_middle": -3.5}},
        {"pitch": {"x_bottom_middle": 2.0}},
    ]
    penalty_area = [
        {"pitch": {"x_bottom_middle": 44.0}},
        {"pitch": {"x_bottom_middle": 47.0}},
    ]
    assert resolve_role_with_pitch("goalkeeper", midfield) == "referee"
    assert resolve_role_with_pitch("goalkeeper", penalty_area) == "goalkeeper"


def test_continuity_report_rejects_global_reid_false_merges() -> None:
    report = continuity_report(
        [
            {
                "object_id": 7,
                "detections": [
                    {"frame": 0, "bbox": [0, 0, 20, 50]},
                    {"frame": 2, "bbox": [400, 0, 420, 50]},
                    {"frame": 100, "bbox": [405, 0, 425, 50]},
                ],
            }
        ]
    )
    assert report["impossible_short_step_events"] == 1
    assert report["long_gap_events_over_3s"] == 1


def test_worker_continuity_metrics_count_tracks_and_visible_people() -> None:
    metrics = track_continuity_metrics(
        {
            1: [
                {"frame": 0, "bbox": [0, 0, 20, 50]},
                {"frame": 2, "bbox": [400, 0, 420, 50]},
            ],
            2: [{"frame": 0, "bbox": [30, 0, 50, 50]}],
        }
    )
    assert metrics["impossible_short_step_events"] == 1
    assert metrics["visible_people_median"] == 1.5
    assert metrics["visible_people_max"] == 2
