from pathlib import Path

import pytest

from live.protocol import LiveFrame
from live.video_sessions import VideoSession, VideoSessionManager


def make_session(tmp_path: Path) -> tuple[VideoSessionManager, VideoSession]:
    manager = VideoSessionManager(root=tmp_path)
    session = VideoSession(
        session_id="session-1",
        source_path=tmp_path / "source.mp4",
        result_path=tmp_path / "frames.json.gz",
        filename="clip.mp4",
        fps=15.0,
        width=1280,
        height=720,
        total_frames=3,
        duration_s=0.2,
        state="ready",
        frames=[
            LiveFrame(
                frame_id=index,
                width=1280,
                height=720,
                inference_ms=50,
                processing_fps=15,
                selected_id=None,
                tracks=[],
            ).model_dump()
            for index in range(3)
        ],
    )
    manager.sessions[session.session_id] = session
    return manager, session


def test_frame_batch_is_clamped_to_indexed_timeline(tmp_path: Path) -> None:
    manager, _ = make_session(tmp_path)
    assert [frame["frame_id"] for frame in manager.frame_batch("session-1", -10, 99)] == [0, 1, 2]


def test_frame_batch_rejects_non_ready_sessions(tmp_path: Path) -> None:
    manager, session = make_session(tmp_path)
    session.state = "running"
    with pytest.raises(ValueError, match="not ready"):
        manager.frame_batch("session-1", 0, 1)


def test_refinement_status_is_missing_until_request_is_queued(tmp_path: Path) -> None:
    manager, _ = make_session(tmp_path)
    status = manager.refinement_status("session-1", 4, 10, 15)
    assert status["state"] == "missing"
    assert status["track_id"] == 4
