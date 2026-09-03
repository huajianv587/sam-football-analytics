import asyncio
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


def test_mac_fallback_completes_when_a40_worker_is_unreachable(tmp_path: Path) -> None:
    manager, session = make_session(tmp_path)
    session.state = "queued"
    calls: list[str] = []

    async def fake_worker(current: VideoSession, worker_url: str) -> None:
        calls.append(worker_url)
        if worker_url == "ws://a40/v1/live/ws":
            raise ConnectionRefusedError("A40 unavailable")
        current.processed_frames = current.total_frames
        current.progress = 100.0
        current.track_count = 2

    manager.live_ws_url = "ws://a40/v1/live/ws"
    manager.local_ws_url = "ws://mac/v1/live/ws"
    manager._run_with_worker = fake_worker  # type: ignore[method-assign]

    asyncio.run(manager._run(session))

    assert calls == ["ws://a40/v1/live/ws", "ws://mac/v1/live/ws"]
    assert session.state == "ready"
    assert session.executor == "mac_mps"
    assert session.result_path.is_file()


def test_mac_fallback_refinement_reports_its_capability_boundary(tmp_path: Path) -> None:
    manager, session = make_session(tmp_path)
    session.executor = "mac_mps"

    result = asyncio.run(manager.refine(session.session_id, 1, 1, 15))

    assert result["state"] == "failed"
    assert "requires A40" in result["message"]
