import struct

import numpy as np
from fastapi.testclient import TestClient

from live.protocol import LiveFrame, LiveTrack
import live.server as live_server
from live.server import FRAME_HEADER
from live.tracking import MotionHistory, largest_polygon, simplify_polygon


def test_live_frame_header_matches_browser_network_packet() -> None:
    packet = FRAME_HEADER.pack(42, 12.5) + b"jpeg"
    frame_id, timestamp = struct.unpack("!Id", packet[:12])
    assert frame_id == 42
    assert timestamp == 12.5
    assert packet[12:] == b"jpeg"


def test_motion_history_uses_box_foot_and_smoothed_pixel_speed() -> None:
    history = MotionHistory(alpha=0.35)
    first = history.update(7, 1.0, (0, 0, 10, 20))
    second = history.update(7, 2.0, (10, 0, 20, 20))
    assert first.point == (5, 20)
    assert second.point == (15, 20)
    assert second.speed_px_s == 3.5
    assert history.trail(7) == [(5, 20), (15, 20)]


def test_motion_history_removes_tracks_that_leave_the_stream() -> None:
    history = MotionHistory(max_missing_frames=1)
    history.update(1, 0, (0, 0, 10, 20))
    history.update(2, 0, (0, 0, 10, 20))
    history.retain({2})
    assert set(history.samples) == {1, 2}
    history.retain({2})
    assert set(history.samples) == {2}


def test_mask_contours_are_compact_polygons() -> None:
    mask = np.zeros((50, 60), dtype=np.uint8)
    mask[10:40, 20:50] = 1
    polygon = largest_polygon(mask)
    assert 4 <= len(polygon) <= 5
    simplified = simplify_polygon(np.asarray([
        [20, 10], [30, 10], [40, 10], [50, 10],
        [50, 40], [20, 40], [20, 20],
    ]))
    assert len(simplified) <= 5


def test_live_protocol_keeps_metric_speed_missing_without_calibration() -> None:
    frame = LiveFrame(
        frame_id=1,
        width=1280,
        height=720,
        inference_ms=50,
        processing_fps=15,
        selected_id=None,
        tracks=[LiveTrack(
            track_id=3,
            bbox=(1, 2, 3, 4),
            confidence=0.8,
            speed_px_s=12,
        )],
    )
    assert frame.tracks[0].speed_kmh is None
    assert frame.tracks[0].mask_source == "lightweight"


def test_websocket_stream_selects_one_track_without_queuing_frames(monkeypatch) -> None:
    class FakeEngine:
        def reset(self) -> None:
            pass

        def process(self, frame_id, timestamp, jpeg, selected_id, refine_bbox=None):
            assert timestamp == 2.5
            assert jpeg == b"jpeg"
            return LiveFrame(
                frame_id=frame_id,
                width=640,
                height=360,
                inference_ms=10,
                processing_fps=20,
                selected_id=selected_id,
            tracks=[],
        )

    monkeypatch.setattr(live_server, "engine", lambda: FakeEngine())
    with TestClient(live_server.app).websocket_connect("/v1/live/ws") as websocket:
        assert websocket.receive_json()["state"] == "loading"
        assert websocket.receive_json()["state"] == "ready"
        websocket.send_text('{"type":"select","track_id":7}')
        websocket.send_bytes(FRAME_HEADER.pack(9, 2.5) + b"jpeg")
        frame = websocket.receive_json()
        assert frame["frame_id"] == 9
        assert frame["selected_id"] == 7
