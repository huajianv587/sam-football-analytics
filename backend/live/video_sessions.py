"""Sequential video precompute and indexed results for deterministic seeking."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

import cv2
from websockets.asyncio.client import connect

from .server import FRAME_HEADER


@dataclass
class VideoSession:
    session_id: str
    source_path: Path
    result_path: Path
    filename: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration_s: float
    state: str = "queued"
    stage: str = "decode"
    progress: float = 0.0
    processed_frames: int = 0
    track_count: int = 0
    message: str | None = None
    frames: list[dict] = field(default_factory=list)
    refinements: dict[str, dict] = field(default_factory=dict)


class VideoSessionManager:
    def __init__(self, root: Path | None = None, live_ws_url: str | None = None) -> None:
        self.root = (root or Path(".cache/live-video")).resolve()
        self.live_ws_url = live_ws_url or os.getenv(
            "LIVE_WS_URL", "ws://127.0.0.1:8010/v1/live/ws"
        )
        self.sessions: dict[str, VideoSession] = {}
        self.tasks: dict[str, asyncio.Task[None]] = {}

    def create(self, content: bytes, filename: str) -> VideoSession:
        session_id = str(uuid4())
        directory = self.root / session_id
        directory.mkdir(parents=True, exist_ok=True)
        source_path = directory / "source.mp4"
        source_path.write_bytes(content)
        capture = cv2.VideoCapture(str(source_path))
        if not capture.isOpened():
            raise ValueError("Unable to decode the uploaded video")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 15.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()
        if not width or not height or not total_frames:
            raise ValueError("Uploaded video has no readable frames")
        session = VideoSession(
            session_id=session_id,
            source_path=source_path,
            result_path=directory / "frames.json.gz",
            filename=filename,
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration_s=round(total_frames / max(fps, 1.0), 3),
        )
        self.sessions[session_id] = session
        self.tasks[session_id] = asyncio.create_task(self._run(session))
        return session

    def status(self, session_id: str) -> VideoSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise LookupError(f"video session {session_id} was not found") from exc

    def frame_batch(self, session_id: str, start: int, end: int) -> list[dict]:
        session = self.status(session_id)
        if session.state != "ready":
            raise ValueError("video analysis is not ready")
        start = max(0, start)
        end = min(session.total_frames, max(start, end))
        return session.frames[start:end]

    async def refine(self, session_id: str, track_id: int, center_frame: int, radius: int) -> dict:
        session = self.status(session_id)
        if session.state != "ready":
            raise ValueError("video analysis is not ready")
        key = f"{track_id}:{center_frame}:{radius}"
        cached = session.refinements.get(key)
        if cached:
            return cached
        result = {"state": "queued", "track_id": track_id, "frame_start": 0, "frame_end": 0}
        session.refinements[key] = result
        asyncio.create_task(self._run_refinement(session, key, track_id, center_frame, radius))
        return result

    def refinement_status(self, session_id: str, track_id: int, center_frame: int, radius: int) -> dict:
        session = self.status(session_id)
        key = f"{track_id}:{center_frame}:{radius}"
        return session.refinements.get(key, {"state": "missing", "track_id": track_id})

    async def _run(self, session: VideoSession) -> None:
        session.state = "running"
        session.stage = "detect"
        try:
            async with connect(self.live_ws_url, max_size=16 * 1024 * 1024) as websocket:
                loading = json.loads(await websocket.recv())
                ready = json.loads(await websocket.recv())
                if loading.get("state") != "loading" or ready.get("state") != "ready":
                    raise RuntimeError(ready.get("message") or "A40 live worker was not ready")
                capture = cv2.VideoCapture(str(session.source_path))
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    encoded_ok, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
                    )
                    if not encoded_ok:
                        raise RuntimeError("Unable to encode video frame")
                    frame_id = session.processed_frames
                    await websocket.send(
                        FRAME_HEADER.pack(frame_id, time.monotonic()) + encoded.tobytes()
                    )
                    payload = json.loads(await websocket.recv())
                    if payload.get("type") != "frame":
                        raise RuntimeError(payload.get("message") or "A40 returned an invalid frame")
                    session.frames.append(payload)
                    session.processed_frames += 1
                    session.track_count = max(
                        session.track_count,
                        len(payload.get("tracks") or []),
                    )
                    session.progress = round(
                        session.processed_frames / max(session.total_frames, 1) * 100, 2
                    )
                capture.release()
            if session.processed_frames != session.total_frames:
                raise RuntimeError(
                    f"decoded {session.processed_frames} of {session.total_frames} frames"
                )
            with gzip.open(session.result_path, "wt", encoding="utf-8") as handle:
                json.dump(session.frames, handle, separators=(",", ":"))
            session.stage = "finalize"
            session.progress = 100.0
            session.state = "ready"
        except Exception as exc:
            session.state = "failed"
            session.stage = "failed"
            session.message = str(exc)[:500]

    async def _run_refinement(
        self,
        session: VideoSession,
        key: str,
        track_id: int,
        center_frame: int,
        radius: int,
    ) -> None:
        result = session.refinements[key]
        start = max(0, center_frame - max(0, radius))
        end = min(session.total_frames - 1, center_frame + max(0, radius))
        result.update({"state": "running", "frame_start": start, "frame_end": end})
        try:
            capture = cv2.VideoCapture(str(session.source_path))
            if not capture.isOpened():
                raise RuntimeError("Unable to reopen source video")
            async with connect(self.live_ws_url, max_size=16 * 1024 * 1024) as websocket:
                await websocket.recv()
                await websocket.recv()
                masks: list[dict] = []
                for index in range(end + 1):
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError(f"Unable to decode refinement frame {index}")
                    if index < start:
                        continue
                    source = next(
                        (item for item in session.frames if item.get("frame_id") == index), None
                    )
                    target = next(
                        (item for item in (source or {}).get("tracks", []) if item.get("track_id") == track_id),
                        None,
                    )
                    if not target:
                        continue
                    encoded_ok, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82]
                    )
                    if not encoded_ok:
                        continue
                    await websocket.send(json.dumps({"type": "refine_box", "bbox": target["bbox"]}))
                    await websocket.send(FRAME_HEADER.pack(index, time.monotonic()) + encoded.tobytes())
                    payload = json.loads(await websocket.recv())
                    refined = payload.get("refined_mask") or []
                    if refined:
                        masks.append({"frame": index, "mask": refined})
                capture.release()
            result.update({"state": "ready", "frames": masks})
            with gzip.open(session.result_path.with_name(f"sam-{key.replace(':', '-')}.json.gz"), "wt") as handle:
                json.dump(result, handle, separators=(",", ":"))
        except Exception as exc:
            result.update({"state": "failed", "message": str(exc)[:500]})
