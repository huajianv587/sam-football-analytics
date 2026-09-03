from __future__ import annotations

import asyncio
import json
import os
import struct
from functools import lru_cache

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .engine import LiveInferenceEngine
from .protocol import LiveStatus

FRAME_HEADER = struct.Struct("!Id")


@lru_cache(maxsize=1)
def engine() -> LiveInferenceEngine:
    return LiveInferenceEngine()


app = FastAPI(title="PitchVision Live Inference", version="0.2.0")


@app.get("/health")
async def health() -> dict:
    loaded = engine.cache_info().currsize > 0
    return {
        "status": "ok",
        "mode": "generic-person-instance-segmentation",
        "loaded": loaded,
        "lightweight_model": os.getenv("LIVE_SEG_MODEL", "yolo11s-seg.pt"),
        "sam_enabled": os.getenv("LIVE_SAM_ENABLED", "true").lower() == "true",
    }


@app.websocket("/v1/live/ws")
async def live_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    selected_id: int | None = None
    try:
        await websocket.send_json(LiveStatus(
            state="loading", message="Loading the generic person segmentation models"
        ).model_dump())
        live_engine = await asyncio.to_thread(engine)
        await asyncio.to_thread(live_engine.reset)
        await websocket.send_json(LiveStatus(
            state="ready", message="A40 live inference is ready"
        ).model_dump())
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            if message.get("text") is not None:
                control = json.loads(message["text"])
                if control.get("type") == "select":
                    selected_id = control.get("track_id")
                elif control.get("type") == "reset":
                    selected_id = None
                    await asyncio.to_thread(live_engine.reset)
                continue
            payload = message.get("bytes")
            if not payload:
                continue
            if len(payload) <= FRAME_HEADER.size:
                raise ValueError("Frame payload is missing JPEG data")
            frame_id, timestamp = FRAME_HEADER.unpack(payload[: FRAME_HEADER.size])
            result = await asyncio.to_thread(
                live_engine.process,
                frame_id,
                timestamp,
                payload[FRAME_HEADER.size :],
                selected_id,
            )
            await websocket.send_json(result.model_dump())
    except WebSocketDisconnect:
        return
    except Exception as exc:
        await websocket.send_json(LiveStatus(state="error", message=str(exc)).model_dump())
        await websocket.close(code=1011)
