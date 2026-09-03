from typing import Literal

from pydantic import BaseModel, Field


class LiveTrack(BaseModel):
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str = "person"
    mask: list[tuple[float, float]] = Field(default_factory=list)
    mask_source: Literal["lightweight", "sam"] = "lightweight"
    trail: list[tuple[float, float]] = Field(default_factory=list)
    speed_px_s: float = 0.0
    speed_kmh: float | None = None


class LiveFrame(BaseModel):
    type: Literal["frame"] = "frame"
    frame_id: int
    width: int
    height: int
    inference_ms: float
    processing_fps: float
    selected_id: int | None
    tracks: list[LiveTrack]


class LiveStatus(BaseModel):
    type: Literal["status"] = "status"
    state: Literal["ready", "loading", "error"]
    message: str
