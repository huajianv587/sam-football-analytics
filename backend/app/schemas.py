from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisMode(StrEnum):
    MANUAL_SAM = "manual_sam"
    AUTO_ALL = "auto_all"


class JobStage(StrEnum):
    DRAFT = "draft"
    QUEUED = "queued"
    NORMALIZE = "normalize"
    RECONSTRUCT = "reconstruct"
    DETECT = "detect"
    TRACK = "track"
    CALIBRATE = "calibrate"
    SEGMENT = "segment"
    IDENTIFY = "identify"
    UPLOAD = "upload"
    COMPLETED = "completed"
    FAILED = "failed"


class NormalizedBox(BaseModel):
    object_id: int = Field(ge=1)
    box: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_box(self) -> "NormalizedBox":
        x1, y1, x2, y2 = self.box
        if not all(0 <= value <= 1 for value in self.box) or x1 >= x2 or y1 >= y2:
            raise ValueError("box must be normalized as x1 < x2 and y1 < y2")
        return self


class CalibrationPair(BaseModel):
    video: tuple[float, float]
    pitch: tuple[float, float]

    @model_validator(mode="after")
    def validate_pair(self) -> "CalibrationPair":
        if not all(0 <= value <= 1 for value in self.video):
            raise ValueError("video point must be normalized")
        x, y = self.pitch
        if not 0 <= x <= 105 or not 0 <= y <= 68:
            raise ValueError("pitch point must be inside a 105m x 68m field")
        return self


class TeamColors(BaseModel):
    team_a: tuple[int, int, int] = (187, 38, 46)
    team_b: tuple[int, int, int] = (130, 200, 235)
    referee: tuple[int, int, int] = (30, 32, 36)


class CreateJobRequest(BaseModel):
    project_id: UUID
    source_path: str
    analysis_mode: AnalysisMode = AnalysisMode.AUTO_ALL
    prompts: list[NormalizedBox] = Field(default_factory=list, max_length=30)
    calibration: list[CalibrationPair] = Field(default_factory=list, max_length=8)
    team_colors: TeamColors = Field(default_factory=TeamColors)

    @model_validator(mode="after")
    def validate_manual_mode(self) -> "CreateJobRequest":
        if self.analysis_mode == AnalysisMode.MANUAL_SAM:
            if not self.prompts:
                raise ValueError("manual_sam requires at least one prompt box")
            if len(self.calibration) < 4:
                raise ValueError("manual_sam requires at least four calibration pairs")
        return self


class JobResponse(BaseModel):
    project_id: UUID
    state: JobState
    slurm_job_id: str | None = None
    progress: int = Field(ge=0, le=100)
    stage: JobStage
    track_count: int = Field(default=0, ge=0)
    message: str | None = None


class IdentityUpdateRequest(BaseModel):
    roster_id: int | None = Field(default=None, ge=1)


class FaceEmbeddingRequest(BaseModel):
    embedding: list[float] = Field(min_length=32, max_length=2048)


class FaceProfileResponse(BaseModel):
    id: str
    label: str
    photo_path: str | None = None
    created_at: str | None = None


class FaceMatchResponse(BaseModel):
    profile_id: str | None
    label: str | None
    score: float


class RefinementResponse(BaseModel):
    project_id: UUID
    object_id: int = Field(ge=1)
    state: Literal["base_ready", "queued", "running", "large_ready", "failed"]
    slurm_job_id: str | None = None
    mask_url: str | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    scheduler: Literal["tc2-slurm"] = "tc2-slurm"


class VideoSessionResponse(BaseModel):
    session_id: str
    filename: str
    state: Literal["queued", "running", "ready", "failed"]
    stage: str
    progress: float = Field(ge=0, le=100)
    processed_frames: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    duration_s: float = Field(ge=0)
    fps: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    track_count: int = Field(ge=0)
    message: str | None = None


class VideoRefineRequest(BaseModel):
    center_frame: int = Field(ge=0)
    radius: int = Field(default=15, ge=0, le=60)
