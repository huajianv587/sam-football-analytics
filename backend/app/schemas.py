from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
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
    prompts: list[NormalizedBox] = Field(min_length=1, max_length=30)
    calibration: list[CalibrationPair] = Field(min_length=4, max_length=8)
    team_colors: TeamColors = Field(default_factory=TeamColors)


class JobResponse(BaseModel):
    project_id: UUID
    state: JobState
    slurm_job_id: str | None = None
    progress: int = Field(ge=0, le=100)
    message: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    scheduler: Literal["tc2-slurm"] = "tc2-slurm"
