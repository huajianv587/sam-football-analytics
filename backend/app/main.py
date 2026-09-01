import json

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from .auth import current_user_id
from .config import get_settings
from .job_runner import JobRunner
from .schemas import CalibrationPair, CreateJobRequest, HealthResponse, JobResponse, NormalizedBox
from .supabase_gateway import SupabaseGateway

app = FastAPI(title="SAM Football Analytics API", version="0.1.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

_runner: JobRunner | None = None


def runner() -> JobRunner:
    global _runner
    if _runner is None:
        _runner = JobRunner(SupabaseGateway())
    return _runner


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/v1/jobs", response_model=JobResponse, status_code=202)
async def create_job(request: CreateJobRequest, owner_id: str = Depends(current_user_id)) -> JobResponse:
    try:
        return await runner().submit(request, owner_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/v1/offline/jobs", response_model=JobResponse, status_code=202)
async def create_offline_job(
    video: UploadFile = File(),
    prompts: str = Form(),
    calibration: str = Form(),
    title: str = Form("Offline Football Analysis"),
    team_a: str = Form("Spain"),
    team_b: str = Form("Argentina"),
) -> JobResponse:
    owner_id = settings.local_admin_user_id
    if not owner_id:
        raise HTTPException(status_code=503, detail="LOCAL_ADMIN_USER_ID is not configured")
    content = await video.read(50 * 1024 * 1024 + 1)
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video size must be 50 MB or less")
    try:
        prompt_models = [NormalizedBox.model_validate(item) for item in json.loads(prompts)]
        calibration_models = [CalibrationPair.model_validate(item) for item in json.loads(calibration)]
        request_data = {"prompts": prompt_models, "calibration": calibration_models}
        CreateJobRequest.model_validate({"project_id": "00000000-0000-0000-0000-000000000000", "source_path": "pending", **request_data})
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise HTTPException(status_code=422, detail="Prompt boxes or calibration data are invalid") from exc

    gateway = runner().gateway
    project = gateway.create_project({
        "owner_id": owner_id,
        "title": title,
        "match_label": "2026 FIFA World Cup Final",
        "team_a": team_a,
        "team_b": team_b,
        "prompts": [item.model_dump(mode="json") for item in prompt_models],
        "calibration": [item.model_dump(mode="json") for item in calibration_models],
    })
    project_id = str(project["id"])
    source_path = f"{owner_id}/{project_id}/source.mp4"
    gateway.upload_video(source_path, content)
    gateway.update_project(project_id, {"source_path": source_path})
    request = CreateJobRequest(
        project_id=project_id,
        source_path=source_path,
        prompts=prompt_models,
        calibration=calibration_models,
    )
    return await runner().submit(request, owner_id)


@app.get("/v1/jobs/{project_id}", response_model=JobResponse)
async def job_status(project_id: str, owner_id: str = Depends(current_user_id)) -> JobResponse:
    try:
        return runner().status(project_id, owner_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/offline/jobs/{project_id}", response_model=JobResponse)
async def offline_job_status(project_id: str) -> JobResponse:
    if not settings.local_admin_user_id:
        raise HTTPException(status_code=503, detail="LOCAL_ADMIN_USER_ID is not configured")
    try:
        return runner().status(project_id, settings.local_admin_user_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
