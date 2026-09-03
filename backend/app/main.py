import json

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .auth import current_user_id
from .config import get_settings
from .job_runner import JobRunner
from .schemas import (
    AnalysisMode,
    CreateJobRequest,
    FaceEmbeddingRequest,
    FaceMatchResponse,
    FaceProfileResponse,
    HealthResponse,
    IdentityUpdateRequest,
    JobResponse,
    RefinementResponse,
)
from .supabase_gateway import SupabaseGateway

app = FastAPI(title="SAM Football Analytics API", version="0.1.0")
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH"],
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


@app.post("/v1/face-profiles", response_model=FaceProfileResponse, status_code=201)
async def create_face_profile(
    label: str = Form(..., min_length=1, max_length=120),
    embedding: str = Form(...),
    photo: UploadFile | None = File(default=None),
    owner_id: str = Depends(current_user_id),
) -> FaceProfileResponse:
    try:
        vector = FaceEmbeddingRequest(embedding=json.loads(embedding)).embedding
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail="embedding must be a JSON float array") from exc
    photo_bytes = await photo.read(5 * 1024 * 1024 + 1) if photo else None
    if photo_bytes and len(photo_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="face photo must be 5 MB or less")
    if photo and photo.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="face photo must be JPEG, PNG or WebP")
    return FaceProfileResponse.model_validate(
        runner().gateway.create_face_profile(owner_id, label, vector, photo_bytes)
    )


@app.get("/v1/face-profiles", response_model=list[FaceProfileResponse])
async def list_face_profiles(owner_id: str = Depends(current_user_id)) -> list[FaceProfileResponse]:
    return [FaceProfileResponse.model_validate(row) for row in runner().gateway.face_profiles(owner_id)]


@app.post("/v1/face-profiles/match", response_model=FaceMatchResponse)
async def match_face(
    request: FaceEmbeddingRequest, owner_id: str = Depends(current_user_id)
) -> FaceMatchResponse:
    return FaceMatchResponse.model_validate(runner().gateway.match_face(owner_id, request.embedding))


@app.post("/v1/jobs", response_model=JobResponse, status_code=202)
async def create_job(request: CreateJobRequest, owner_id: str = Depends(current_user_id)) -> JobResponse:
    try:
        return await runner().submit(request, owner_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/v1/offline/jobs", response_model=JobResponse, status_code=202)
async def create_offline_job(
    video: UploadFile = File(),
    title: str = Form("Offline Football Analysis"),
    match_label: str = Form("Unspecified Match"),
    team_a: str = Form("Team A"),
    team_b: str = Form("Team B"),
    owner_id: str = Depends(current_user_id),
) -> JobResponse:
    if video.content_type != "video/mp4":
        raise HTTPException(status_code=415, detail="Only MP4 video is supported")
    content = await video.read(50 * 1024 * 1024 + 1)
    if not content:
        raise HTTPException(status_code=422, detail="Video is empty")
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video size must be 50 MB or less")
    gateway = runner().gateway
    project = gateway.create_project({
        "owner_id": owner_id,
        "title": title,
        "match_label": match_label,
        "team_a": team_a,
        "team_b": team_b,
        "analysis_mode": AnalysisMode.AUTO_ALL.value,
        "stage": "queued",
        "progress": 0,
    })
    project_id = str(project["id"])
    source_path = f"{owner_id}/{project_id}/source.mp4"
    gateway.upload_video(source_path, content)
    gateway.update_project(project_id, {"source_path": source_path})
    request = CreateJobRequest(
        project_id=project_id,
        source_path=source_path,
        analysis_mode=AnalysisMode.AUTO_ALL,
    )
    return await runner().submit(request, owner_id)


@app.get("/v1/jobs/{project_id}", response_model=JobResponse)
async def job_status(project_id: str, owner_id: str = Depends(current_user_id)) -> JobResponse:
    try:
        return runner().status(project_id, owner_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/offline/jobs/{project_id}", response_model=JobResponse)
async def offline_job_status(
    project_id: str, owner_id: str = Depends(current_user_id)
) -> JobResponse:
    try:
        return runner().status(project_id, owner_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/offline/latest", response_model=JobResponse)
async def latest_offline_job(owner_id: str = Depends(current_user_id)) -> JobResponse:
    try:
        project = runner().gateway.latest_project(owner_id)
        return runner().status(str(project["id"]), owner_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/projects/{project_id}/results")
async def project_results(
    project_id: str, owner_id: str = Depends(current_user_id)
) -> dict:
    try:
        return runner().gateway.results_bundle(project_id, owner_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.patch("/v1/projects/{project_id}/tracks/{object_id}/identity")
async def update_track_identity(
    project_id: str,
    object_id: int,
    request: IdentityUpdateRequest,
    owner_id: str = Depends(current_user_id),
) -> dict:
    try:
        return runner().gateway.update_track_identity(
            project_id, object_id, owner_id, request.roster_id
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/v1/projects/{project_id}/tracks/{object_id}/refine",
    response_model=RefinementResponse,
    status_code=202,
)
async def refine_track(
    project_id: str,
    object_id: int,
    owner_id: str = Depends(current_user_id),
) -> RefinementResponse:
    try:
        return RefinementResponse.model_validate(
            await runner().submit_refinement(project_id, object_id, owner_id)
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/v1/projects/{project_id}/tracks/{object_id}/refine",
    response_model=RefinementResponse,
)
async def refinement_status(
    project_id: str,
    object_id: int,
    owner_id: str = Depends(current_user_id),
) -> RefinementResponse:
    try:
        return RefinementResponse.model_validate(
            runner().refinement_status(project_id, object_id, owner_id)
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
