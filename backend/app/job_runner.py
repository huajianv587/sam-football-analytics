import asyncio
import json
import shlex
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import get_settings
from .schemas import AnalysisMode, CreateJobRequest, JobResponse, JobStage, JobState
from .supabase_gateway import SupabaseGateway


CORE_CONTENT_TYPES = {
    "tracks.json": "application/json",
    "foreground.mp4": "video/mp4",
    "metrics.json": "application/json",
    "normalized.mp4": "video/mp4",
}


def numeric_mask_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.glob("*.json.gz")
            if path.name.removesuffix(".json.gz").isdigit()
        ),
        key=lambda path: int(path.name.removesuffix(".json.gz")),
    )


class JobRunner:
    def __init__(self, gateway: SupabaseGateway) -> None:
        self.gateway = gateway
        self.settings = get_settings()
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.refinement_tasks: dict[str, asyncio.Task[None]] = {}
        self.discovered_counts: dict[str, int] = {}

    async def submit_refinement(
        self, project_id: str, object_id: int, owner_id: str
    ) -> dict[str, Any]:
        project = self.gateway.project(project_id, owner_id)
        if project["status"] != "completed":
            raise ValueError("project analysis is not complete")
        track = self.gateway.track(project_id, object_id)
        status = (track.get("metrics") or {}).get("mask_refinement_status")
        task_key = f"{project_id}:{object_id}"
        if status in {"queued", "running", "large_ready"}:
            return self.refinement_status(project_id, object_id, owner_id)
        self.gateway.update_track_metrics(
            project_id,
            object_id,
            {"mask_refinement_status": "queued", "mask_refinement_error": None},
        )
        self.refinement_tasks[task_key] = asyncio.create_task(
            self._run_refinement(project_id, object_id)
        )
        return self.refinement_status(project_id, object_id, owner_id)

    def refinement_status(
        self, project_id: str, object_id: int, owner_id: str
    ) -> dict[str, Any]:
        self.gateway.project(project_id, owner_id)
        track = self.gateway.track(project_id, object_id)
        metrics = track.get("metrics") or {}
        status = metrics.get("mask_refinement_status") or (
            "large_ready" if metrics.get("mask_model_tier") == "large" else "base_ready"
        )
        mask_url = (
            self.gateway.signed_artifact(track["mask_path"])
            if status == "large_ready" and track.get("mask_path")
            else None
        )
        return {
            "project_id": project_id,
            "object_id": object_id,
            "state": status,
            "slurm_job_id": metrics.get("mask_refinement_job_id"),
            "mask_url": mask_url,
            "message": metrics.get("mask_refinement_error"),
        }

    async def _run_refinement(self, project_id: str, object_id: int) -> None:
        task_key = f"{project_id}:{object_id}"
        local_dir = (
            self.settings.local_job_root.resolve()
            / "refinements"
            / project_id
            / str(object_id)
        )
        remote_dir = (
            f"{self.settings.tc2_remote_root}/refinements/{project_id}/"
            f"{object_id}-{uuid4().hex[:10]}"
        )
        try:
            if local_dir.exists():
                shutil.rmtree(local_dir, ignore_errors=True)
            local_dir.mkdir(parents=True)
            project = self.gateway.project(project_id)
            track = self.gateway.track(project_id, object_id)
            normalized_path = project.get("normalized_video_path")
            if not normalized_path:
                raise RuntimeError("normalized video artifact is missing")
            self.gateway.download_artifact(normalized_path, local_dir / "normalized.mp4")
            (local_dir / "payload.json").write_text(
                json.dumps(
                    {
                        "object_id": object_id,
                        "role": track.get("role", "player"),
                        "detections": track.get("detections") or [],
                    }
                )
            )
            await self._remote("mkdir", "-p", remote_dir)
            await self._rsync(
                str(local_dir / "normalized.mp4"),
                str(local_dir / "payload.json"),
                str(Path(__file__).parents[1] / "worker"),
                str(Path(__file__).parents[1] / "scripts" / "refine.sbatch"),
                destination=f"{self._host()}:{remote_dir}/",
            )
            output = await self._remote(
                "sbatch", "--parsable", f"{remote_dir}/refine.sbatch", remote_dir
            )
            slurm_job_id = output.strip().split(";")[0]
            self.gateway.update_track_metrics(
                project_id,
                object_id,
                {
                    "mask_refinement_status": "running",
                    "mask_refinement_job_id": slurm_job_id,
                },
            )
            state = await self._wait_for_slurm(slurm_job_id)
            if state != "COMPLETED":
                raise RuntimeError(f"SAM Large refinement {slurm_job_id} ended with {state}")
            result_dir = local_dir / "results"
            result_dir.mkdir()
            await self._rsync(
                f"{self._host()}:{remote_dir}/results/", destination=str(result_dir) + "/"
            )
            mask_file = result_dir / f"{object_id}.json.gz"
            if not mask_file.is_file() or mask_file.stat().st_size == 0:
                raise RuntimeError("SAM Large refinement mask is missing")
            track = self.gateway.track(project_id, object_id)
            mask_path = track.get("mask_path")
            if not mask_path:
                raise RuntimeError("track mask storage path is missing")
            self.gateway.upload_artifact(mask_path, mask_file, "application/gzip")
            self.gateway.update_track_metrics(
                project_id,
                object_id,
                {
                    "mask_model_tier": "large",
                    "mask_refinement_status": "large_ready",
                    "mask_refinement_error": None,
                },
            )
        except Exception as exc:
            self.gateway.update_track_metrics(
                project_id,
                object_id,
                {
                    "mask_refinement_status": "failed",
                    "mask_refinement_error": str(exc)[:500],
                },
            )
        finally:
            self.refinement_tasks.pop(task_key, None)

    async def _wait_for_slurm(self, job_id: str) -> str:
        while True:
            output = await self._remote(
                "sacct", "-j", job_id, "-X", "--noheader", "--parsable2", "--format=State"
            )
            state = next(
                (
                    line.split("|")[0].split("+")[0]
                    for line in output.splitlines()
                    if line.strip()
                ),
                "",
            )
            if state in {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
            }:
                return state
            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def submit(self, request: CreateJobRequest, owner_id: str) -> JobResponse:
        project_id = str(request.project_id)
        project = self.gateway.project(project_id, owner_id)
        expected_prefix = f"{owner_id}/{project_id}/"
        if not request.source_path.startswith(expected_prefix) or request.source_path != project.get(
            "source_path"
        ):
            raise ValueError("source path does not belong to this project")
        if project["status"] in {"queued", "running"}:
            return self.status(project_id, owner_id)

        payload = request.model_dump(mode="json")
        payload["roster"] = self.gateway.roster(project["match_label"])
        payload["team_a"] = project["team_a"]
        payload["team_b"] = project["team_b"]
        payload["match_label"] = project["match_label"]
        self.gateway.update_project(
            project_id,
            {
                "status": "queued",
                "stage": JobStage.QUEUED.value,
                "progress": 2,
                "analysis_mode": request.analysis_mode.value,
                "error_message": None,
            },
        )
        self.tasks[project_id] = asyncio.create_task(self._run(project_id, payload))
        self.discovered_counts[project_id] = 0
        return JobResponse(
            project_id=request.project_id,
            state=JobState.QUEUED,
            stage=JobStage.QUEUED,
            progress=2,
        )

    def status(self, project_id: str, owner_id: str) -> JobResponse:
        project = self.gateway.project(project_id, owner_id)
        state = JobState(project["status"])
        progress = int(project.get("progress") or 0)
        stage = JobStage(project.get("stage") or state.value)
        return JobResponse(
            project_id=project["id"],
            state=state,
            slurm_job_id=project.get("slurm_job_id"),
            progress=progress,
            stage=stage,
            track_count=(
                self.gateway.track_count(project_id)
                if state == JobState.COMPLETED
                else self.discovered_counts.get(project_id, 0)
            ),
            message=project.get("error_message"),
        )

    async def _run(self, project_id: str, payload: dict[str, Any]) -> None:
        local_dir = self.settings.local_job_root.resolve() / project_id
        remote_dir = f"{self.settings.tc2_remote_root}/jobs/{project_id}/{uuid4().hex[:12]}"
        try:
            if local_dir.exists():
                shutil.rmtree(local_dir, ignore_errors=True)
            local_dir.mkdir(parents=True, exist_ok=True)
            self.gateway.download_video(payload["source_path"], local_dir / "source.mp4")
            (local_dir / "payload.json").write_text(json.dumps(payload, ensure_ascii=False))

            await self._remote("mkdir", "-p", remote_dir)
            await self._rsync(
                str(local_dir / "source.mp4"),
                str(local_dir / "payload.json"),
                str(Path(__file__).parents[1] / "worker"),
                str(Path(__file__).parents[1] / "gsr"),
                str(Path(__file__).parents[1] / "scripts" / "job.sbatch"),
                destination=f"{self._host()}:{remote_dir}/",
            )
            output = await self._remote("sbatch", "--parsable", f"{remote_dir}/job.sbatch", remote_dir)
            slurm_job_id = output.strip().split(";")[0]
            self.gateway.update_project(
                project_id,
                {
                    "status": "running",
                    "stage": JobStage.NORMALIZE.value,
                    "progress": 5,
                    "slurm_job_id": slurm_job_id,
                },
            )

            state = await self._wait_for_job(slurm_job_id, project_id, remote_dir)
            if state != "COMPLETED":
                raise RuntimeError(f"Slurm job {slurm_job_id} ended with {state}")

            results_dir = local_dir / "results"
            results_dir.mkdir()
            await self._rsync(f"{self._host()}:{remote_dir}/results/", destination=str(results_dir) + "/")
            mode = AnalysisMode(payload["analysis_mode"])
            content_types = dict(CORE_CONTENT_TYPES)
            if mode == AnalysisMode.AUTO_ALL:
                content_types["calibration.json.gz"] = "application/gzip"
            else:
                content_types["masks.json.gz"] = "application/gzip"
            for name in content_types:
                artifact = results_dir / name
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise RuntimeError(f"missing output artifact: {name}")

            artifact_paths: dict[str, str] = {}
            owner_id = str(self.gateway.project(project_id)["owner_id"])
            self.gateway.update_project(
                project_id, {"stage": JobStage.UPLOAD.value, "progress": 92}
            )
            for name, content_type in content_types.items():
                object_path = f"{owner_id}/{project_id}/{name}"
                self.gateway.upload_artifact(object_path, results_dir / name, content_type)
                artifact_paths[name] = object_path

            mask_paths: dict[int, str] = {}
            masks_dir = results_dir / "masks"
            if mode == AnalysisMode.AUTO_ALL:
                mask_files = numeric_mask_files(masks_dir)
                if not mask_files:
                    raise RuntimeError("missing per-track mask artifacts")
                for mask_file in mask_files:
                    object_id = int(mask_file.name.removesuffix(".json.gz"))
                    object_path = f"{owner_id}/{project_id}/masks/{mask_file.name}"
                    self.gateway.upload_artifact(object_path, mask_file, "application/gzip")
                    mask_paths[object_id] = object_path

            self.gateway.replace_tracks(
                project_id, results_dir / "tracks.json", mask_paths=mask_paths
            )
            self.gateway.update_project(
                project_id,
                {
                    "status": "completed",
                    "stage": JobStage.COMPLETED.value,
                    "progress": 100,
                    "normalized_video_path": artifact_paths["normalized.mp4"],
                    "mask_manifest_path": artifact_paths.get("masks.json.gz"),
                    "calibration_path": artifact_paths.get("calibration.json.gz"),
                    "foreground_video_path": artifact_paths["foreground.mp4"],
                    "metrics_path": artifact_paths["metrics.json"],
                },
            )
        except Exception as exc:
            self.gateway.update_project(
                project_id,
                {
                    "status": "failed",
                    "stage": JobStage.FAILED.value,
                    "progress": 100,
                    "error_message": str(exc)[:500],
                },
            )
        finally:
            self.tasks.pop(project_id, None)

    async def _wait_for_job(self, job_id: str, project_id: str, remote_dir: str) -> str:
        last_progress: tuple[str, int] | None = None
        while True:
            progress_output = await self._remote(
                "bash", "-lc", f"test -s '{remote_dir}/progress.json' && cat '{remote_dir}/progress.json' || true"
            )
            if progress_output:
                progress = json.loads(progress_output)
                current = (str(progress["stage"]), int(progress["progress"]))
                if "track_count" in progress:
                    self.discovered_counts[project_id] = int(progress["track_count"])
                if current != last_progress:
                    self.gateway.update_project(
                        project_id,
                        {"stage": current[0], "progress": current[1]},
                    )
                    last_progress = current
            output = await self._remote(
                "sacct", "-j", job_id, "-X", "--noheader", "--parsable2", "--format=State"
            )
            state = next((line.split("|")[0].split("+")[0] for line in output.splitlines() if line.strip()), "")
            if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL"}:
                return state
            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def _remote(self, *command: str) -> str:
        return await self._command(
            "ssh",
            "-o", "BatchMode=yes",
            "-p", str(self.settings.tc2_port),
            self._host(),
            shlex.join(command),
        )

    async def _rsync(self, *sources: str, destination: str) -> str:
        return await self._command(
            "rsync", "-az", "--exclude=._*", "--exclude=__pycache__", "-e",
            f"ssh -p {self.settings.tc2_port} -o BatchMode=yes", *sources, destination
        )

    async def _command(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            raise RuntimeError(stderr.decode().strip() or f"command failed: {args[0]}")
        return stdout.decode()

    def _host(self) -> str:
        return f"{self.settings.tc2_user}@{self.settings.tc2_host}"
