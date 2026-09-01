import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from .config import get_settings
from .schemas import CreateJobRequest, JobResponse, JobState
from .supabase_gateway import SupabaseGateway


CONTENT_TYPES = {
    "masks.json.gz": "application/gzip",
    "tracks.json": "application/json",
    "foreground.mp4": "video/mp4",
    "metrics.json": "application/json",
    "normalized.mp4": "video/mp4",
}


class JobRunner:
    def __init__(self, gateway: SupabaseGateway) -> None:
        self.gateway = gateway
        self.settings = get_settings()
        self.tasks: dict[str, asyncio.Task[None]] = {}

    async def submit(self, request: CreateJobRequest, owner_id: str) -> JobResponse:
        project_id = str(request.project_id)
        project = self.gateway.project(project_id, owner_id)
        if project["status"] in {"queued", "running"}:
            return self.status(project_id, owner_id)

        payload = request.model_dump(mode="json")
        payload["roster"] = self.gateway.roster(project["match_label"])
        payload["team_a"] = project["team_a"]
        payload["team_b"] = project["team_b"]
        self.gateway.update_project(project_id, {"status": "queued", "error_message": None})
        self.tasks[project_id] = asyncio.create_task(self._run(project_id, payload))
        return JobResponse(project_id=request.project_id, state=JobState.QUEUED, progress=5)

    def status(self, project_id: str, owner_id: str) -> JobResponse:
        project = self.gateway.project(project_id, owner_id)
        state = JobState(project["status"])
        progress = {JobState.QUEUED: 5, JobState.RUNNING: 50, JobState.COMPLETED: 100, JobState.FAILED: 100}[state]
        return JobResponse(
            project_id=project["id"],
            state=state,
            slurm_job_id=project.get("slurm_job_id"),
            progress=progress,
            message=project.get("error_message"),
        )

    async def _run(self, project_id: str, payload: dict[str, Any]) -> None:
        local_dir = self.settings.local_job_root.resolve() / project_id
        remote_dir = f"{self.settings.tc2_remote_root}/jobs/{project_id}"
        try:
            if local_dir.exists():
                shutil.rmtree(local_dir)
            local_dir.mkdir(parents=True)
            self.gateway.download_video(payload["source_path"], local_dir / "source.mp4")
            (local_dir / "payload.json").write_text(json.dumps(payload, ensure_ascii=False))

            await self._remote("mkdir", "-p", remote_dir)
            await self._rsync(
                str(local_dir / "source.mp4"),
                str(local_dir / "payload.json"),
                str(Path(__file__).parents[1] / "worker"),
                str(Path(__file__).parents[1] / "scripts" / "job.sbatch"),
                destination=f"{self._host()}:{remote_dir}/",
            )
            output = await self._remote("sbatch", "--parsable", f"{remote_dir}/job.sbatch", remote_dir)
            slurm_job_id = output.strip().split(";")[0]
            self.gateway.update_project(project_id, {"status": "running", "slurm_job_id": slurm_job_id})

            state = await self._wait_for_job(slurm_job_id)
            if state != "COMPLETED":
                raise RuntimeError(f"Slurm job {slurm_job_id} ended with {state}")

            results_dir = local_dir / "results"
            results_dir.mkdir()
            await self._rsync(f"{self._host()}:{remote_dir}/results/", destination=str(results_dir) + "/")
            for name in CONTENT_TYPES:
                artifact = results_dir / name
                if not artifact.is_file() or artifact.stat().st_size == 0:
                    raise RuntimeError(f"missing output artifact: {name}")

            artifact_paths: dict[str, str] = {}
            owner_id = str(self.gateway.project(project_id)["owner_id"])
            for name, content_type in CONTENT_TYPES.items():
                object_path = f"{owner_id}/{project_id}/{name}"
                self.gateway.upload_artifact(object_path, results_dir / name, content_type)
                artifact_paths[name] = object_path
            self.gateway.replace_tracks(project_id, results_dir / "tracks.json")
            self.gateway.update_project(
                project_id,
                {
                    "status": "completed",
                    "normalized_video_path": artifact_paths["normalized.mp4"],
                    "mask_manifest_path": artifact_paths["masks.json.gz"],
                    "foreground_video_path": artifact_paths["foreground.mp4"],
                    "metrics_path": artifact_paths["metrics.json"],
                },
            )
        except Exception as exc:
            self.gateway.update_project(project_id, {"status": "failed", "error_message": str(exc)[:500]})
        finally:
            self.tasks.pop(project_id, None)

    async def _wait_for_job(self, job_id: str) -> str:
        while True:
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
            *command,
        )

    async def _rsync(self, *sources: str, destination: str) -> str:
        return await self._command(
            "rsync", "-az", "-e", f"ssh -p {self.settings.tc2_port} -o BatchMode=yes", *sources, destination
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
