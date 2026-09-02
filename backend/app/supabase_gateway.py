import json
from pathlib import Path
from typing import Any

from supabase import Client, create_client

from .config import get_settings


def identity_update_values(
    track: dict[str, Any], roster: dict[str, Any] | None, *, manual: bool
) -> dict[str, Any]:
    automatic = (track.get("metrics") or {}).get("automatic_identity") or {}
    if manual:
        if roster is None:
            raise ValueError("manual identity requires a roster player")
        return {
            "roster_id": roster["id"],
            "team": roster["team"],
            "jersey_number": roster["squad_number"],
            "player_name": roster["player_name"],
            "identity_source": "manual",
            "identity_confidence": 1,
        }
    if roster is not None:
        return {
            "roster_id": roster["id"],
            "team": roster["team"],
            "jersey_number": roster["squad_number"],
            "player_name": roster["player_name"],
            "identity_source": "automatic",
            "identity_confidence": float(automatic.get("confidence") or 0),
        }
    return {
        "roster_id": None,
        "team": automatic.get("team", track["team"]),
        "jersey_number": automatic.get("jersey_number"),
        "player_name": None,
        "identity_source": "unidentified",
        "identity_confidence": float(automatic.get("confidence") or 0),
    }


class SupabaseGateway:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.supabase_url or not settings.supabase_secret_key:
            raise RuntimeError("Supabase backend credentials are not configured")
        self.client: Client = create_client(settings.supabase_url, settings.supabase_secret_key)

    def project(self, project_id: str, owner_id: str | None = None) -> dict[str, Any]:
        query = self.client.table("projects").select("*").eq("id", project_id)
        if owner_id:
            query = query.eq("owner_id", owner_id)
        rows = query.limit(1).execute().data
        if not rows:
            raise LookupError("project not found")
        return rows[0]

    def update_project(self, project_id: str, values: dict[str, Any]) -> None:
        self.client.table("projects").update(values).eq("id", project_id).execute()

    def create_project(self, values: dict[str, Any]) -> dict[str, Any]:
        return self.client.table("projects").insert(values).execute().data[0]

    def latest_project(self, owner_id: str) -> dict[str, Any]:
        rows = self.client.table("projects").select("*").eq(
            "owner_id", owner_id
        ).order("created_at", desc=True).limit(1).execute().data
        if not rows:
            raise LookupError("no analysis project found")
        return rows[0]

    def roster(self, match_label: str) -> list[dict[str, Any]]:
        return self.client.table("roster").select("id,team,squad_number,player_name,position").eq(
            "match_label", match_label
        ).execute().data

    def track_count(self, project_id: str) -> int:
        response = self.client.table("tracks").select("id", count="exact").eq(
            "project_id", project_id
        ).execute()
        return int(response.count or 0)

    def download_video(self, object_path: str, destination: Path) -> None:
        destination.write_bytes(self.client.storage.from_("videos").download(object_path))

    def upload_video(self, object_path: str, content: bytes) -> None:
        self.client.storage.from_("videos").upload(
            object_path,
            content,
            {"content-type": "video/mp4", "upsert": "true"},
        )

    def upload_artifact(self, object_path: str, source: Path, content_type: str) -> None:
        self.client.storage.from_("artifacts").upload(
            object_path,
            source.read_bytes(),
            {"content-type": content_type, "upsert": "true"},
        )

    def results_bundle(self, project_id: str, owner_id: str) -> dict[str, Any]:
        project = self.project(project_id, owner_id)
        if project["status"] != "completed":
            raise LookupError("analysis results are not ready")
        tracks = self.client.table("tracks").select("*").eq(
            "project_id", project_id
        ).order("object_id").execute().data
        roster = self.roster(project["match_label"])
        storage = self.client.storage.from_("artifacts")

        core_paths = {
            "video": project.get("normalized_video_path"),
            "foreground": project.get("foreground_video_path"),
            "metrics": project.get("metrics_path"),
            "legacyMasks": project.get("mask_manifest_path"),
        }
        mask_tracks = [track for track in tracks if track.get("mask_path")]
        paths = [path for path in core_paths.values() if path] + [
            track["mask_path"] for track in mask_tracks
        ]
        signed_items = storage.create_signed_urls(paths, 3600)
        signed_by_path = {
            path: item["signedUrl"] for path, item in zip(paths, signed_items, strict=True)
        }
        masks_by_track = {
            str(track["object_id"]): signed_by_path[track["mask_path"]]
            for track in mask_tracks
        }
        urls = {
            key: signed_by_path.get(path) if path else None
            for key, path in core_paths.items()
        } | {
            "masksByTrack": masks_by_track,
        }
        if not all(urls[key] for key in ("video", "foreground", "metrics")):
            raise LookupError("analysis artifacts are incomplete")
        return {"project": project, "tracks": tracks, "roster": roster, "urls": urls}

    def replace_tracks(
        self,
        project_id: str,
        tracks_path: Path,
        mask_paths: dict[int, str] | None = None,
    ) -> None:
        tracks = json.loads(tracks_path.read_text())
        self.client.table("tracks").delete().eq("project_id", project_id).execute()
        rows = []
        for track in tracks:
            object_id = int(track["object_id"])
            row = {**track, "project_id": project_id}
            if mask_paths:
                row["mask_path"] = mask_paths.get(object_id)
            rows.append(row)
        if rows:
            self.client.table("tracks").insert(rows).execute()

    def update_track_identity(
        self,
        project_id: str,
        object_id: int,
        owner_id: str,
        roster_id: int | None,
    ) -> dict[str, Any]:
        project = self.project(project_id, owner_id)
        track_rows = self.client.table("tracks").select("*").eq(
            "project_id", project_id
        ).eq("object_id", object_id).limit(1).execute().data
        if not track_rows:
            raise LookupError("track not found")
        track = track_rows[0]

        if roster_id is None:
            automatic_id = track.get("auto_roster_id")
            roster = self._roster_entry(automatic_id, project["match_label"]) if automatic_id else None
            values = identity_update_values(track, roster, manual=False)
        else:
            roster = self._roster_entry(roster_id, project["match_label"])
            if roster["team"] not in {project["team_a"], project["team_b"]}:
                raise ValueError("roster player is not part of this project")
            values = identity_update_values(track, roster, manual=True)
        return self.client.table("tracks").update(values).eq(
            "project_id", project_id
        ).eq("object_id", object_id).execute().data[0]

    def _roster_entry(self, roster_id: int, match_label: str) -> dict[str, Any]:
        rows = self.client.table("roster").select(
            "id,match_label,team,squad_number,player_name,position"
        ).eq("id", roster_id).eq("match_label", match_label).limit(1).execute().data
        if not rows:
            raise LookupError("roster player not found")
        return rows[0]
