import json
from pathlib import Path
from typing import Any

from supabase import Client, create_client

from .config import get_settings


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

    def roster(self, match_label: str) -> list[dict[str, Any]]:
        return self.client.table("roster").select("team,squad_number,player_name,position").eq(
            "match_label", match_label
        ).execute().data

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

    def replace_tracks(self, project_id: str, tracks_path: Path) -> None:
        tracks = json.loads(tracks_path.read_text())
        self.client.table("tracks").delete().eq("project_id", project_id).execute()
        rows = [{**track, "project_id": project_id} for track in tracks]
        if rows:
            self.client.table("tracks").insert(rows).execute()
