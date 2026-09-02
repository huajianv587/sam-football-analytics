import os

os.environ["AUTH_DISABLED"] = "true"
os.environ["LOCAL_ADMIN_USER_ID"] = "00000000-0000-4000-8000-000000000001"

from fastapi.testclient import TestClient

from app.config import get_settings
from app.auth import current_user_id
from app.job_runner import numeric_mask_files
from app.main import app
from app.supabase_gateway import identity_update_values


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "scheduler": "tc2-slurm"}


def test_auth_disabled_uses_the_configured_admin() -> None:
    import asyncio

    assert asyncio.run(current_user_id()) == "00000000-0000-4000-8000-000000000001"


def test_job_and_identity_endpoints_require_a_bearer_token(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_DISABLED", "false")
    get_settings.cache_clear()
    try:
        client = TestClient(app)
        project_id = "00000000-0000-4000-8000-000000000001"
        assert client.get(f"/v1/jobs/{project_id}").status_code == 401
        assert client.patch(
            f"/v1/projects/{project_id}/tracks/1/identity",
            json={"roster_id": None},
        ).status_code == 401
    finally:
        monkeypatch.setenv("AUTH_DISABLED", "true")
        get_settings.cache_clear()


def test_manual_identity_override_and_automatic_restore() -> None:
    track = {
        "team": "Argentina",
        "metrics": {
            "automatic_identity": {
                "team": "Argentina",
                "jersey_number": 10,
                "confidence": 0.82,
            }
        },
    }
    roster = {
        "id": 7,
        "team": "Argentina",
        "squad_number": 10,
        "player_name": "Test Player",
    }
    manual = identity_update_values(track, roster, manual=True)
    assert manual["identity_source"] == "manual"
    assert manual["player_name"] == "Test Player"
    restored = identity_update_values(track, roster, manual=False)
    assert restored["identity_source"] == "automatic"
    assert restored["identity_confidence"] == 0.82


def test_identity_restore_can_return_to_unidentified() -> None:
    track = {
        "team": "Spain",
        "metrics": {"automatic_identity": {"team": "Spain", "jersey_number": None}},
    }
    restored = identity_update_values(track, None, manual=False)
    assert restored["identity_source"] == "unidentified"
    assert restored["player_name"] is None


def test_mask_upload_ignores_macos_appledouble_files(tmp_path) -> None:
    for name in ("._17.json.gz", "21.json.gz", "3.json.gz"):
        (tmp_path / name).write_bytes(b"data")
    assert [path.name for path in numeric_mask_files(tmp_path)] == [
        "3.json.gz",
        "21.json.gz",
    ]
