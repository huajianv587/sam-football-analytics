import os

os.environ["AUTH_DISABLED"] = "true"

from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "scheduler": "tc2-slurm"}
