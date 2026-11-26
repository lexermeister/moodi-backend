from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE"])
def test_configured_web_origin_can_preflight(recordings_dir: Path, method: str) -> None:
    settings = Settings(
        recordings_dir=recordings_dir, cors_allowed_origins=["https://demo.example"]
    )
    with TestClient(create_app(settings)) as client:
        response = client.options(
            "/recordings",
            headers={
                "Origin": "https://demo.example",
                "Access-Control-Request-Method": method,
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://demo.example"
    assert "access-control-allow-credentials" not in response.headers


def test_unlisted_web_origins_cannot_preflight(recordings_dir: Path) -> None:
    settings = Settings(
        recordings_dir=recordings_dir, cors_allowed_origins=["https://demo.example"]
    )
    with TestClient(create_app(settings)) as client:
        response = client.options(
            "/recordings",
            headers={
                "Origin": "https://other.example",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_web_origin_headers_on_success_and_validation_errors(recordings_dir: Path) -> None:
    settings = Settings(
        recordings_dir=recordings_dir, cors_allowed_origins=["https://demo.example"]
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/health", headers={"Origin": "https://demo.example"})
        upload = client.post("/recordings", headers={"Origin": "https://demo.example"})
    assert health.status_code == 200
    assert upload.status_code == 422
    for response in [health, upload]:
        assert response.headers["access-control-allow-origin"] == "https://demo.example"


def test_cors_is_disabled_by_default(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "https://demo.example"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_origins_can_be_configured_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", '["http://localhost:8082"]')
    assert Settings(_env_file=None).cors_allowed_origins == ["http://localhost:8082"]
