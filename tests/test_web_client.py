from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.actions import MoodEntryValues
from tests.test_actions import entry_values
from tests.test_session_insights import FakeOpenAIClient


def test_browser_origin_can_preflight_and_read_api(recordings_dir: Path):
    settings = Settings(recordings_dir=recordings_dir, cors_origins=["http://localhost:8081"])
    with TestClient(create_app(settings)) as client:
        preflight = client.options(
            "/recordings",
            headers={
                "Origin": "http://localhost:8081",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers["access-control-allow-origin"] == "http://localhost:8081"
        response = client.get("/health", headers={"Origin": "http://localhost:8081"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:8081"
        denied = client.get("/health", headers={"Origin": "https://untrusted.invalid"})
        assert "access-control-allow-origin" not in denied.headers


def test_browser_webm_can_be_uploaded_transcribed_and_summarized(recordings_dir: Path):
    provider = FakeOpenAIClient()
    settings = Settings(recordings_dir=recordings_dir, openai_api_key="test-key")
    with TestClient(create_app(settings, transcription_client=provider)) as client:
        audio = b"\x1a\x45\xdf\xa3browser-test-audio"
        uploaded = client.post(
            "/recordings", files={"file": ("session.webm", audio, "audio/webm;codecs=opus")}
        )
        assert uploaded.status_code == 201
        recording = uploaded.json()
        assert (recordings_dir / recording["filename"]).read_bytes() == audio
        transcript = client.post(f"/recordings/{recording['id']}/transcription")
        assert transcript.status_code == 200
        assert transcript.json()["status"] == "completed"
        insights = client.post(f"/recordings/{recording['id']}/insights", json={})
        assert insights.status_code == 200
        assert insights.json()["result"]["status"] == "draft"


def test_browser_reflection_steps_round_trip_without_discarding_answers():
    values = entry_values()
    values.update(
        thoughts="Work stayed on my mind", actions_taken="Went for a walk", current_step="thoughts"
    )
    saved = MoodEntryValues.model_validate(values)
    assert saved.thoughts == "Work stayed on my mind"
    assert saved.actions_taken == "Went for a walk"
    assert saved.current_step == "thoughts"
