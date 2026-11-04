import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.transcription import OPENAI_MAX_FILE_BYTES


class FakeTranscriptions:
    def __init__(
        self,
        error: Exception | None = None,
        response_text: str = "Ovo je test transkripcije.",
    ) -> None:
        self.error = error
        self.response_text = response_text
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        audio_file = kwargs.pop("file")
        self.calls.append(
            {
                **kwargs,
                "filename": Path(audio_file.name).name,
                "audio_bytes": audio_file.read(),
            }
        )
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            text=self.response_text,
            languages=[SimpleNamespace(code="sr")],
        )


class FakeOpenAIClient:
    def __init__(
        self,
        error: Exception | None = None,
        response_text: str = "Ovo je test transkripcije.",
    ) -> None:
        self.transcriptions = FakeTranscriptions(error, response_text)
        self.audio = SimpleNamespace(transcriptions=self.transcriptions)


@pytest.fixture
def fake_openai() -> FakeOpenAIClient:
    return FakeOpenAIClient()


@pytest.fixture
def transcription_client(
    recordings_dir: Path,
    fake_openai: FakeOpenAIClient,
) -> Iterator[TestClient]:
    settings = Settings(
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )
    with TestClient(create_app(settings, transcription_client=fake_openai)) as test_client:
        yield test_client


def upload_recording(client: TestClient, filename: str = "voice.m4a") -> dict[str, Any]:
    content_type = "audio/aac" if filename.endswith(".aac") else "audio/mp4"
    response = client.post(
        "/recordings",
        files={"file": (filename, b"audio test bytes", content_type)},
    )
    assert response.status_code == 201
    return response.json()


def test_transcription_returns_and_saves_completed_transcript(
    transcription_client: TestClient,
    recordings_dir: Path,
    fake_openai: FakeOpenAIClient,
) -> None:
    recording = upload_recording(transcription_client)

    response = transcription_client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 200
    payload = response.json()
    assert payload["recording_id"] == recording["id"]
    assert payload["text"] == "Ovo je test transkripcije."
    assert payload["languages"] == [{"code": "sr"}]
    assert payload["model"] == "gpt-transcribe"
    assert payload["status"] == "completed"

    transcript_path = recordings_dir / f"{recording['id']}.transcript.json"
    saved = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert saved == payload

    assert len(fake_openai.transcriptions.calls) == 1
    call = fake_openai.transcriptions.calls[0]
    assert call["model"] == "gpt-transcribe"
    assert call["filename"] == recording["filename"]
    assert call["audio_bytes"] == b"audio test bytes"
    assert call["extra_body"] == {"languages": ["sr", "en"]}


def test_transcription_rejects_unknown_recording(
    transcription_client: TestClient,
    fake_openai: FakeOpenAIClient,
) -> None:
    response = transcription_client.post(f"/recordings/{uuid4()}/transcription")

    assert response.status_code == 404
    assert fake_openai.transcriptions.calls == []


def test_transcription_rejects_invalid_recording_id(transcription_client: TestClient) -> None:
    response = transcription_client.post("/recordings/../../etc/passwd/transcription")

    assert response.status_code in {404, 422}


def test_transcription_rejects_oversized_recording_before_openai(
    transcription_client: TestClient,
    recordings_dir: Path,
    fake_openai: FakeOpenAIClient,
) -> None:
    recording_id = uuid4()
    recording_path = recordings_dir / f"{recording_id}.wav"
    with recording_path.open("wb") as recording:
        recording.truncate(OPENAI_MAX_FILE_BYTES + 1)

    response = transcription_client.post(f"/recordings/{recording_id}/transcription")

    assert response.status_code == 413
    assert recording_path.exists()
    assert fake_openai.transcriptions.calls == []
    assert not (recordings_dir / f"{recording_id}.transcript.json").exists()


def test_transcription_rejects_raw_aac_before_openai(
    transcription_client: TestClient,
    recordings_dir: Path,
    fake_openai: FakeOpenAIClient,
) -> None:
    recording = upload_recording(transcription_client, filename="voice.aac")

    response = transcription_client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 415
    assert (recordings_dir / recording["filename"]).exists()
    assert fake_openai.transcriptions.calls == []


def test_openai_failure_preserves_recording_and_does_not_save_transcript(
    recordings_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_openai = FakeOpenAIClient(error=RuntimeError("provider unavailable"))
    settings = Settings(
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )

    with TestClient(create_app(settings, transcription_client=fake_openai)) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 502
    assert response.json() == {"detail": "Could not transcribe recording"}
    assert (recordings_dir / recording["filename"]).exists()
    assert not (recordings_dir / f"{recording['id']}.transcript.json").exists()

    logs = caplog.text
    assert "transcription_provider_failed" in logs
    assert recording["id"] in logs
    assert "provider_error_type=RuntimeError" in logs
    assert "provider unavailable" not in logs
    assert "test-key" not in logs


def test_no_speech_returns_validation_response_and_does_not_save_transcript(
    recordings_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_openai = FakeOpenAIClient(response_text="   ")
    settings = Settings(
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )

    with TestClient(create_app(settings, transcription_client=fake_openai)) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 422
    assert response.json() == {"detail": "No speech was detected in the recording"}
    assert (recordings_dir / recording["filename"]).exists()
    assert not (recordings_dir / f"{recording['id']}.transcript.json").exists()
    assert "reason=no_speech" in caplog.text
    assert "transcription_provider_failed" not in caplog.text


def test_transcription_requires_api_key(recordings_dir: Path) -> None:
    settings = Settings(recordings_dir=recordings_dir, max_upload_mb=30, openai_api_key=None)

    with TestClient(create_app(settings)) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 503
    assert response.json() == {"detail": "OpenAI transcription is not configured"}


def test_recording_lookup_uses_uuid_inside_storage_directory(
    transcription_client: TestClient,
    recordings_dir: Path,
) -> None:
    recording = upload_recording(transcription_client)
    recording_id = UUID(recording["id"])

    response = transcription_client.post(f"/recordings/{recording_id}/transcription")

    assert response.status_code == 200
    assert (recordings_dir / f"{recording_id}.transcript.json").is_file()
