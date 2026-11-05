from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.main import create_app
from app.models import Transcription
from app.services.transcription import OPENAI_MAX_FILE_BYTES


class FakeTranscriptions:
    def __init__(
        self,
        error: Exception | None = None,
        response_text: str = "Ovo je test transkripcije.",
        response_request_id: str = "req_test_success",
    ) -> None:
        self.error = error
        self.response_text = response_text
        self.response_request_id = response_request_id
        self.before_response: Callable[[], None] | None = None
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
        if self.before_response is not None:
            self.before_response()
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            _request_id=self.response_request_id,
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


class FakeProviderError(RuntimeError):
    status_code = 503
    request_id = "req_test_failure"


@pytest.fixture
def fake_openai() -> FakeOpenAIClient:
    return FakeOpenAIClient()


@pytest.fixture
def transcription_client(
    recordings_dir: Path,
    database_engine: Engine,
    fake_openai: FakeOpenAIClient,
) -> Iterator[TestClient]:
    settings = Settings(
        app_env="testing",
        database_url="sqlite+pysqlite://",
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )
    with TestClient(
        create_app(
            settings,
            transcription_client=fake_openai,
            database_engine=database_engine,
        )
    ) as test_client:
        yield test_client


def upload_recording(client: TestClient, filename: str = "voice.m4a") -> dict[str, Any]:
    content_types = {".aac": "audio/aac", ".m4a": "audio/mp4", ".wav": "audio/wav"}
    content_type = content_types[Path(filename).suffix]
    response = client.post(
        "/recordings",
        files={"file": (filename, b"audio test bytes", content_type)},
    )
    assert response.status_code == 201
    return response.json()


def test_transcription_returns_and_saves_completed_attempt(
    transcription_client: TestClient,
    recordings_dir: Path,
    database_engine: Engine,
    fake_openai: FakeOpenAIClient,
) -> None:
    recording = upload_recording(transcription_client)

    response = transcription_client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 200
    payload = response.json()
    transcription_id = UUID(payload["transcription_id"])
    assert payload["recording_id"] == recording["id"]
    assert payload["text"] == "Ovo je test transkripcije."
    assert payload["languages"] == [{"code": "sr"}]
    assert payload["model"] == "gpt-transcribe"
    assert payload["status"] == "completed"

    assert not (recordings_dir / f"{recording['id']}.transcript.json").exists()
    with Session(database_engine) as session:
        saved = session.get(Transcription, transcription_id)
        assert saved is not None
        assert saved.recording_id == UUID(recording["id"])
        assert saved.status == "completed"
        assert saved.provider == "openai"
        assert saved.provider_request_id == "req_test_success"
        assert saved.text == "Ovo je test transkripcije."
        assert saved.languages == ["sr"]
        assert saved.error_code is None

    assert len(fake_openai.transcriptions.calls) == 1
    call = fake_openai.transcriptions.calls[0]
    assert call["model"] == "gpt-transcribe"
    assert call["filename"] == recording["filename"]
    assert call["audio_bytes"] == b"audio test bytes"
    assert call["extra_body"] == {"languages": ["sr", "en"]}
    assert call["extra_headers"] == {"X-Client-Request-Id": str(transcription_id)}


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
    recording = upload_recording(transcription_client, filename="voice.wav")
    recording_id = UUID(recording["id"])
    recording_path = recordings_dir / recording["filename"]
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
    database_engine: Engine,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_openai = FakeOpenAIClient(error=FakeProviderError("provider unavailable"))
    settings = Settings(
        app_env="testing",
        database_url="sqlite+pysqlite://",
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )

    with TestClient(
        create_app(
            settings,
            transcription_client=fake_openai,
            database_engine=database_engine,
        )
    ) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 502
    assert response.json() == {"detail": "Could not transcribe recording"}
    assert (recordings_dir / recording["filename"]).exists()
    assert not (recordings_dir / f"{recording['id']}.transcript.json").exists()

    with Session(database_engine) as session:
        attempt = session.scalars(select(Transcription)).one()
        assert attempt.status == "failed"
        assert attempt.provider_request_id == "req_test_failure"
        assert attempt.error_code == "provider_error"
        assert attempt.text is None

    logs = caplog.text
    assert "transcription_provider_failed" in logs
    assert recording["id"] in logs
    assert "provider_error_type=FakeProviderError" in logs
    assert "provider_request_id=req_test_failure" in logs
    assert "provider unavailable" not in logs
    assert "test-key" not in logs


def test_no_speech_returns_validation_response_and_does_not_save_transcript(
    recordings_dir: Path,
    database_engine: Engine,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_openai = FakeOpenAIClient(response_text="   ")
    settings = Settings(
        app_env="testing",
        database_url="sqlite+pysqlite://",
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )

    with TestClient(
        create_app(
            settings,
            transcription_client=fake_openai,
            database_engine=database_engine,
        )
    ) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 422
    assert response.json() == {"detail": "No speech was detected in the recording"}
    assert (recordings_dir / recording["filename"]).exists()
    assert not (recordings_dir / f"{recording['id']}.transcript.json").exists()
    with Session(database_engine) as session:
        attempt = session.scalars(select(Transcription)).one()
        assert attempt.status == "failed"
        assert attempt.provider_request_id == "req_test_success"
        assert attempt.error_code == "no_speech"
    assert "reason=no_speech" in caplog.text
    assert "transcription_provider_failed" not in caplog.text


def test_transcription_requires_api_key(recordings_dir: Path, database_engine: Engine) -> None:
    settings = Settings(
        app_env="testing",
        database_url="sqlite+pysqlite://",
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key=None,
    )

    with TestClient(create_app(settings, database_engine=database_engine)) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 503
    assert response.json() == {"detail": "OpenAI transcription is not configured"}


def test_recording_lookup_uses_database_storage_key(
    transcription_client: TestClient,
    recordings_dir: Path,
    database_engine: Engine,
) -> None:
    recording = upload_recording(transcription_client)
    recording_id = UUID(recording["id"])

    response = transcription_client.post(f"/recordings/{recording_id}/transcription")

    assert response.status_code == 200
    assert not (recordings_dir / f"{recording_id}.transcript.json").exists()
    with Session(database_engine) as session:
        attempt = session.scalars(select(Transcription)).one()
        assert attempt.recording_id == recording_id


def test_retry_creates_a_separate_transcription_attempt(
    transcription_client: TestClient,
    database_engine: Engine,
) -> None:
    recording = upload_recording(transcription_client)

    first = transcription_client.post(f"/recordings/{recording['id']}/transcription")
    second = transcription_client.post(f"/recordings/{recording['id']}/transcription")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["transcription_id"] != second.json()["transcription_id"]
    with Session(database_engine) as session:
        attempts = session.scalars(
            select(Transcription).where(Transcription.recording_id == UUID(recording["id"]))
        ).all()
        assert len(attempts) == 2


def test_attempt_is_committed_before_provider_call(
    transcription_client: TestClient,
    database_engine: Engine,
    fake_openai: FakeOpenAIClient,
) -> None:
    observed_statuses: list[str] = []

    def observe_attempt() -> None:
        with Session(database_engine) as session:
            observed_statuses.append(session.scalars(select(Transcription)).one().status)

    fake_openai.transcriptions.before_response = observe_attempt
    recording = upload_recording(transcription_client)

    response = transcription_client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 200
    assert observed_statuses == ["started"]
