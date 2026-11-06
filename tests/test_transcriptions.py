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
from app.services.transcription import OPENAI_MAX_FILE_BYTES, RoleAttributionDecision


class FakeTranscriptions:
    def __init__(
        self,
        error: Exception | None = None,
        response_text: str = (
            "How have you been feeling this week? I've been feeling anxious. When did it start?"
        ),
        segments: list[Any] | None = None,
    ) -> None:
        self.error = error
        self.response_text = response_text
        self.segments = segments if segments is not None else default_segments()
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
            segments=self.segments,
        )


class FakeResponses:
    def __init__(
        self,
        therapist_speaker: str = "speaker_1",
        error: Exception | None = None,
        output_parsed: Any | None = None,
    ) -> None:
        self.therapist_speaker = therapist_speaker
        self.error = error
        self.output_parsed = output_parsed
        self.calls: list[dict[str, Any]] = []

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        parsed = self.output_parsed
        if parsed is None:
            parsed = RoleAttributionDecision(therapist_speaker=self.therapist_speaker)
        return SimpleNamespace(output_parsed=parsed, _request_id="role-request-id")


class FakeOpenAIClient:
    def __init__(
        self,
        error: Exception | None = None,
        role_error: Exception | None = None,
        therapist_speaker: str = "speaker_1",
        role_output_parsed: Any | None = None,
        response_text: str = (
            "How have you been feeling this week? I've been feeling anxious. When did it start?"
        ),
        segments: list[Any] | None = None,
    ) -> None:
        self.transcriptions = FakeTranscriptions(error, response_text, segments)
        self.audio = SimpleNamespace(transcriptions=self.transcriptions)
        self.responses = FakeResponses(
            therapist_speaker=therapist_speaker,
            error=role_error,
            output_parsed=role_output_parsed,
        )


def default_segments() -> list[Any]:
    return [
        SimpleNamespace(speaker="provider-a", text="How have you been feeling?"),
        SimpleNamespace(speaker="provider-a", text="This week?"),
        SimpleNamespace(speaker="provider-b", text="I've been feeling anxious."),
        SimpleNamespace(speaker="provider-a", text="When did it start?"),
    ]


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
        openai_transcription_model="gpt-4o-transcribe-diarize",
        openai_role_attribution_model="gpt-4o-mini",
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
    assert payload["text"] == (
        "How have you been feeling? This week? I've been feeling anxious. When did it start?"
    )
    assert payload["turns"] == [
        {"role": "therapist", "text": "How have you been feeling? This week?"},
        {"role": "client", "text": "I've been feeling anxious."},
        {"role": "therapist", "text": "When did it start?"},
    ]
    assert payload["languages"] == []
    assert payload["model"] == "gpt-4o-transcribe-diarize"
    assert payload["role_model"] == "gpt-4o-mini"
    assert payload["role_prompt_version"] == "v1"
    assert payload["status"] == "completed"

    transcript_path = recordings_dir / f"{recording['id']}.transcript.json"
    saved = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert saved == {
        **payload,
        "turns": [
            {
                "speaker": "speaker_1",
                "role": "therapist",
                "text": "How have you been feeling? This week?",
            },
            {
                "speaker": "speaker_2",
                "role": "client",
                "text": "I've been feeling anxious.",
            },
            {
                "speaker": "speaker_1",
                "role": "therapist",
                "text": "When did it start?",
            },
        ],
    }

    assert len(fake_openai.transcriptions.calls) == 1
    call = fake_openai.transcriptions.calls[0]
    assert call["model"] == "gpt-4o-transcribe-diarize"
    assert call["filename"] == recording["filename"]
    assert call["audio_bytes"] == b"audio test bytes"
    assert call["response_format"] == "diarized_json"
    assert call["chunking_strategy"] == "auto"
    assert "prompt" not in call

    assert len(fake_openai.responses.calls) == 1
    role_call = fake_openai.responses.calls[0]
    assert role_call["model"] == "gpt-4o-mini"
    assert role_call["text_format"] is RoleAttributionDecision
    assert role_call["store"] is False
    assert "How have you been feeling? This week?" in role_call["input"][1]["content"]


def test_role_attribution_can_assign_second_speaker_as_therapist(
    recordings_dir: Path,
) -> None:
    fake_openai = FakeOpenAIClient(therapist_speaker="speaker_2")
    settings = Settings(
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )

    with TestClient(create_app(settings, transcription_client=fake_openai)) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 200
    assert response.json()["turns"] == [
        {"role": "client", "text": "How have you been feeling? This week?"},
        {"role": "therapist", "text": "I've been feeling anxious."},
        {"role": "client", "text": "When did it start?"},
    ]


def test_ambiguous_role_attribution_does_not_save_completed_transcript(
    recordings_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_openai = FakeOpenAIClient(therapist_speaker="ambiguous")
    settings = Settings(
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )

    with TestClient(create_app(settings, transcription_client=fake_openai)) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 422
    assert response.json() == {"detail": "Could not determine client and therapist roles"}
    assert (recordings_dir / recording["filename"]).exists()
    assert not (recordings_dir / f"{recording['id']}.transcript.json").exists()
    assert len(fake_openai.responses.calls) == 1
    assert "role_attribution_ambiguous" in caplog.text


def test_role_attribution_failure_does_not_expose_provider_details(
    recordings_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_openai = FakeOpenAIClient(role_error=RuntimeError("sensitive role provider failure"))
    settings = Settings(
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )

    with TestClient(create_app(settings, transcription_client=fake_openai)) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 502
    assert response.json() == {"detail": "Could not assign transcript roles"}
    assert (recordings_dir / recording["filename"]).exists()
    assert not (recordings_dir / f"{recording['id']}.transcript.json").exists()
    assert "role_attribution_provider_failed" in caplog.text
    assert "provider_error_type=RuntimeError" in caplog.text
    assert "sensitive role provider failure" not in caplog.text


def test_transcription_rejects_unknown_recording(
    transcription_client: TestClient,
    fake_openai: FakeOpenAIClient,
) -> None:
    response = transcription_client.post(f"/recordings/{uuid4()}/transcription")

    assert response.status_code == 404
    assert fake_openai.transcriptions.calls == []
    assert fake_openai.responses.calls == []


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
    assert fake_openai.responses.calls == []
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
    assert fake_openai.responses.calls == []


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
    assert fake_openai.responses.calls == []

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
    fake_openai = FakeOpenAIClient(response_text="   ", segments=[])
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
    assert fake_openai.responses.calls == []


def test_transcription_rejects_response_without_speaker_segments(
    recordings_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_openai = FakeOpenAIClient(response_text="There is speech.", segments=[])
    settings = Settings(
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )

    with TestClient(create_app(settings, transcription_client=fake_openai)) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 422
    assert response.json() == {"detail": "Transcription did not include speaker labels"}
    assert not (recordings_dir / f"{recording['id']}.transcript.json").exists()
    assert "reason=missing_speaker_segments" in caplog.text
    assert fake_openai.responses.calls == []


def test_transcription_rejects_segment_without_speaker_label(
    recordings_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    segments = [
        SimpleNamespace(speaker="provider-a", text="Hello."),
        SimpleNamespace(speaker=None, text="Hi."),
    ]
    fake_openai = FakeOpenAIClient(segments=segments)
    settings = Settings(
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )

    with TestClient(create_app(settings, transcription_client=fake_openai)) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 422
    assert response.json() == {"detail": "A transcript segment did not include a speaker label"}
    assert not (recordings_dir / f"{recording['id']}.transcript.json").exists()
    assert "reason=missing_speaker_label" in caplog.text
    assert fake_openai.responses.calls == []


@pytest.mark.parametrize(
    "segments",
    [
        [SimpleNamespace(speaker="provider-a", text="Only one speaker.")],
        [
            SimpleNamespace(speaker="provider-a", text="First."),
            SimpleNamespace(speaker="provider-b", text="Second."),
            SimpleNamespace(speaker="provider-c", text="Third."),
        ],
    ],
    ids=["one-speaker", "three-speakers"],
)
def test_transcription_requires_exactly_two_speakers(
    recordings_dir: Path,
    caplog: pytest.LogCaptureFixture,
    segments: list[Any],
) -> None:
    fake_openai = FakeOpenAIClient(segments=segments)
    settings = Settings(
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
    )

    with TestClient(create_app(settings, transcription_client=fake_openai)) as client:
        recording = upload_recording(client)
        response = client.post(f"/recordings/{recording['id']}/transcription")

    assert response.status_code == 422
    assert response.json() == {"detail": "Transcription must contain exactly two speakers"}
    assert not (recordings_dir / f"{recording['id']}.transcript.json").exists()
    assert "reason=unexpected_speaker_count" in caplog.text
    assert fake_openai.responses.calls == []


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
