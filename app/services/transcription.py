from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID, uuid4

from openai import OpenAI
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.models import Transcription

OPENAI_MAX_FILE_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".m4a", ".wav"}
TRANSCRIPTION_PROMPT = (
    "A single-speaker personal reflection or therapy journal. "
    "Preserve the original language and transcribe only what is spoken."
)


class TranscriptionError(Exception):
    """Base error raised while transcribing or persisting a transcript."""


class TranscriptionNotConfiguredError(TranscriptionError):
    """Raised when no OpenAI API key is configured."""


class UnsupportedTranscriptionFormatError(TranscriptionError):
    """Raised when OpenAI does not support the saved recording format."""


class RecordingTooLargeForTranscriptionError(TranscriptionError):
    """Raised when a recording exceeds OpenAI's file-size limit."""


class NoSpeechDetectedError(TranscriptionError):
    """Raised when the provider returns no transcribable speech."""

    def __init__(
        self,
        message: str,
        *,
        transcription_id: UUID,
        provider_request_id: str | None,
    ) -> None:
        super().__init__(message)
        self.transcription_id = transcription_id
        self.provider_request_id = provider_request_id


class TranscriptionProviderError(TranscriptionError):
    """Raised when the transcription provider request fails."""

    def __init__(
        self,
        message: str,
        *,
        transcription_id: UUID,
        provider_error_type: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.transcription_id = transcription_id
        self.provider_error_type = provider_error_type
        self.status_code = status_code
        self.request_id = request_id


class TranscriptionPersistenceError(TranscriptionError):
    """Raised when a transcription attempt cannot be persisted."""

    def __init__(self, message: str, *, transcription_id: UUID) -> None:
        super().__init__(message)
        self.transcription_id = transcription_id


class TranscriptionClient(Protocol):
    audio: Any


@dataclass(frozen=True)
class CompletedTranscription:
    transcription_id: UUID
    recording_id: UUID
    text: str
    languages: list[str]
    model: str
    provider_request_id: str | None
    completed_at: datetime


class TranscriptionService:
    def __init__(
        self,
        api_key: str | None,
        model: str,
        client: TranscriptionClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.client = client

    def transcribe(
        self,
        recording_id: UUID,
        recording_path: Path,
        session: Session,
    ) -> CompletedTranscription:
        self._validate_recording(recording_path)
        client = self._get_client()
        attempt = self._create_attempt(recording_id, session)

        try:
            with recording_path.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    prompt=TRANSCRIPTION_PROMPT,
                    extra_body={"languages": ["sr", "en"]},
                    extra_headers={"X-Client-Request-Id": str(attempt.id)},
                )
        except Exception as error:
            provider_request_id = self._request_id(error)
            self._fail_attempt(
                attempt,
                session,
                error_code="provider_error",
                provider_request_id=provider_request_id,
            )
            raise TranscriptionProviderError(
                "OpenAI transcription failed",
                transcription_id=attempt.id,
                provider_error_type=type(error).__name__,
                status_code=getattr(error, "status_code", None),
                request_id=provider_request_id,
            ) from error

        provider_request_id = self._request_id(response)
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            self._fail_attempt(
                attempt,
                session,
                error_code="no_speech",
                provider_request_id=provider_request_id,
            )
            raise NoSpeechDetectedError(
                "No speech was detected in the recording",
                transcription_id=attempt.id,
                provider_request_id=provider_request_id,
            )

        languages = self._language_codes(getattr(response, "languages", []))
        completed_at = datetime.now(UTC)
        attempt.status = "completed"
        attempt.provider_request_id = provider_request_id
        attempt.text = text
        attempt.languages = languages
        attempt.completed_at = completed_at
        self._commit_attempt(attempt, session)

        return CompletedTranscription(
            transcription_id=attempt.id,
            recording_id=recording_id,
            text=text,
            languages=languages,
            model=self.model,
            provider_request_id=provider_request_id,
            completed_at=completed_at,
        )

    def _get_client(self) -> TranscriptionClient:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise TranscriptionNotConfiguredError("OpenAI transcription is not configured")

        self.client = OpenAI(api_key=self.api_key)
        return self.client

    def _create_attempt(self, recording_id: UUID, session: Session) -> Transcription:
        attempt = Transcription(
            id=uuid4(),
            recording_id=recording_id,
            status="started",
            provider="openai",
            model=self.model,
            languages=[],
        )
        session.add(attempt)
        self._commit_attempt(attempt, session)
        return attempt

    @staticmethod
    def _fail_attempt(
        attempt: Transcription,
        session: Session,
        *,
        error_code: str,
        provider_request_id: str | None,
    ) -> None:
        attempt.status = "failed"
        attempt.provider_request_id = provider_request_id
        attempt.error_code = error_code
        attempt.completed_at = datetime.now(UTC)
        TranscriptionService._commit_attempt(attempt, session)

    @staticmethod
    def _commit_attempt(attempt: Transcription, session: Session) -> None:
        transcription_id = attempt.id
        try:
            session.commit()
        except SQLAlchemyError as error:
            session.rollback()
            raise TranscriptionPersistenceError(
                "Failed to save transcription state",
                transcription_id=transcription_id,
            ) from error

    @staticmethod
    def _validate_recording(recording_path: Path) -> None:
        if recording_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise UnsupportedTranscriptionFormatError(
                "Only M4A and WAV recordings can be transcribed"
            )
        if recording_path.stat().st_size > OPENAI_MAX_FILE_BYTES:
            raise RecordingTooLargeForTranscriptionError(
                "Recording exceeds OpenAI's 25 MB transcription limit"
            )

    @staticmethod
    def _language_codes(languages: Any) -> list[str]:
        codes: list[str] = []
        for language in languages or []:
            code = (
                language.get("code")
                if isinstance(language, dict)
                else getattr(language, "code", None)
            )
            if isinstance(code, str):
                codes.append(code)
        return codes

    @staticmethod
    def _request_id(value: Any) -> str | None:
        request_id = getattr(value, "_request_id", None) or getattr(value, "request_id", None)
        return request_id if isinstance(request_id, str) else None
