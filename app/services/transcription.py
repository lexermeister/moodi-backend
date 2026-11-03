import json
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from openai import OpenAI

OPENAI_MAX_FILE_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".m4a", ".wav"}
TRANSCRIPTION_PROMPT = (
    "A single-speaker personal reflection or therapy journal. "
    "Preserve the original language and transcribe only what is spoken."
)


class TranscriptionError(Exception):
    """Base error raised while transcribing or saving a transcript."""


class TranscriptionNotConfiguredError(TranscriptionError):
    """Raised when no OpenAI API key is configured."""


class UnsupportedTranscriptionFormatError(TranscriptionError):
    """Raised when OpenAI does not support the saved recording format."""


class RecordingTooLargeForTranscriptionError(TranscriptionError):
    """Raised when a recording exceeds OpenAI's file-size limit."""


class TranscriptionProviderError(TranscriptionError):
    """Raised when the transcription provider request fails."""


class TranscriptStorageError(TranscriptionError):
    """Raised when a completed transcript cannot be saved."""


class TranscriptionClient(Protocol):
    audio: Any


@dataclass(frozen=True)
class CompletedTranscription:
    recording_id: UUID
    text: str
    languages: list[str]
    model: str
    completed_at: datetime


class TranscriptionService:
    def __init__(
        self,
        directory: Path,
        api_key: str | None,
        model: str,
        client: TranscriptionClient | None = None,
    ) -> None:
        self.directory = directory
        self.api_key = api_key
        self.model = model
        self.client = client

    def transcribe(self, recording_id: UUID, recording_path: Path) -> CompletedTranscription:
        self._validate_recording(recording_path)
        client = self._get_client()

        try:
            with recording_path.open("rb") as audio_file:
                response = client.audio.transcriptions.create(
                    model=self.model,
                    file=audio_file,
                    prompt=TRANSCRIPTION_PROMPT,
                    extra_body={"languages": ["sr", "en"]},
                )
        except Exception as error:
            raise TranscriptionProviderError("OpenAI transcription failed") from error

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise TranscriptionProviderError("OpenAI returned an empty transcription")

        completed = CompletedTranscription(
            recording_id=recording_id,
            text=text,
            languages=self._language_codes(getattr(response, "languages", [])),
            model=self.model,
            completed_at=datetime.now(UTC),
        )
        self._save(completed)
        return completed

    def _get_client(self) -> TranscriptionClient:
        if self.client is not None:
            return self.client
        if not self.api_key:
            raise TranscriptionNotConfiguredError("OpenAI transcription is not configured")

        self.client = OpenAI(api_key=self.api_key)
        return self.client

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

    def _save(self, transcription: CompletedTranscription) -> None:
        destination = self.directory / f"{transcription.recording_id}.transcript.json"
        temporary = self.directory / f".{transcription.recording_id}.transcript.tmp"
        payload = {
            "recording_id": str(transcription.recording_id),
            "text": transcription.text,
            "languages": [{"code": code} for code in transcription.languages],
            "model": transcription.model,
            "status": "completed",
            "completed_at": transcription.completed_at.isoformat().replace("+00:00", "Z"),
        }

        try:
            with temporary.open("x", encoding="utf-8") as output:
                json.dump(payload, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
        except Exception as error:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise TranscriptStorageError("Failed to save transcript") from error
