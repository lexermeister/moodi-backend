import json
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import UUID

from openai import OpenAI

OPENAI_MAX_FILE_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".m4a", ".wav"}
EXPECTED_SPEAKER_COUNT = 2


class TranscriptionError(Exception):
    """Base error raised while transcribing or saving a transcript."""


class TranscriptionNotConfiguredError(TranscriptionError):
    """Raised when no OpenAI API key is configured."""


class UnsupportedTranscriptionFormatError(TranscriptionError):
    """Raised when OpenAI does not support the saved recording format."""


class RecordingTooLargeForTranscriptionError(TranscriptionError):
    """Raised when a recording exceeds OpenAI's file-size limit."""


class NoSpeechDetectedError(TranscriptionError):
    """Raised when the provider returns no transcribable speech."""


class DiarizationValidationError(TranscriptionError):
    """Raised when a transcript cannot be attributed to exactly two speakers."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class TranscriptionProviderError(TranscriptionError):
    """Raised when the transcription provider request fails."""

    def __init__(
        self,
        message: str,
        *,
        provider_error_type: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_error_type = provider_error_type
        self.status_code = status_code
        self.request_id = request_id


class TranscriptStorageError(TranscriptionError):
    """Raised when a completed transcript cannot be saved."""


class TranscriptionClient(Protocol):
    audio: Any


@dataclass(frozen=True)
class TranscriptTurn:
    speaker: Literal["speaker_1", "speaker_2"]
    text: str


@dataclass(frozen=True)
class CompletedTranscription:
    recording_id: UUID
    text: str
    turns: list[TranscriptTurn]
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
                    response_format="diarized_json",
                    chunking_strategy="auto",
                )
        except Exception as error:
            raise TranscriptionProviderError(
                "OpenAI transcription failed",
                provider_error_type=type(error).__name__,
                status_code=getattr(error, "status_code", None),
                request_id=getattr(error, "request_id", None),
            ) from error

        turns = self._normalize_turns(response)
        text = " ".join(turn.text for turn in turns)

        completed = CompletedTranscription(
            recording_id=recording_id,
            text=text,
            turns=turns,
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

    @classmethod
    def _normalize_turns(cls, response: Any) -> list[TranscriptTurn]:
        segments = getattr(response, "segments", None)
        if not segments:
            text = getattr(response, "text", None)
            if not isinstance(text, str) or not text.strip():
                raise NoSpeechDetectedError("No speech was detected in the recording")
            raise DiarizationValidationError(
                "Transcription did not include speaker labels",
                reason="missing_speaker_segments",
            )

        speaker_labels: dict[str, Literal["speaker_1", "speaker_2"]] = {}
        turns: list[TranscriptTurn] = []

        for segment in segments:
            speaker = cls._segment_value(segment, "speaker")
            if not isinstance(speaker, str) or not speaker.strip():
                raise DiarizationValidationError(
                    "A transcript segment did not include a speaker label",
                    reason="missing_speaker_label",
                )
            speaker = speaker.strip()

            segment_text = cls._segment_value(segment, "text")
            if not isinstance(segment_text, str) or not segment_text.strip():
                raise DiarizationValidationError(
                    "A transcript segment did not include speech",
                    reason="empty_segment",
                )

            if speaker not in speaker_labels:
                if len(speaker_labels) == EXPECTED_SPEAKER_COUNT:
                    raise DiarizationValidationError(
                        "Transcription must contain exactly two speakers",
                        reason="unexpected_speaker_count",
                    )
                normalized_speaker: Literal["speaker_1", "speaker_2"] = (
                    "speaker_1" if not speaker_labels else "speaker_2"
                )
                speaker_labels[speaker] = normalized_speaker

            normalized_speaker = speaker_labels[speaker]
            normalized_text = segment_text.strip()
            if turns and turns[-1].speaker == normalized_speaker:
                previous = turns[-1]
                turns[-1] = TranscriptTurn(
                    speaker=previous.speaker,
                    text=f"{previous.text} {normalized_text}",
                )
            else:
                turns.append(TranscriptTurn(speaker=normalized_speaker, text=normalized_text))

        if len(speaker_labels) != EXPECTED_SPEAKER_COUNT:
            raise DiarizationValidationError(
                "Transcription must contain exactly two speakers",
                reason="unexpected_speaker_count",
            )

        return turns

    @staticmethod
    def _segment_value(segment: Any, field: str) -> Any:
        if isinstance(segment, dict):
            return segment.get(field)
        return getattr(segment, field, None)

    def _save(self, transcription: CompletedTranscription) -> None:
        destination = self.directory / f"{transcription.recording_id}.transcript.json"
        temporary = self.directory / f".{transcription.recording_id}.transcript.tmp"
        payload = {
            "recording_id": str(transcription.recording_id),
            "text": transcription.text,
            "turns": [{"speaker": turn.speaker, "text": turn.text} for turn in transcription.turns],
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
