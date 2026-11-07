import json
import os
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

type SourceSpeaker = Literal["speaker_1", "speaker_2"]
type TranscriptRole = Literal["client", "therapist"]


class TranscriptStorageError(Exception):
    """Raised when a completed transcript cannot be loaded or saved."""


@dataclass(frozen=True)
class TranscriptTurn:
    source_speaker: SourceSpeaker
    role: TranscriptRole
    text: str


@dataclass(frozen=True)
class CompletedTranscription:
    recording_id: UUID
    text: str
    turns: list[TranscriptTurn]
    languages: list[str]
    model: str
    role_model: str
    role_prompt_version: str
    completed_at: datetime


class TranscriptStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def load(self, recording_id: UUID) -> CompletedTranscription | None:
        path = self.directory / f"{recording_id}.transcript.json"
        if not path.is_file():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("recording_id") != str(recording_id):
                raise ValueError("Transcript recording ID does not match its filename")

            turns = [
                TranscriptTurn(
                    source_speaker=turn["speaker"],
                    role=turn["role"],
                    text=turn["text"],
                )
                for turn in payload["turns"]
            ]
            if any(
                turn.source_speaker not in ("speaker_1", "speaker_2")
                or turn.role not in ("client", "therapist")
                or not turn.text.strip()
                for turn in turns
            ):
                raise ValueError("Transcript contains an invalid turn")

            completed_at = datetime.fromisoformat(payload["completed_at"].replace("Z", "+00:00"))
            if completed_at.tzinfo is None:
                completed_at = completed_at.replace(tzinfo=UTC)

            return CompletedTranscription(
                recording_id=recording_id,
                text=payload["text"],
                turns=turns,
                languages=[language["code"] for language in payload["languages"]],
                model=payload["model"],
                role_model=payload["role_model"],
                role_prompt_version=payload["role_prompt_version"],
                completed_at=completed_at,
            )
        except Exception as error:
            raise TranscriptStorageError("Failed to load transcript") from error

    def save(self, transcription: CompletedTranscription) -> None:
        destination = self.directory / f"{transcription.recording_id}.transcript.json"
        temporary = self.directory / f".{transcription.recording_id}.transcript.tmp"
        payload = {
            "recording_id": str(transcription.recording_id),
            "text": transcription.text,
            "turns": [
                {
                    "speaker": turn.source_speaker,
                    "role": turn.role,
                    "text": turn.text,
                }
                for turn in transcription.turns
            ],
            "languages": [{"code": code} for code in transcription.languages],
            "model": transcription.model,
            "role_model": transcription.role_model,
            "role_prompt_version": transcription.role_prompt_version,
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
