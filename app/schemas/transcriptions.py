from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class TranscriptionLanguage(BaseModel):
    code: str


class TranscriptTurn(BaseModel):
    speaker: Literal["speaker_1", "speaker_2"]
    text: str


class TranscriptionResponse(BaseModel):
    recording_id: UUID
    text: str
    turns: list[TranscriptTurn]
    languages: list[TranscriptionLanguage]
    model: str
    status: Literal["completed"]
    completed_at: datetime
