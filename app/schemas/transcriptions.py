from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class TranscriptionLanguage(BaseModel):
    code: str


class TranscriptTurn(BaseModel):
    role: Literal["client", "therapist"]
    text: str


class TranscriptionResponse(BaseModel):
    recording_id: UUID
    text: str
    turns: list[TranscriptTurn]
    languages: list[TranscriptionLanguage]
    model: str
    role_model: str | None
    role_prompt_version: str | None
    status: Literal["completed"]
    completed_at: datetime
