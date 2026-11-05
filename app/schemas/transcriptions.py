from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class TranscriptionLanguage(BaseModel):
    code: str


class TranscriptionResponse(BaseModel):
    transcription_id: UUID
    recording_id: UUID
    text: str
    languages: list[TranscriptionLanguage]
    model: str
    status: Literal["completed"]
    completed_at: datetime
