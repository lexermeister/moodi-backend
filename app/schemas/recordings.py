from typing import Literal
from uuid import UUID

from pydantic import BaseModel


class RecordingResponse(BaseModel):
    id: UUID
    filename: str
    size_bytes: int
    status: Literal["saved"]
