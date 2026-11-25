from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.session_insights import StrictModel


class PracticeAttemptData(StrictModel):
    status: Literal["draft", "confirmed"]
    relevant_date: date
    outcome: Literal["completed", "partly_completed", "skipped", "stopped_early"] | None = None
    actual_minutes: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    tension_before: int | None = Field(default=None, ge=0, le=10, strict=True)
    tension_after: int | None = Field(default=None, ge=0, le=10, strict=True)
    before_rating_retrospective: bool = True
    notes: str | None = Field(default=None, max_length=2000)
    timer_elapsed_seconds: int = Field(default=0, ge=0)
    timer_used: bool = False
    timer_pause_count: int = Field(default=0, ge=0)
    current_step: Literal["choice", "timer", "outcome", "tension", "notes", "review"] = "choice"

    @model_validator(mode="after")
    def validate_confirmation(self) -> "PracticeAttemptData":
        if self.status == "confirmed" and self.outcome is None:
            raise ValueError("Choose an outcome before saving")
        return self


class SavePracticeAttemptRequest(PracticeAttemptData):
    action_revision: int = Field(ge=1)
    expected_revision: int = Field(ge=0)


class PracticeAttemptVersion(PracticeAttemptData):
    revision: int = Field(ge=1)
    saved_at: datetime


class PracticeAttempt(PracticeAttemptData):
    id: UUID
    action_id: UUID
    action_revision: int = Field(ge=1)
    revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    history: list[PracticeAttemptVersion] = Field(default_factory=list)


class PracticeAttemptList(StrictModel):
    attempts: list[PracticeAttempt]
