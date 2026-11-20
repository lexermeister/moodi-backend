from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from app.schemas.session_insights import StrictModel

type ActivitySelectionOrigin = Literal["agreed", "session_mention", "predefined", "own_idea"]
type ActivityOutcome = Literal["completed", "partly_completed", "skipped", "stopped_early"]


class ActivitySuggestion(StrictModel):
    id: str
    label: str
    source: Literal["moodi_prototype_v1"]


class ActivitySuggestionList(StrictModel):
    suggestions: list[ActivitySuggestion]


class ActivitySelection(StrictModel):
    origin: ActivitySelectionOrigin
    label: str = Field(min_length=1, max_length=200)
    suggestion_id: str | None = Field(default=None, max_length=100)
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("label")
    @classmethod
    def nonblank_label(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("An activity label is required")
        return stripped


class ActivityAttemptData(StrictModel):
    status: Literal["draft", "confirmed"]
    selection: ActivitySelection
    planned_minutes: int | None = Field(default=None, ge=1, le=1440)
    outcome: ActivityOutcome | None = None
    actual_minutes: int | None = Field(default=None, ge=1, le=1440)
    how_it_went: str | None = Field(default=None, max_length=2000)
    feelings_after: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_confirmation(self) -> "ActivityAttemptData":
        if self.status == "confirmed" and self.outcome is None:
            raise ValueError("A confirmed activity attempt needs an outcome")
        return self


class SaveActivityAttemptRequest(ActivityAttemptData):
    action_revision: int = Field(ge=1)
    expected_revision: int = Field(ge=0)


class ActivityAttemptVersion(ActivityAttemptData):
    revision: int = Field(ge=1)
    saved_at: datetime


class ActivityAttempt(ActivityAttemptData):
    id: UUID
    action_id: UUID
    action_revision: int = Field(ge=1)
    revision: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime
    history: list[ActivityAttemptVersion] = Field(default_factory=list)


class ActivityAttemptList(StrictModel):
    attempts: list[ActivityAttempt]
