from datetime import UTC, date, datetime, timedelta
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator

from app.schemas.session_insights import RecordMoodConfiguration, StrictModel


class RecordMoodActionSettings(StrictModel):
    exact_task: str = Field(min_length=1, max_length=1000)
    frequency: str = Field(min_length=1, max_length=200)
    timing: str | None = Field(default=None, max_length=200)
    reference_period: str = Field(min_length=1, max_length=200)
    start_date: date | None = None
    end_date: date | None = None
    duration_days: Literal[14] = 14
    configuration: RecordMoodConfiguration

    @model_validator(mode="after")
    def validate_date_range(self) -> "RecordMoodActionSettings":
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("Action end date cannot precede its start date")
        if (
            self.start_date
            and self.end_date
            and (self.end_date - self.start_date).days + 1 != self.duration_days
        ):
            raise ValueError("Action date range must contain exactly 14 days")
        return self


class MoodAction(StrictModel):
    id: UUID
    type: Literal["record_mood"]
    state: Literal["active", "paused", "superseded"]
    recording_id: UUID
    transcript_version: str
    candidate_id: str
    source_ids: list[str]
    source_excerpt: str
    settings: RecordMoodActionSettings
    created_at: datetime
    revision: int = Field(ge=1)


class MoodCandidateReviewRequest(StrictModel):
    draft_id: str
    source_version: str
    disposition: Literal["activate", "reject"]
    settings: RecordMoodActionSettings | None = None

    @model_validator(mode="after")
    def validate_disposition(self) -> "MoodCandidateReviewRequest":
        if self.disposition == "activate" and self.settings is None:
            raise ValueError("Activation requires reviewed settings")
        if self.disposition == "reject" and self.settings is not None:
            raise ValueError("Rejected candidates cannot include settings")
        return self


class MoodCandidateReviewResponse(StrictModel):
    recording_id: UUID
    candidate_id: str
    draft_id: str
    source_version: str
    disposition: Literal["activated", "rejected"]
    reviewed_at: datetime
    action: MoodAction | None


class MoodEntryValues(StrictModel):
    relevant_date: date
    entry_time: datetime
    timezone_offset_minutes: int = Field(ge=-840, le=840)
    day_key: date
    mood_value: int | None = Field(default=None, ge=0, le=10)
    mood_label: str | None = Field(default=None, min_length=1, max_length=120)
    context: str | None = Field(default=None, min_length=1, max_length=2000)
    scale_id: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_captured_date(self) -> "MoodEntryValues":
        if self.entry_time.tzinfo is None or self.entry_time.utcoffset() is None:
            raise ValueError("Entry time must include a timezone offset")
        captured = self.entry_time.astimezone(UTC) + timedelta(minutes=self.timezone_offset_minutes)
        if captured.date() != self.day_key or self.relevant_date != self.day_key:
            raise ValueError("Day key must match entry time, captured offset, and relevant date")
        if self.mood_value is not None and self.mood_label is not None:
            raise ValueError("An entry cannot contain both a mood rating and mood label")
        if self.mood_value is None and self.scale_id is not None:
            raise ValueError("Scale metadata requires a numeric mood rating")
        if self.mood_value is not None and self.scale_id is None:
            raise ValueError("Numeric mood rating requires scale metadata")
        return self


class SaveMoodEntryDraftRequest(StrictModel):
    values: MoodEntryValues
    expected_revision: int | None = Field(default=None, ge=1)


class ConfirmMoodEntryRequest(StrictModel):
    values: MoodEntryValues
    expected_revision: int | None = Field(default=None, ge=1)
    idempotency_key: str = Field(min_length=1, max_length=120)


class AmendMoodEntryRequest(StrictModel):
    values: MoodEntryValues
    expected_revision: int = Field(ge=1)


class MoodEntryRevision(StrictModel):
    revision: int = Field(ge=1)
    values: MoodEntryValues
    status: Literal["draft", "confirmed"]
    changed_at: datetime


class MoodEntry(StrictModel):
    id: UUID
    action_id: UUID
    status: Literal["draft", "confirmed"]
    values: MoodEntryValues
    revision: int = Field(ge=1)
    revision_history: list[MoodEntryRevision]
    idempotency_key: str | None
    created_at: datetime
    updated_at: datetime
    confirmed_at: datetime | None


class MoodActionListResponse(StrictModel):
    actions: list[MoodAction]


class MoodEntryListResponse(StrictModel):
    entries: list[MoodEntry]


class MoodEntryExportRequest(StrictModel):
    entry_ids: list[UUID] = Field(min_length=1, max_length=500)


class MoodEntryExportResponse(StrictModel):
    action: MoodAction
    entries: list[MoodEntry]
    exported_at: datetime
