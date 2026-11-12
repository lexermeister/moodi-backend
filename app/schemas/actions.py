from datetime import UTC, date, datetime, timedelta
from typing import Literal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, model_validator

from app.schemas.session_insights import RecordMoodConfiguration, StrictModel


class RecordMoodActionSettings(StrictModel):
    exact_task: str = Field(min_length=1, max_length=1000)
    frequency: str = Field(min_length=1, max_length=200)
    timing: str | None = Field(default=None, max_length=200)
    reference_period: str = Field(min_length=1, max_length=200)
    start_date: date | None = None
    end_date: date | None = None
    duration_kind: Literal["fixed", "ongoing"] = "fixed"
    duration_days: int | None = Field(default=14, ge=1, le=365)
    preferred_check_in_time: str | None = Field(
        default=None,
        pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$",
    )
    preferred_time_zone: str | None = Field(default=None, min_length=1, max_length=120)
    schedule_origin: Literal["moodi_default", "user_selected"] = "moodi_default"
    configuration: RecordMoodConfiguration

    @model_validator(mode="after")
    def validate_date_range(self) -> "RecordMoodActionSettings":
        if (self.preferred_check_in_time is None) != (self.preferred_time_zone is None):
            raise ValueError("Preferred check-in time and timezone must be supplied together")
        if self.preferred_time_zone is not None:
            try:
                ZoneInfo(self.preferred_time_zone)
            except ZoneInfoNotFoundError as error:
                raise ValueError("Preferred timezone must be a valid IANA timezone") from error
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("Action end date cannot precede its start date")
        if self.duration_kind == "ongoing":
            if self.duration_days is not None or self.end_date is not None:
                raise ValueError("Ongoing Actions cannot have a duration or end date")
        else:
            if self.duration_days is None:
                raise ValueError("Fixed Actions require a duration")
            if (
                self.start_date
                and self.end_date
                and (self.end_date - self.start_date).days + 1 != self.duration_days
            ):
                raise ValueError("Action date range must match its duration")
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
    disposition: Literal["activated", "linked_existing", "rejected"]
    reviewed_at: datetime
    action: MoodAction | None


class MoodCandidateResolutionResponse(StrictModel):
    recording_id: UUID
    candidate_id: str
    draft_id: str
    source_version: str
    status: Literal[
        "needs_setup",
        "already_active",
        "activated",
        "linked_existing",
        "rejected",
        "conflict",
    ]
    action: MoodAction | None
    reason: Literal["incompatible_ongoing_action"] | None = None


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
