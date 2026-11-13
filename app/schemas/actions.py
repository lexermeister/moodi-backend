from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, model_validator

from app.schemas.session_insights import RecordRatingConfiguration, StrictModel
from app.services.rating_metrics import RatingMetric, get_rating_metric


class RecordRatingActionSettings(StrictModel):
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
    configuration: RecordRatingConfiguration

    @model_validator(mode="after")
    def validate_date_range(self) -> "RecordRatingActionSettings":
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


class RatingAction(StrictModel):
    id: UUID
    type: Literal["record_rating"]
    metric: RatingMetric
    state: Literal["active", "paused", "superseded"]
    recording_id: UUID
    transcript_version: str
    candidate_id: str
    source_ids: list[str]
    source_excerpt: str
    settings: RecordRatingActionSettings
    created_at: datetime
    revision: int = Field(ge=1)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_mood_action(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if migrated.get("type") == "record_mood":
            migrated["type"] = "record_rating"
        migrated.setdefault("metric", "mood")
        return migrated

    @model_validator(mode="after")
    def validate_metric_configuration(self) -> "RatingAction":
        scale = self.settings.configuration.scale
        if scale is not None:
            definition = get_rating_metric(self.metric)
            if scale.construct_name != definition.construct or scale.id != definition.scale_id:
                raise ValueError("Rating Action scale does not match its metric")
        return self


class RatingCandidateReviewRequest(StrictModel):
    draft_id: str
    source_version: str
    disposition: Literal["activate", "reject"]
    settings: RecordRatingActionSettings | None = None

    @model_validator(mode="after")
    def validate_disposition(self) -> "RatingCandidateReviewRequest":
        if self.disposition == "activate" and self.settings is None:
            raise ValueError("Activation requires reviewed settings")
        if self.disposition == "reject" and self.settings is not None:
            raise ValueError("Rejected candidates cannot include settings")
        return self


class RatingCandidateReviewResponse(StrictModel):
    recording_id: UUID
    candidate_id: str
    draft_id: str
    source_version: str
    disposition: Literal["activated", "linked_existing", "rejected"]
    reviewed_at: datetime
    action: RatingAction | None


class RatingCandidateResolutionResponse(StrictModel):
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
    action: RatingAction | None
    reason: Literal["incompatible_ongoing_action"] | None = None


class RatingEntryValues(StrictModel):
    relevant_date: date
    entry_time: datetime
    timezone_offset_minutes: int = Field(ge=-840, le=840)
    day_key: date
    rating_value: int | None = Field(default=None, ge=0, le=10)
    rating_label: str | None = Field(default=None, min_length=1, max_length=120)
    context: str | None = Field(default=None, min_length=1, max_length=2000)
    scale_id: str | None = Field(default=None, min_length=1, max_length=120)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_mood_values(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        migrated = dict(value)
        if "mood_value" in migrated and "rating_value" not in migrated:
            migrated["rating_value"] = migrated.pop("mood_value")
        if "mood_label" in migrated and "rating_label" not in migrated:
            migrated["rating_label"] = migrated.pop("mood_label")
        return migrated

    @model_validator(mode="after")
    def validate_captured_date(self) -> "RatingEntryValues":
        if self.entry_time.tzinfo is None or self.entry_time.utcoffset() is None:
            raise ValueError("Entry time must include a timezone offset")
        captured = self.entry_time.astimezone(UTC) + timedelta(minutes=self.timezone_offset_minutes)
        if captured.date() != self.day_key or self.relevant_date != self.day_key:
            raise ValueError("Day key must match entry time, captured offset, and relevant date")
        if self.rating_value is not None and self.rating_label is not None:
            raise ValueError("An entry cannot contain both a numeric rating and rating label")
        if self.rating_value is None and self.scale_id is not None:
            raise ValueError("Scale metadata requires a numeric rating")
        if self.rating_value is not None and self.scale_id is None:
            raise ValueError("Numeric rating requires scale metadata")
        return self


class SaveRatingEntryDraftRequest(StrictModel):
    values: RatingEntryValues
    expected_revision: int | None = Field(default=None, ge=1)


class ConfirmRatingEntryRequest(StrictModel):
    values: RatingEntryValues
    expected_revision: int | None = Field(default=None, ge=1)
    idempotency_key: str = Field(min_length=1, max_length=120)


class AmendRatingEntryRequest(StrictModel):
    values: RatingEntryValues
    expected_revision: int = Field(ge=1)


class RatingEntryRevision(StrictModel):
    revision: int = Field(ge=1)
    values: RatingEntryValues
    status: Literal["draft", "confirmed"]
    changed_at: datetime


class RatingEntry(StrictModel):
    id: UUID
    action_id: UUID
    status: Literal["draft", "confirmed"]
    values: RatingEntryValues
    revision: int = Field(ge=1)
    revision_history: list[RatingEntryRevision]
    idempotency_key: str | None
    created_at: datetime
    updated_at: datetime
    confirmed_at: datetime | None


class RatingActionListResponse(StrictModel):
    actions: list[RatingAction]


class RatingEntryListResponse(StrictModel):
    entries: list[RatingEntry]


class RatingEntryExportRequest(StrictModel):
    entry_ids: list[UUID] = Field(min_length=1, max_length=500)


class RatingEntryExportResponse(StrictModel):
    action: RatingAction
    entries: list[RatingEntry]
    exported_at: datetime
