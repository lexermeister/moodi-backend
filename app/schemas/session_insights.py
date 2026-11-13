from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.services.rating_metrics import RatingConstruct, RatingMetric, get_rating_metric

type InsightVariant = Literal["P01", "P02", "P03"]
type InsightSectionKey = Literal[
    "today_focus",
    "key_takeaways",
    "explored_practised",
    "agreements",
    "additional_topics",
    "next_time",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionInsightsRequest(StrictModel):
    session_date: date | None = None


class InsightFact(StrictModel):
    id: str
    kind: Literal[
        "topic",
        "client_report",
        "therapist_statement",
        "practice",
        "response",
        "goal",
        "decision",
        "open_question",
        "prior_record",
    ]
    text: str
    source_ids: list[str]


class RatingScaleDefinition(StrictModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    id: str = Field(min_length=1, max_length=120)
    construct_name: RatingConstruct = Field(alias="construct")
    minimum: Literal[0]
    maximum: Literal[10]
    minimum_label: str = Field(min_length=1, max_length=80)
    maximum_label: str = Field(min_length=1, max_length=80)
    direction: Literal["low_to_high"]
    version: str = Field(min_length=1, max_length=40)


class RecordRatingConfiguration(StrictModel):
    input_kind: Literal["numeric_0_10", "label"]
    context_enabled: bool = False
    scale: RatingScaleDefinition | None = None
    scale_origin: Literal["session", "moodi_default"] | None = None

    @model_validator(mode="after")
    def validate_scale(self) -> "RecordRatingConfiguration":
        if self.input_kind == "numeric_0_10" and self.scale is None:
            raise ValueError("Numeric rating configuration requires scale metadata")
        if self.input_kind == "numeric_0_10" and self.scale_origin is None:
            self.scale_origin = "session"
        if self.input_kind == "label" and self.scale is not None:
            raise ValueError("Label rating configuration cannot include a numeric scale")
        if self.input_kind == "label" and self.scale_origin is not None:
            raise ValueError("Label rating configuration cannot include a scale origin")
        return self


class RecordRatingCandidate(StrictModel):
    id: str
    type: Literal["record_rating"]
    state: Literal["review_ready"]
    actor: Literal["client"]
    metric: RatingMetric
    exact_task: str
    source_excerpt: str
    frequency: str | None
    timing: str | None
    reference_period: str | None
    start_date: date | None
    end_date: date | None
    duration_days: Literal[14] = 14
    configuration: RecordRatingConfiguration | None
    unknowns: list[str]
    source_ids: list[str]

    @model_validator(mode="after")
    def validate_review_state(self) -> "RecordRatingCandidate":
        if self.state == "review_ready" and (
            self.configuration is None
            or not self.frequency
            or not self.reference_period
            or self.unknowns
        ):
            raise ValueError("Review-ready rating candidate is incomplete")
        if self.configuration is not None and self.configuration.scale is not None:
            definition = get_rating_metric(self.metric)
            if (
                self.configuration.scale.construct_name != definition.construct
                or self.configuration.scale.id != definition.scale_id
            ):
                raise ValueError("Rating candidate scale does not match its metric")
        return self


class InsightBullet(StrictModel):
    text: str
    fact_ids: list[str]
    candidate_ids: list[str]
    module: (
        Literal[
            "practice_details",
            "progress_and_barriers",
            "ideas_not_agreed",
            "details_to_clarify",
            "coverage",
        ]
        | None
    )


class InsightSection(StrictModel):
    key: InsightSectionKey
    heading: str
    bullets: list[InsightBullet]


class InsightReviewFlag(StrictModel):
    code: Literal[
        "source_gap",
        "unclear_speaker",
        "ambiguous_agreement",
        "missing_action_detail",
        "conflicting_source",
        "unsupported_session_type",
        "safety_sensitive_content",
    ]
    detail: str
    source_ids: list[str]
    candidate_ids: list[str]


class SessionInsightsResult(StrictModel):
    session_id: str
    source_version: str
    status: Literal["draft", "insufficient_source"]
    coverage_note: str | None
    primary_variant: InsightVariant | None
    secondary_variant: InsightVariant | None
    selection_note: str | None
    facts: list[InsightFact]
    action_candidates: list[RecordRatingCandidate]
    sections: list[InsightSection]
    review_flags: list[InsightReviewFlag]


class InsightDraftBullet(InsightBullet):
    item_id: str


class InsightDraftSection(StrictModel):
    key: InsightSectionKey
    heading: str
    bullets: list[InsightDraftBullet]


class SessionInsightsDraftResult(StrictModel):
    session_id: str
    source_version: str
    status: Literal["draft", "insufficient_source"]
    coverage_note: str | None
    primary_variant: InsightVariant | None
    secondary_variant: InsightVariant | None
    selection_note: str | None
    facts: list[InsightFact]
    action_candidates: list[RecordRatingCandidate]
    sections: list[InsightDraftSection]
    review_flags: list[InsightReviewFlag]


class SessionInsightsResponse(StrictModel):
    recording_id: UUID
    draft_id: str
    source_version: str
    prompt_version: str
    model: str
    generated_at: datetime
    result: SessionInsightsDraftResult
