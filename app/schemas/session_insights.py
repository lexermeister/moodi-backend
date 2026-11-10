from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict

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


class ActionMention(StrictModel):
    id: str
    status: Literal["agreed", "suggested", "declined", "unresolved", "superseded"]
    eligibility: Literal["review_candidate", "clarify", "not_eligible"]
    actor: Literal["client", "therapist", "shared", "unknown"]
    activity: str
    source_excerpt: str
    frequency: str | None
    timing: str | None
    duration: str | None
    condition: str | None
    endpoint: str | None
    replaces_action_id: str | None
    source_ids: list[str]


class InsightBullet(StrictModel):
    text: str
    fact_ids: list[str]
    action_ids: list[str]
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
    action_ids: list[str]


class SessionInsightsResult(StrictModel):
    session_id: str
    source_version: str
    status: Literal["draft", "insufficient_source"]
    coverage_note: str | None
    primary_variant: InsightVariant | None
    secondary_variant: InsightVariant | None
    selection_note: str | None
    facts: list[InsightFact]
    action_mentions: list[ActionMention]
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
    action_mentions: list[ActionMention]
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
