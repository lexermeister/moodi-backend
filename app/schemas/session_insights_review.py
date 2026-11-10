from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field

from app.schemas.session_insights import InsightSectionKey, StrictModel

type ReviewDisposition = Literal["retained", "edited", "removed"]
type ReviewProvenance = Literal["source_backed", "client_edited"]


class InsightReviewItemRequest(StrictModel):
    item_id: str
    disposition: ReviewDisposition
    displayed_text: str | None = Field(max_length=4000)


class SessionInsightsReviewRequest(StrictModel):
    draft_id: str
    source_version: str
    items: list[InsightReviewItemRequest]


class ReviewedInsightItem(StrictModel):
    item_id: str
    disposition: ReviewDisposition
    original_text: str
    displayed_text: str | None
    provenance: ReviewProvenance


class ReviewedInsightSection(StrictModel):
    key: InsightSectionKey
    heading: str
    items: list[ReviewedInsightItem]


class SessionInsightsReviewResponse(StrictModel):
    recording_id: UUID
    draft_id: str
    source_version: str
    status: Literal["reviewed"]
    reviewed_at: datetime
    sections: list[ReviewedInsightSection]
