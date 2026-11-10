import json
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from app.schemas.session_insights_review import (
    ReviewedInsightItem,
    ReviewedInsightSection,
    SessionInsightsReviewRequest,
    SessionInsightsReviewResponse,
)
from app.services.session_insights import CompletedSessionInsights, SessionInsightsService


class InsightsReviewError(Exception):
    """Base error raised while loading or saving a reviewed insights layer."""


class InsightsReviewDraftNotFoundError(InsightsReviewError):
    """Raised when no generated insights draft exists for review."""


class InsightsReviewNotFoundError(InsightsReviewError):
    """Raised when no reviewed layer has been saved."""


class InsightsReviewConflictError(InsightsReviewError):
    """Raised when a review belongs to a different generated draft."""


class InsightsReviewValidationError(InsightsReviewError):
    """Raised when a review does not match its generated draft."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class InsightsReviewStorageError(InsightsReviewError):
    """Raised when a reviewed layer cannot be loaded or saved."""


class SessionInsightsReviewService:
    def __init__(
        self,
        directory: Path,
        insights_service: SessionInsightsService,
    ) -> None:
        self.directory = directory
        self.insights_service = insights_service

    def load(self, recording_id: UUID) -> SessionInsightsReviewResponse:
        draft = self._load_draft(recording_id)
        path = self._path(recording_id)
        if not path.is_file():
            raise InsightsReviewNotFoundError("No reviewed session insights were found")

        try:
            review = SessionInsightsReviewResponse.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as error:
            raise InsightsReviewStorageError("Failed to load reviewed session insights") from error

        if review.recording_id != recording_id:
            raise InsightsReviewStorageError("Reviewed insights do not match their filename")
        self._ensure_current(review.draft_id, review.source_version, draft)
        self._validate_stored_review(review, draft)
        return review

    def save(
        self,
        recording_id: UUID,
        request: SessionInsightsReviewRequest,
    ) -> SessionInsightsReviewResponse:
        draft = self._load_draft(recording_id)
        self._ensure_current(request.draft_id, request.source_version, draft)
        draft_response = draft.response()
        if draft_response.result.status != "draft":
            raise InsightsReviewValidationError(
                "There are no generated insights to review",
                reason="insufficient_source",
            )

        expected_items = {
            bullet.item_id: (section.key, section.heading, bullet.text)
            for section in draft_response.result.sections
            for bullet in section.bullets
        }
        provided_ids = [item.item_id for item in request.items]
        if len(provided_ids) != len(set(provided_ids)):
            raise InsightsReviewValidationError(
                "Review item IDs are duplicated",
                reason="duplicate_review_items",
            )
        if set(provided_ids) != set(expected_items):
            raise InsightsReviewValidationError(
                "Review items do not match the generated draft",
                reason="invalid_review_items",
            )

        requested_items = {item.item_id: item for item in request.items}
        sections: list[ReviewedInsightSection] = []
        for draft_section in draft_response.result.sections:
            reviewed_items: list[ReviewedInsightItem] = []
            for draft_bullet in draft_section.bullets:
                requested = requested_items[draft_bullet.item_id]
                displayed_text: str | None
                provenance = "source_backed"

                if requested.disposition == "retained":
                    if requested.displayed_text != draft_bullet.text:
                        raise InsightsReviewValidationError(
                            "Retained review text must match the generated draft",
                            reason="changed_retained_item",
                        )
                    displayed_text = draft_bullet.text
                elif requested.disposition == "edited":
                    displayed_text = (requested.displayed_text or "").strip()
                    if not displayed_text:
                        raise InsightsReviewValidationError(
                            "Edited review text cannot be empty",
                            reason="empty_edited_item",
                        )
                    if displayed_text == draft_bullet.text:
                        raise InsightsReviewValidationError(
                            "Unchanged review text must be retained",
                            reason="unchanged_edited_item",
                        )
                    provenance = "client_edited"
                else:
                    if requested.displayed_text is not None:
                        raise InsightsReviewValidationError(
                            "Removed review text must be null",
                            reason="text_on_removed_item",
                        )
                    displayed_text = None

                reviewed_items.append(
                    ReviewedInsightItem(
                        item_id=draft_bullet.item_id,
                        disposition=requested.disposition,
                        original_text=draft_bullet.text,
                        displayed_text=displayed_text,
                        provenance=provenance,
                    )
                )

            sections.append(
                ReviewedInsightSection(
                    key=draft_section.key,
                    heading=draft_section.heading,
                    items=reviewed_items,
                )
            )

        review = SessionInsightsReviewResponse(
            recording_id=recording_id,
            draft_id=draft.draft_id,
            source_version=draft.source_version,
            status="reviewed",
            reviewed_at=datetime.now(UTC),
            sections=sections,
        )
        self._save(review)
        return review

    def _load_draft(self, recording_id: UUID) -> CompletedSessionInsights:
        draft = self.insights_service.load_completed(recording_id)
        if draft is None:
            raise InsightsReviewDraftNotFoundError(
                "Generate session insights before reviewing them"
            )
        return draft

    @staticmethod
    def _ensure_current(
        draft_id: str,
        source_version: str,
        draft: CompletedSessionInsights,
    ) -> None:
        if draft_id != draft.draft_id or source_version != draft.source_version:
            raise InsightsReviewConflictError(
                "Session insights changed; review the latest draft before saving"
            )

    @staticmethod
    def _validate_stored_review(
        review: SessionInsightsReviewResponse,
        draft: CompletedSessionInsights,
    ) -> None:
        draft_response = draft.response()
        expected_sections = {
            section.key: (
                section.heading,
                {bullet.item_id: bullet.text for bullet in section.bullets},
            )
            for section in draft_response.result.sections
        }
        if len(review.sections) != len(expected_sections):
            raise InsightsReviewStorageError("Reviewed insights sections are invalid")

        seen_items: set[str] = set()
        for section in review.sections:
            expected = expected_sections.get(section.key)
            if expected is None or section.heading != expected[0]:
                raise InsightsReviewStorageError("Reviewed insights sections are invalid")
            expected_items = expected[1]
            if {item.item_id for item in section.items} != set(expected_items):
                raise InsightsReviewStorageError("Reviewed insights items are invalid")
            for item in section.items:
                if item.item_id in seen_items or item.original_text != expected_items[item.item_id]:
                    raise InsightsReviewStorageError("Reviewed insights items are invalid")
                if item.disposition == "retained" and (
                    item.displayed_text != item.original_text or item.provenance != "source_backed"
                ):
                    raise InsightsReviewStorageError("Reviewed insights items are invalid")
                if item.disposition == "edited" and (
                    not (item.displayed_text or "").strip()
                    or item.displayed_text == item.original_text
                    or item.provenance != "client_edited"
                ):
                    raise InsightsReviewStorageError("Reviewed insights items are invalid")
                if item.disposition == "removed" and (
                    item.displayed_text is not None or item.provenance != "source_backed"
                ):
                    raise InsightsReviewStorageError("Reviewed insights items are invalid")
                seen_items.add(item.item_id)

    def _path(self, recording_id: UUID) -> Path:
        return self.directory / f"{recording_id}.insights-review.json"

    def _save(self, review: SessionInsightsReviewResponse) -> None:
        destination = self._path(review.recording_id)
        temporary = self.directory / f".{review.recording_id}.insights-review.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as output:
                json.dump(review.model_dump(mode="json"), output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
        except Exception as error:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise InsightsReviewStorageError("Failed to save reviewed session insights") from error
