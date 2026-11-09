import hashlib
import json
import os
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import monotonic
from typing import Protocol
from uuid import UUID

from app.core.logging import get_logger
from app.schemas.session_insights import SessionInsightsResponse, SessionInsightsResult
from app.services.transcript_store import TranscriptStore

SESSION_INSIGHTS_PROMPT_VERSION = "v0.3"
MAX_GENERATION_ATTEMPTS = 2
logger = get_logger("services.session_insights")


class SessionInsightsError(Exception):
    """Base error raised while generating or storing session insights."""


class TranscriptNotFoundForInsightsError(SessionInsightsError):
    """Raised when insights are requested before transcription is complete."""


class InsightsNotConfiguredError(SessionInsightsError):
    """Raised when no OpenAI API key is configured."""


class InsightsProviderError(SessionInsightsError):
    """Raised when the insights provider request fails."""

    def __init__(
        self,
        message: str,
        *,
        provider_error_type: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider_error_type = provider_error_type
        self.status_code = status_code
        self.request_id = request_id


class InsightsValidationError(SessionInsightsError):
    """Raised when a structured result is not supported by its transcript evidence."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class InsightsStorageError(SessionInsightsError):
    """Raised when completed insights cannot be loaded or saved."""


class InsightsGenerator(Protocol):
    model: str

    def generate(
        self,
        recording_id: UUID,
        prompt: str,
        input_data: dict[str, object],
    ) -> SessionInsightsResult: ...


@dataclass(frozen=True)
class CompletedSessionInsights:
    recording_id: UUID
    source_version: str
    prompt_version: str
    model: str
    request_fingerprint: str
    generated_at: datetime
    result: SessionInsightsResult

    def response(self) -> SessionInsightsResponse:
        return SessionInsightsResponse(
            recording_id=self.recording_id,
            source_version=self.source_version,
            prompt_version=self.prompt_version,
            model=self.model,
            generated_at=self.generated_at,
            result=self.result,
        )


class SessionInsightsService:
    def __init__(
        self,
        directory: Path,
        transcript_store: TranscriptStore,
        generator: InsightsGenerator,
        prompt_path: Path,
    ) -> None:
        self.directory = directory
        self.transcript_store = transcript_store
        self.generator = generator
        self.prompt = prompt_path.read_text(encoding="utf-8")

    def generate(
        self,
        recording_id: UUID,
        session_date: date | None,
    ) -> CompletedSessionInsights:
        transcript = self.transcript_store.load(recording_id)
        if transcript is None:
            raise TranscriptNotFoundForInsightsError(
                "Generate a transcript before requesting session insights"
            )

        segments = [
            {
                "id": f"S{index}",
                "origin": "transcript",
                "role": turn.role,
                "text": turn.text,
            }
            for index, turn in enumerate(transcript.turns, start=1)
        ]
        source_version = self._source_version(segments)
        input_data: dict[str, object] = {
            "session_id": str(recording_id),
            "source_version": source_version,
            "session_date": session_date.isoformat() if session_date else None,
            "coverage": "complete_for_input",
            "preferred_variant": None,
            "output_language": "en",
            "role_attribution": "model_inferred",
            "segments": segments,
        }
        request_fingerprint = self._request_fingerprint(input_data)

        cached = self._load(recording_id)
        if cached is not None and cached.request_fingerprint == request_fingerprint:
            logger.info(
                "insights_cache_hit recording_id=%s source_version=%s prompt_version=%s model=%s",
                recording_id,
                source_version,
                SESSION_INSIGHTS_PROMPT_VERSION,
                self.generator.model,
            )
            return cached

        started_at = monotonic()
        logger.info(
            "insights_started recording_id=%s source_version=%s prompt_version=%s model=%s",
            recording_id,
            source_version,
            SESSION_INSIGHTS_PROMPT_VERSION,
            self.generator.model,
        )

        result: SessionInsightsResult | None = None
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            candidate = self.generator.generate(recording_id, self.prompt, input_data)
            try:
                self._validate(candidate, input_data)
            except InsightsValidationError as error:
                logger.warning(
                    "insights_validation_failed recording_id=%s reason=%s attempt=%s",
                    recording_id,
                    error.reason,
                    attempt,
                )
                if attempt == MAX_GENERATION_ATTEMPTS:
                    raise
            else:
                result = candidate
                break

        if result is None:
            raise InsightsValidationError(
                "Session insights failed evidence validation",
                reason="missing_valid_result",
            )

        completed = CompletedSessionInsights(
            recording_id=recording_id,
            source_version=source_version,
            prompt_version=SESSION_INSIGHTS_PROMPT_VERSION,
            model=self.generator.model,
            request_fingerprint=request_fingerprint,
            generated_at=datetime.now(UTC),
            result=result,
        )
        self._save(completed)
        logger.info(
            "insights_completed recording_id=%s source_version=%s model=%s fact_count=%s "
            "action_count=%s section_count=%s duration_ms=%s",
            recording_id,
            source_version,
            self.generator.model,
            len(result.facts),
            len(result.action_mentions),
            len(result.sections),
            round((monotonic() - started_at) * 1000),
        )
        return completed

    @staticmethod
    def _source_version(segments: list[dict[str, str]]) -> str:
        canonical = json.dumps(segments, ensure_ascii=False, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def _request_fingerprint(self, input_data: dict[str, object]) -> str:
        canonical = json.dumps(
            {
                "input": input_data,
                "model": self.generator.model,
                "prompt_version": SESSION_INSIGHTS_PROMPT_VERSION,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _validate(result: SessionInsightsResult, input_data: dict[str, object]) -> None:
        if result.session_id != input_data["session_id"]:
            raise InsightsValidationError("Session ID changed", reason="session_id_mismatch")
        if result.source_version != input_data["source_version"]:
            raise InsightsValidationError(
                "Source version changed", reason="source_version_mismatch"
            )

        segments = input_data["segments"]
        if not isinstance(segments, list):
            raise InsightsValidationError("Input segments are invalid", reason="invalid_input")
        segment_text = {segment["id"]: segment["text"] for segment in segments}
        source_ids = set(segment_text)

        fact_ids = [fact.id for fact in result.facts]
        action_ids = [action.id for action in result.action_mentions]
        if len(fact_ids) != len(set(fact_ids)) or any(
            re.fullmatch(r"F\d+", identifier) is None for identifier in fact_ids
        ):
            raise InsightsValidationError("Fact IDs are invalid", reason="invalid_fact_ids")
        if len(action_ids) != len(set(action_ids)) or any(
            re.fullmatch(r"A\d+", identifier) is None for identifier in action_ids
        ):
            raise InsightsValidationError("Action IDs are invalid", reason="invalid_action_ids")

        for fact in result.facts:
            if (
                not fact.text.strip()
                or not fact.source_ids
                or not set(fact.source_ids) <= source_ids
            ):
                raise InsightsValidationError(
                    "A fact has invalid evidence", reason="invalid_fact_evidence"
                )

        actions_by_id = {action.id: action for action in result.action_mentions}
        for action in result.action_mentions:
            if (
                not action.activity.strip()
                or not action.source_excerpt.strip()
                or not action.source_ids
                or not set(action.source_ids) <= source_ids
            ):
                raise InsightsValidationError(
                    "An action has invalid evidence", reason="invalid_action_evidence"
                )
            if not any(
                action.source_excerpt in segment_text[source_id] for source_id in action.source_ids
            ):
                raise InsightsValidationError(
                    "An action excerpt is not exact", reason="inexact_action_excerpt"
                )
            if action.eligibility == "review_candidate" and (
                action.status != "agreed" or action.actor != "client"
            ):
                raise InsightsValidationError(
                    "An ineligible action is marked for review",
                    reason="invalid_review_candidate",
                )
            if action.status in {"suggested", "declined", "superseded"} and (
                action.eligibility != "not_eligible"
            ):
                raise InsightsValidationError(
                    "An action has invalid eligibility", reason="invalid_action_eligibility"
                )
            if action.replaces_action_id is not None:
                replaced = actions_by_id.get(action.replaces_action_id)
                if replaced is None or replaced.id == action.id or replaced.status != "superseded":
                    raise InsightsValidationError(
                        "An action replacement is invalid", reason="invalid_action_replacement"
                    )

        for action in result.action_mentions:
            seen = {action.id}
            current = action
            while current.replaces_action_id is not None:
                if current.replaces_action_id in seen:
                    raise InsightsValidationError(
                        "Action replacements contain a cycle",
                        reason="cyclic_action_replacement",
                    )
                seen.add(current.replaces_action_id)
                current = actions_by_id[current.replaces_action_id]

        section_keys = [section.key for section in result.sections]
        if len(section_keys) != len(set(section_keys)):
            raise InsightsValidationError(
                "Section keys are duplicated", reason="duplicate_sections"
            )
        known_actions = set(action_ids)
        for section in result.sections:
            if not section.heading.strip() or not section.bullets:
                raise InsightsValidationError("A section is empty", reason="empty_section")
            for bullet in section.bullets:
                if not bullet.text.strip():
                    raise InsightsValidationError(
                        "A section bullet is empty",
                        reason="empty_bullet",
                    )

        for flag in result.review_flags:
            if (
                not flag.detail.strip()
                or not set(flag.source_ids) <= source_ids
                or not set(flag.action_ids) <= known_actions
            ):
                raise InsightsValidationError(
                    "A review flag has invalid references", reason="invalid_review_flag"
                )

        if result.status == "insufficient_source":
            if (
                result.primary_variant is not None
                or result.secondary_variant is not None
                or result.selection_note is not None
                or result.facts
                or result.action_mentions
                or result.sections
            ):
                raise InsightsValidationError(
                    "Insufficient source result contains draft content",
                    reason="invalid_insufficient_source",
                )
        else:
            if result.primary_variant is None or not result.facts or not result.sections:
                raise InsightsValidationError(
                    "Draft result is incomplete", reason="incomplete_draft"
                )
            if result.secondary_variant == result.primary_variant:
                raise InsightsValidationError("Variants must differ", reason="duplicate_variants")

    def _load(self, recording_id: UUID) -> CompletedSessionInsights | None:
        path = self.directory / f"{recording_id}.insights.json"
        if not path.is_file():
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            request_fingerprint = payload.pop("request_fingerprint")
            response = SessionInsightsResponse.model_validate(payload)
            if response.recording_id != recording_id:
                raise ValueError("Insights recording ID does not match its filename")
            return CompletedSessionInsights(
                recording_id=response.recording_id,
                source_version=response.source_version,
                prompt_version=response.prompt_version,
                model=response.model,
                request_fingerprint=request_fingerprint,
                generated_at=response.generated_at,
                result=response.result,
            )
        except Exception as error:
            raise InsightsStorageError("Failed to load session insights") from error

    def _save(self, completed: CompletedSessionInsights) -> None:
        destination = self.directory / f"{completed.recording_id}.insights.json"
        temporary = self.directory / f".{completed.recording_id}.insights.tmp"
        payload = completed.response().model_dump(mode="json")
        payload["request_fingerprint"] = completed.request_fingerprint

        try:
            with temporary.open("x", encoding="utf-8") as output:
                json.dump(payload, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(destination)
        except Exception as error:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise InsightsStorageError("Failed to save session insights") from error
