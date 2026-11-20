import hashlib
import json
import os
import re
import unicodedata
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from time import monotonic
from typing import Protocol
from uuid import UUID

from app.core.logging import get_logger
from app.schemas.session_insights import (
    EnjoyableActivityCandidate,
    InsightDraftBullet,
    InsightDraftSection,
    InsightReviewFlag,
    MoodScaleDefinition,
    RecordMoodCandidate,
    RecordMoodConfiguration,
    SessionInsightsDraftResult,
    SessionInsightsResponse,
    SessionInsightsResult,
)
from app.services.transcript_store import TranscriptStore

SESSION_INSIGHTS_PROMPT_VERSION = "v0.6.3"
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
    reasoning_effort: str

    def generate(
        self,
        recording_id: UUID,
        prompt: str,
        input_data: dict[str, object],
    ) -> SessionInsightsResult: ...


def _to_draft_result(result: SessionInsightsResult) -> SessionInsightsDraftResult:
    item_number = 1
    sections: list[InsightDraftSection] = []
    for section in result.sections:
        bullets: list[InsightDraftBullet] = []
        for bullet in section.bullets:
            bullets.append(
                InsightDraftBullet(
                    **bullet.model_dump(),
                    item_id=f"B{item_number}",
                )
            )
            item_number += 1
        sections.append(
            InsightDraftSection(
                key=section.key,
                heading=section.heading,
                bullets=bullets,
            )
        )

    return SessionInsightsDraftResult(
        **result.model_dump(exclude={"sections"}),
        sections=sections,
    )


def _to_generated_result(result: SessionInsightsDraftResult) -> SessionInsightsResult:
    payload = result.model_dump()
    for section in payload["sections"]:
        for bullet in section["bullets"]:
            bullet.pop("item_id")
    return SessionInsightsResult.model_validate(payload)


@dataclass(frozen=True)
class CompletedSessionInsights:
    recording_id: UUID
    draft_id: str
    source_version: str
    prompt_version: str
    model: str
    request_fingerprint: str
    generated_at: datetime
    result: SessionInsightsResult

    def response(self) -> SessionInsightsResponse:
        return SessionInsightsResponse(
            recording_id=self.recording_id,
            draft_id=self.draft_id,
            source_version=self.source_version,
            prompt_version=self.prompt_version,
            model=self.model,
            generated_at=self.generated_at,
            result=_to_draft_result(self.result),
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
        if (
            cached is not None
            and cached.prompt_version == SESSION_INSIGHTS_PROMPT_VERSION
            and cached.request_fingerprint == request_fingerprint
        ):
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
            try:
                candidate = self.generator.generate(recording_id, self.prompt, input_data)
            except InsightsProviderError as error:
                if (
                    error.provider_error_type
                    not in {
                        "ValidationError",
                        "InvalidSessionInsightsResponse",
                    }
                    or attempt == MAX_GENERATION_ATTEMPTS
                ):
                    raise
                logger.warning(
                    "insights_provider_response_invalid recording_id=%s provider_error_type=%s "
                    "attempt=%s",
                    recording_id,
                    error.provider_error_type,
                    attempt,
                )
                continue
            candidate, omitted_mentions = self._omit_unlinked_activity_mentions(
                candidate, input_data
            )
            if omitted_mentions:
                logger.warning(
                    "insights_activity_mentions_omitted recording_id=%s count=%s attempt=%s",
                    recording_id,
                    omitted_mentions,
                    attempt,
                )
            candidate, omitted_actions = self._omit_unsupported_activity_candidates(
                candidate, input_data
            )
            if omitted_actions:
                logger.warning(
                    "insights_activity_candidates_omitted recording_id=%s count=%s attempt=%s",
                    recording_id,
                    omitted_actions,
                    attempt,
                )
            candidate, recovered_mood = self._recover_explicit_mood_commitment(
                candidate, input_data
            )
            if recovered_mood:
                logger.info(
                    "insights_explicit_mood_candidate_recovered recording_id=%s attempt=%s",
                    recording_id,
                    attempt,
                )
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
            draft_id=self._draft_id(request_fingerprint, result),
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
            "candidate_count=%s section_count=%s duration_ms=%s",
            recording_id,
            source_version,
            self.generator.model,
            len(result.facts),
            len(result.action_candidates),
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
                "reasoning_effort": self.generator.reasoning_effort,
                "prompt_version": SESSION_INSIGHTS_PROMPT_VERSION,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    @staticmethod
    def _draft_id(request_fingerprint: str, result: SessionInsightsResult) -> str:
        canonical = json.dumps(
            {
                "request_fingerprint": request_fingerprint,
                "result": result.model_dump(mode="json"),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    def load_completed(self, recording_id: UUID) -> CompletedSessionInsights | None:
        return self._load(recording_id)

    @staticmethod
    def _omit_unlinked_activity_mentions(
        result: SessionInsightsResult, input_data: dict[str, object]
    ) -> tuple[SessionInsightsResult, int]:
        segments = input_data.get("segments")
        if not isinstance(segments, list):
            return result, 0
        source_ids = {segment["id"] for segment in segments}
        omitted = 0
        candidates = []
        for candidate in result.action_candidates:
            if not isinstance(candidate, EnjoyableActivityCandidate):
                candidates.append(candidate)
                continue
            valid_mentions = [
                mention
                for mention in candidate.activity_mentions
                if set(mention.source_ids) <= source_ids
            ]
            omitted += len(candidate.activity_mentions) - len(valid_mentions)
            candidates.append(candidate.model_copy(update={"activity_mentions": valid_mentions}))
        return result.model_copy(update={"action_candidates": candidates}), omitted

    @staticmethod
    def _normalize_excerpt(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).translate(
            str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-"})
        )
        return " ".join(normalized.casefold().split())

    @staticmethod
    def _recover_explicit_mood_commitment(
        result: SessionInsightsResult, input_data: dict[str, object]
    ) -> tuple[SessionInsightsResult, bool]:
        if result.status != "draft" or any(
            candidate.type == "record_mood" for candidate in result.action_candidates
        ):
            return result, False
        segments = input_data.get("segments")
        if not isinstance(segments, list):
            return result, False

        commitment = re.compile(
            r"(?:^|(?<=[.!?])\s+)I(?:['’]ll| will|['’]m going to| plan to)\s+"
            r"(?:rate|record|log|track|write down)\s+"
            r"(?:my\s+|the\s+)?(?:overall\s+|daily\s+)?mood\b",
            re.IGNORECASE,
        )
        retraction = re.compile(
            r"\bI\s+(?:won['’]t|will not|decided not to|changed my mind about|"
            r"am not going to)\s+(?:\w+\s+){0,5}?(?:rate|record|log|track|"
            r"write down)?\s*(?:my\s+|the\s+)?mood\b",
            re.IGNORECASE,
        )
        for index, segment in enumerate(segments):
            if segment.get("role") != "client" or not isinstance(segment.get("text"), str):
                continue
            text = segment["text"]
            match = commitment.search(text)
            if match is None:
                continue
            sentence_end = re.search(r"[.!?]", text[match.end() :])
            end = match.end() + sentence_end.end() if sentence_end else len(text)
            task = text[match.start() : end].strip()
            if re.search(r"\b(?:if|maybe|might|probably)\b", task, re.IGNORECASE):
                continue
            later_client_text = [text[end:]] + [
                later["text"]
                for later in segments[index + 1 :]
                if later.get("role") == "client" and isinstance(later.get("text"), str)
            ]
            if any(retraction.search(later_text) for later_text in later_client_text):
                continue
            frequency_match = re.search(
                r"\b(?:each|every)\s+(?:morning|evening|day|night|week)\b"
                r"|\b(?:daily|weekly|once a day)\b",
                task,
                re.IGNORECASE,
            )
            timing_match = re.search(
                r"\b(?:around|at)\s+(?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?|"
                r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b",
                task,
                re.IGNORECASE,
            )
            optional_energy = bool(
                re.search(r"\bI\s+can\s+add\s+an?\s+energy\s+rating\b", text, re.I)
            )
            existing_numbers = [
                int(candidate.id[1:])
                for candidate in result.action_candidates
                if re.fullmatch(r"C\d+", candidate.id)
            ]
            candidate = RecordMoodCandidate(
                id=f"C{max(existing_numbers, default=0) + 1}",
                type="record_mood",
                state="review_ready",
                actor="client",
                exact_task=task,
                source_excerpt=task,
                what_to_record=["mood", "energy"] if optional_energy else ["mood"],
                frequency=frequency_match.group() if frequency_match else "as needed",
                timing=timing_match.group() if timing_match else None,
                reference_period="whole day",
                start_date=None,
                end_date=None,
                duration_days=14,
                configuration=RecordMoodConfiguration(
                    input_kind="numeric_0_10",
                    context_enabled=True,
                    scale_origin="moodi_default",
                    scale=MoodScaleDefinition(
                        id="overall-day-mood-0-10-v1",
                        construct="overall_day_mood",
                        minimum=0,
                        maximum=10,
                        minimum_label="worst",
                        maximum_label="best",
                        direction="low_to_high",
                        version="1",
                    ),
                ),
                unknowns=[],
                source_ids=[segment["id"]],
            )
            return result.model_copy(
                update={"action_candidates": [*result.action_candidates, candidate]}
            ), True
        return result, False

    @staticmethod
    def _omit_unsupported_activity_candidates(
        result: SessionInsightsResult, input_data: dict[str, object]
    ) -> tuple[SessionInsightsResult, int]:
        segments = input_data.get("segments")
        if not isinstance(segments, list):
            return result, 0
        segments_by_id = {segment["id"]: segment["text"] for segment in segments}
        kept = []
        omitted_ids = set()
        seen_assignments = set()
        unsupported_omitted = False
        for candidate in result.action_candidates:
            if not isinstance(candidate, EnjoyableActivityCandidate):
                kept.append(candidate)
                continue
            excerpt = SessionInsightsService._normalize_excerpt(candidate.source_excerpt)
            if (
                not candidate.exact_task.strip()
                or not candidate.source_ids
                or not set(candidate.source_ids) <= segments_by_id.keys()
                or not excerpt
                or not any(
                    excerpt in SessionInsightsService._normalize_excerpt(segments_by_id[source_id])
                    for source_id in candidate.source_ids
                )
            ):
                omitted_ids.add(candidate.id)
                unsupported_omitted = True
            else:
                assignment = (
                    tuple(sorted(candidate.source_ids)),
                    candidate.assignment_kind,
                    SessionInsightsService._normalize_excerpt(candidate.agreed_activity or ""),
                )
                if assignment in seen_assignments:
                    omitted_ids.add(candidate.id)
                else:
                    seen_assignments.add(assignment)
                    kept.append(candidate)
        if not omitted_ids:
            return result, 0
        sections = [
            section.model_copy(
                update={
                    "bullets": [
                        bullet.model_copy(
                            update={
                                "candidate_ids": [
                                    identifier
                                    for identifier in bullet.candidate_ids
                                    if identifier not in omitted_ids
                                ]
                            }
                        )
                        for bullet in section.bullets
                    ]
                }
            )
            for section in result.sections
        ]
        flags = [
            flag.model_copy(
                update={
                    "candidate_ids": [
                        identifier
                        for identifier in flag.candidate_ids
                        if identifier not in omitted_ids
                    ]
                }
            )
            for flag in result.review_flags
        ]
        if unsupported_omitted:
            flags.append(
                InsightReviewFlag(
                    code="missing_action_detail",
                    detail=(
                        "An activity action was omitted because its transcript evidence "
                        "could not be verified."
                    ),
                    source_ids=[],
                    candidate_ids=[],
                )
            )
        return result.model_copy(
            update={"action_candidates": kept, "sections": sections, "review_flags": flags}
        ), len(omitted_ids)

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
        source_ids = {segment["id"] for segment in segments}
        segments_by_id = {segment["id"]: segment["text"] for segment in segments}

        fact_ids = [fact.id for fact in result.facts]
        if sum(candidate.type == "record_mood" for candidate in result.action_candidates) > 1:
            raise InsightsValidationError(
                "Only one combined Record mood candidate is supported",
                reason="multiple_mood_candidates",
            )
        candidate_ids = [candidate.id for candidate in result.action_candidates]
        if len(fact_ids) != len(set(fact_ids)) or any(
            re.fullmatch(r"F\d+", identifier) is None for identifier in fact_ids
        ):
            raise InsightsValidationError("Fact IDs are invalid", reason="invalid_fact_ids")
        if len(candidate_ids) != len(set(candidate_ids)) or any(
            re.fullmatch(r"C\d+", identifier) is None for identifier in candidate_ids
        ):
            raise InsightsValidationError(
                "Candidate IDs are invalid", reason="invalid_candidate_ids"
            )

        for fact in result.facts:
            if (
                not fact.text.strip()
                or not fact.source_ids
                or not set(fact.source_ids) <= source_ids
            ):
                raise InsightsValidationError(
                    "A fact has invalid evidence", reason="invalid_fact_evidence"
                )

        for candidate in result.action_candidates:
            if (
                not candidate.exact_task.strip()
                or not candidate.source_excerpt.strip()
                or not candidate.source_ids
                or not set(candidate.source_ids) <= source_ids
            ):
                raise InsightsValidationError(
                    "A candidate has invalid evidence", reason="invalid_candidate_evidence"
                )
            if not any(
                SessionInsightsService._normalize_excerpt(candidate.source_excerpt)
                in SessionInsightsService._normalize_excerpt(segments_by_id[source_id])
                for source_id in candidate.source_ids
            ):
                raise InsightsValidationError(
                    "A candidate excerpt is not present in its cited evidence",
                    reason="invalid_candidate_excerpt",
                )
            if isinstance(candidate, EnjoyableActivityCandidate):
                for mention in candidate.activity_mentions:
                    if not set(mention.source_ids) <= source_ids:
                        raise InsightsValidationError(
                            "An activity mention has invalid evidence",
                            reason="invalid_activity_mention_evidence",
                        )

        section_keys = [section.key for section in result.sections]
        if len(section_keys) != len(set(section_keys)):
            raise InsightsValidationError(
                "Section keys are duplicated", reason="duplicate_sections"
            )
        known_candidates = set(candidate_ids)
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
                or not set(flag.candidate_ids) <= known_candidates
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
                or result.action_candidates
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
            if payload.get("prompt_version") not in {
                SESSION_INSIGHTS_PROMPT_VERSION,
                "v0.5.0",
                "v0.5.1",
                "v0.6.0",
                "v0.6.1",
            }:
                return None
            request_fingerprint = payload.pop("request_fingerprint")
            if "draft_id" not in payload:
                legacy_result = SessionInsightsResult.model_validate(payload["result"])
                payload["draft_id"] = self._draft_id(request_fingerprint, legacy_result)
                payload["result"] = _to_draft_result(legacy_result).model_dump(mode="json")
            response = SessionInsightsResponse.model_validate(payload)
            if response.recording_id != recording_id:
                raise ValueError("Insights recording ID does not match its filename")
            result = _to_generated_result(response.result)
            if response.draft_id != self._draft_id(request_fingerprint, result):
                raise ValueError("Insights draft ID does not match its contents")
            return CompletedSessionInsights(
                recording_id=response.recording_id,
                draft_id=response.draft_id,
                source_version=response.source_version,
                prompt_version=response.prompt_version,
                model=response.model,
                request_fingerprint=request_fingerprint,
                generated_at=response.generated_at,
                result=result,
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
