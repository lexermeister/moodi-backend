import json
import os
import re
import shutil
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import RLock
from uuid import NAMESPACE_URL, UUID, uuid5
from zoneinfo import ZoneInfo

from app.schemas.actions import (
    ActionReviewConfirmationRequest,
    ActionReviewConfirmationResponse,
    ActivityAction,
    ActivityActionSettings,
    AmendMoodEntryRequest,
    ConfirmMoodEntryRequest,
    MoodAction,
    MoodCandidateResolutionResponse,
    MoodCandidateReviewRequest,
    MoodCandidateReviewResponse,
    MoodEntry,
    MoodEntryRevision,
    MoodEntryValues,
    RecordMoodActionSettings,
    SaveMoodEntryDraftRequest,
)
from app.schemas.session_insights import EnjoyableActivityCandidate, RecordMoodCandidate
from app.services.session_insights import CompletedSessionInsights, SessionInsightsService


class ActionError(Exception):
    """Base error raised by the Actions storage and validation boundary."""


class ActionNotFoundError(ActionError):
    """Raised when an Action or mood entry does not exist."""


class ActionConflictError(ActionError):
    """Raised when a write conflicts with newer reviewed state."""


class ActionValidationError(ActionError):
    """Raised when reviewed input violates the Action configuration."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


class ActionStorageError(ActionError):
    """Raised when an Action file cannot be durably read or written."""


class ActionService:
    def __init__(self, directory: Path, insights_service: SessionInsightsService) -> None:
        self.root = directory / "actions"
        self.actions_directory = self.root / "records"
        self.reviews_directory = self.root / "reviews"
        self.confirmations_directory = self.root / "confirmations"
        self.entries_directory = self.root / "entries"
        self.deletions_directory = self.root / ".deleting"
        self.insights_service = insights_service
        self._mutation_lock = RLock()

    def prepare(self) -> None:
        self.actions_directory.mkdir(parents=True, exist_ok=True)
        self.reviews_directory.mkdir(parents=True, exist_ok=True)
        self.confirmations_directory.mkdir(parents=True, exist_ok=True)
        self.entries_directory.mkdir(parents=True, exist_ok=True)
        self.deletions_directory.mkdir(parents=True, exist_ok=True)
        self._recover_pending_deletions()

    def review_candidate(
        self,
        recording_id: UUID,
        candidate_id: str,
        request: MoodCandidateReviewRequest,
    ) -> MoodCandidateReviewResponse:
        with self._mutation_lock:
            return self._review_candidate(recording_id, candidate_id, request)

    def _review_candidate(
        self,
        recording_id: UUID,
        candidate_id: str,
        request: MoodCandidateReviewRequest,
    ) -> MoodCandidateReviewResponse:
        self.prepare()
        if re.fullmatch(r"C\d+", candidate_id) is None:
            raise ActionNotFoundError("Record mood candidate was not found")

        draft = self._load_current_draft(recording_id)
        if request.draft_id != draft.draft_id or request.source_version != draft.source_version:
            raise ActionConflictError(
                "Session insights changed; review the latest candidate before saving"
            )
        candidate = next(
            (item for item in draft.result.action_candidates if item.id == candidate_id),
            None,
        )
        if candidate is None:
            raise ActionNotFoundError("Record mood candidate was not found")
        if isinstance(candidate, EnjoyableActivityCandidate):
            return self._review_activity_candidate(recording_id, candidate, draft, request)
        if request.settings is not None and not isinstance(
            request.settings, RecordMoodActionSettings
        ):
            raise ActionValidationError(
                "Mood settings are required", reason="invalid_settings_type"
            )

        existing = self._load_review(recording_id, candidate_id)
        if existing is not None:
            if (
                existing.draft_id != request.draft_id
                or existing.source_version != request.source_version
            ):
                raise ActionConflictError(
                    "The candidate was already reviewed from a different insights draft"
                )
            requested_dispositions = (
                {"activated", "linked_existing"}
                if request.disposition == "activate"
                else {"rejected"}
            )
            if existing.disposition not in requested_dispositions:
                raise ActionConflictError("The candidate has already been reviewed")
            if (
                request.disposition == "activate"
                and existing.action is not None
                and request.settings is not None
                and (
                    set(request.settings.what_to_record)
                    != set(existing.action.settings.what_to_record)
                    or request.settings.cadence != existing.action.settings.cadence
                    or request.settings.weekly_weekday != existing.action.settings.weekly_weekday
                    or request.settings.preferred_check_in_time
                    != existing.action.settings.preferred_check_in_time
                    or request.settings.preferred_time_zone
                    != existing.action.settings.preferred_time_zone
                )
            ):
                raise ActionConflictError(
                    "This candidate was activated with different measurements"
                )
            return self._review_with_current_action(existing)

        reviewed_at = datetime.now(UTC)
        action: MoodAction | None = None
        review_disposition = "rejected"
        if request.disposition == "activate":
            settings = request.settings
            if settings is None:
                raise ActionValidationError(
                    "Activation requires reviewed settings",
                    reason="missing_reviewed_settings",
                )
            start_date = (
                None
                if settings.duration_kind == "unscheduled"
                else settings.start_date
                or (
                    reviewed_at.astimezone(ZoneInfo(settings.preferred_time_zone)).date()
                    if settings.preferred_time_zone
                    else reviewed_at.date()
                )
            )
            if settings.duration_kind in ("ongoing", "unscheduled"):
                end_date = None
            else:
                if settings.duration_days is None:
                    raise ActionValidationError(
                        "Fixed Actions require a duration",
                        reason="missing_duration",
                    )
                end_date = settings.end_date or start_date + timedelta(
                    days=settings.duration_days - 1
                )
            settings = settings.model_copy(
                update={
                    "start_date": start_date,
                    "end_date": end_date,
                }
            )
            settings = RecordMoodActionSettings.model_validate(settings.model_dump())
            self._validate_reviewed_settings(candidate, settings)
            ongoing_action = self._find_active_ongoing_action()
            if ongoing_action is not None:
                if not self._candidate_matches_action(candidate, ongoing_action):
                    raise ActionConflictError(
                        "An active Record mood Action uses a different mood scale"
                    )
                settings = settings.model_copy(
                    update={"start_date": ongoing_action.settings.start_date or settings.start_date}
                )
                settings = RecordMoodActionSettings.model_validate(settings.model_dump())
                action = ongoing_action
                if action.settings != settings:
                    action = action.model_copy(
                        update={"settings": settings, "revision": action.revision + 1}
                    )
                    self._atomic_write(self._action_path(action.id), action.model_dump(mode="json"))
                review_disposition = "linked_existing"
            else:
                action_id = uuid5(
                    NAMESPACE_URL,
                    f"moodi:{recording_id}:{draft.source_version}:{candidate_id}",
                )
                action = MoodAction(
                    id=action_id,
                    type="record_mood",
                    state="active",
                    recording_id=recording_id,
                    transcript_version=draft.source_version,
                    candidate_id=candidate_id,
                    source_ids=candidate.source_ids,
                    source_excerpt=candidate.source_excerpt,
                    settings=settings,
                    created_at=reviewed_at,
                    revision=1,
                )
                existing_action = self._load_action_if_present(action.id)
                if existing_action is None:
                    self._atomic_write(self._action_path(action.id), action.model_dump(mode="json"))
                elif (
                    existing_action.recording_id != recording_id
                    or existing_action.transcript_version != draft.source_version
                    or existing_action.candidate_id != candidate_id
                    or existing_action.settings != settings
                ):
                    raise ActionConflictError("A different Action already uses this candidate")
                else:
                    action = existing_action
                review_disposition = "activated"

        review = MoodCandidateReviewResponse(
            recording_id=recording_id,
            candidate_id=candidate_id,
            draft_id=draft.draft_id,
            source_version=draft.source_version,
            disposition=review_disposition,
            reviewed_at=reviewed_at,
            action=action,
        )
        self._atomic_write(
            self._review_path(recording_id, candidate_id),
            review.model_dump(mode="json"),
        )
        return review

    def _review_activity_candidate(
        self,
        recording_id: UUID,
        candidate: EnjoyableActivityCandidate,
        draft: CompletedSessionInsights,
        request: MoodCandidateReviewRequest,
    ) -> MoodCandidateReviewResponse:
        existing = self._load_review(recording_id, candidate.id)
        if existing is not None:
            if (
                existing.draft_id != draft.draft_id
                or existing.source_version != draft.source_version
            ):
                raise ActionConflictError(
                    "The candidate was already reviewed from a different insights draft"
                )
            if request.disposition == "reject" and existing.disposition == "rejected":
                return existing
            if request.disposition == "activate" and existing.disposition == "activated":
                if (
                    not isinstance(request.settings, ActivityActionSettings)
                    or not isinstance(existing.action, ActivityAction)
                    or request.settings != existing.action.settings
                ):
                    raise ActionConflictError("The activity was activated with different settings")
                return self._review_with_current_action(existing)
            raise ActionConflictError("The candidate has already been reviewed")

        reviewed_at = datetime.now(UTC)
        action: ActivityAction | None = None
        if request.disposition == "activate":
            settings = request.settings
            if not isinstance(settings, ActivityActionSettings):
                raise ActionValidationError(
                    "Activity settings are required", reason="invalid_settings_type"
                )
            if (
                settings.exact_task != candidate.exact_task
                or settings.assignment_kind != candidate.assignment_kind
                or settings.agreed_activity != candidate.agreed_activity
            ):
                raise ActionValidationError(
                    "The agreed activity cannot be changed during setup",
                    reason="agreed_assignment_changed",
                )
            action_id = uuid5(
                NAMESPACE_URL,
                f"moodi:{recording_id}:{draft.source_version}:{candidate.id}",
            )
            action = ActivityAction(
                id=action_id,
                type="do_enjoyable_or_meaningful_activity",
                state="active",
                recording_id=recording_id,
                transcript_version=draft.source_version,
                candidate_id=candidate.id,
                source_ids=candidate.source_ids,
                source_excerpt=candidate.source_excerpt,
                agreed_frequency=candidate.frequency,
                agreed_timing=candidate.timing,
                agreed_reference_period=candidate.reference_period,
                activity_mentions=candidate.activity_mentions,
                settings=settings,
                created_at=reviewed_at,
                revision=1,
            )
            prior = self._load_action_if_present(action_id)
            if prior is not None:
                if (
                    not isinstance(prior, ActivityAction)
                    or prior.recording_id != recording_id
                    or prior.transcript_version != draft.source_version
                    or prior.candidate_id != candidate.id
                    or prior.settings != settings
                ):
                    raise ActionConflictError("A different Action already uses this candidate")
                action = prior
            else:
                self._atomic_write(self._action_path(action.id), action.model_dump(mode="json"))
        review = MoodCandidateReviewResponse(
            recording_id=recording_id,
            candidate_id=candidate.id,
            draft_id=draft.draft_id,
            source_version=draft.source_version,
            disposition="activated" if action is not None else "rejected",
            reviewed_at=reviewed_at,
            action=action,
        )
        self._atomic_write(
            self._review_path(recording_id, candidate.id), review.model_dump(mode="json")
        )
        return review

    def resolve_candidate(
        self, recording_id: UUID, candidate_id: str
    ) -> MoodCandidateResolutionResponse:
        with self._mutation_lock:
            self.prepare()
            if re.fullmatch(r"C\d+", candidate_id) is None:
                raise ActionNotFoundError("Record mood candidate was not found")
            draft = self._load_current_draft(recording_id)
            candidate = next(
                (item for item in draft.result.action_candidates if item.id == candidate_id),
                None,
            )
            if candidate is None:
                raise ActionNotFoundError("Record mood candidate was not found")

            review = self._load_review(recording_id, candidate_id)
            if review is not None:
                if (
                    review.draft_id != draft.draft_id
                    or review.source_version != draft.source_version
                ):
                    raise ActionConflictError(
                        "Session insights changed; review the latest candidate"
                    )
                return MoodCandidateResolutionResponse(
                    recording_id=recording_id,
                    candidate_id=candidate_id,
                    draft_id=draft.draft_id,
                    source_version=draft.source_version,
                    status=review.disposition,
                    action=self._review_with_current_action(review).action,
                )

            if isinstance(candidate, EnjoyableActivityCandidate):
                return MoodCandidateResolutionResponse(
                    recording_id=recording_id,
                    candidate_id=candidate_id,
                    draft_id=draft.draft_id,
                    source_version=draft.source_version,
                    status="needs_setup",
                    action=None,
                )

            ongoing_action = self._find_active_ongoing_action()
            if ongoing_action is None:
                return MoodCandidateResolutionResponse(
                    recording_id=recording_id,
                    candidate_id=candidate_id,
                    draft_id=draft.draft_id,
                    source_version=draft.source_version,
                    status="needs_setup",
                    action=None,
                )
            compatible = self._candidate_matches_action(candidate, ongoing_action)
            return MoodCandidateResolutionResponse(
                recording_id=recording_id,
                candidate_id=candidate_id,
                draft_id=draft.draft_id,
                source_version=draft.source_version,
                status="already_active" if compatible else "conflict",
                action=ongoing_action,
                reason=None if compatible else "incompatible_ongoing_action",
            )

    def load_candidate_review(
        self, recording_id: UUID, candidate_id: str
    ) -> MoodCandidateReviewResponse:
        self._recover_pending_deletions()
        review = self._load_review(recording_id, candidate_id)
        if review is None:
            raise ActionNotFoundError("Action candidate review was not found")
        draft = self._load_current_draft(recording_id)
        if review.draft_id != draft.draft_id or review.source_version != draft.source_version:
            raise ActionConflictError("Session insights changed; review the latest candidate")
        return self._review_with_current_action(review)

    def confirm_action_reviews(
        self, recording_id: UUID, request: ActionReviewConfirmationRequest
    ) -> ActionReviewConfirmationResponse:
        with self._mutation_lock:
            self.prepare()
            draft = self._load_current_draft(recording_id)
            if request.draft_id != draft.draft_id or request.source_version != draft.source_version:
                raise ActionConflictError("Session insights changed; review the latest actions")
            candidate_ids = [candidate.id for candidate in draft.result.action_candidates]
            if not candidate_ids:
                raise ActionValidationError(
                    "There are no proposed actions to confirm", reason="no_action_candidates"
                )
            action_ids = self._reviewed_action_ids(
                recording_id, candidate_ids, draft.draft_id, draft.source_version
            )
            path = self.confirmations_directory / f"{recording_id}.json"
            if path.is_file():
                existing = self._load_model(
                    path, ActionReviewConfirmationResponse, "Failed to load Action confirmation"
                )
                if (
                    existing.draft_id == draft.draft_id
                    and existing.source_version == draft.source_version
                    and existing.candidate_ids == candidate_ids
                    and existing.action_ids == action_ids
                ):
                    return existing
            confirmation = ActionReviewConfirmationResponse(
                recording_id=recording_id,
                draft_id=draft.draft_id,
                source_version=draft.source_version,
                candidate_ids=candidate_ids,
                action_ids=action_ids,
                confirmed_at=datetime.now(UTC),
            )
            self._atomic_write(path, confirmation.model_dump(mode="json"))
            return confirmation

    def load_action_review_confirmation(
        self, recording_id: UUID
    ) -> ActionReviewConfirmationResponse:
        with self._mutation_lock:
            self.prepare()
            path = self.confirmations_directory / f"{recording_id}.json"
            if not path.is_file():
                raise ActionNotFoundError("Action review confirmation was not found")
            confirmation = self._load_model(
                path, ActionReviewConfirmationResponse, "Failed to load Action confirmation"
            )
            draft = self._load_current_draft(recording_id)
            candidate_ids = [candidate.id for candidate in draft.result.action_candidates]
            if (
                confirmation.draft_id != draft.draft_id
                or confirmation.source_version != draft.source_version
                or confirmation.candidate_ids != candidate_ids
            ):
                raise ActionConflictError("Session insights changed; review the latest actions")
            try:
                action_ids = self._reviewed_action_ids(
                    recording_id, candidate_ids, draft.draft_id, draft.source_version
                )
            except ActionValidationError as error:
                raise ActionNotFoundError(
                    "Action review confirmation is no longer current"
                ) from error
            if confirmation.action_ids != action_ids:
                raise ActionNotFoundError("Action review confirmation is no longer current")
            return confirmation

    def _reviewed_action_ids(
        self, recording_id: UUID, candidate_ids: list[str], draft_id: str, source_version: str
    ) -> list[UUID]:
        action_ids: list[UUID] = []
        for candidate_id in candidate_ids:
            review = self._load_review(recording_id, candidate_id)
            if (
                review is None
                or review.draft_id != draft_id
                or review.source_version != source_version
            ):
                raise ActionValidationError(
                    "Review every proposed action before confirming",
                    reason="unreviewed_action_candidate",
                )
            if review.disposition == "rejected" and review.action is None:
                continue
            action = (
                self._load_action_if_present(review.action.id)
                if review.action is not None
                else None
            )
            if review.disposition not in ("activated", "linked_existing") or (
                action is None or action.state != "active"
            ):
                raise ActionValidationError(
                    "Review every proposed action before confirming",
                    reason="unreviewed_action_candidate",
                )
            action_ids.append(action.id)
        return action_ids

    def list_actions(self, recording_id: UUID | None = None) -> list[MoodAction | ActivityAction]:
        self._recover_pending_deletions()
        actions: list[MoodAction | ActivityAction] = []
        if not self.actions_directory.is_dir():
            return actions
        try:
            paths = list(self.actions_directory.glob("*.json"))
        except OSError as error:
            raise ActionStorageError("Failed to list Actions") from error
        for path in paths:
            action = self._load_stored_action(path)
            if recording_id is None or action.recording_id == recording_id:
                actions.append(action)
        return sorted(actions, key=lambda item: item.created_at, reverse=True)

    def load_action(self, action_id: UUID) -> MoodAction | ActivityAction:
        self._recover_pending_deletions()
        action = self._load_action_if_present(action_id)
        if action is None:
            raise ActionNotFoundError("Mood Action was not found")
        return action

    def delete_action(self, action_id: UUID) -> tuple[MoodAction | ActivityAction, int]:
        with self._mutation_lock:
            return self._delete_action(action_id)

    def _delete_action(self, action_id: UUID) -> tuple[MoodAction | ActivityAction, int]:
        self.prepare()
        action = self.load_action(action_id)
        action_path = self._action_path(action_id)
        review_paths = self._review_paths_for_action(action_id)
        confirmation_paths = self._confirmation_paths_for_action(action_id)
        entries_path = self._entry_directory(action_id)
        deletion_path = self._deletion_path(action_id)

        try:
            entry_count = (
                sum(1 for path in entries_path.glob("*.json") if path.is_file())
                if entries_path.is_dir()
                else 0
            )
            deletion_path.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            raise ActionStorageError("Failed to prepare Action deletion") from error

        moves = [
            *[
                (review_path, deletion_path / "reviews" / review_path.name)
                for review_path in review_paths
            ],
            *[(path, deletion_path / "confirmations" / path.name) for path in confirmation_paths],
            (entries_path, deletion_path / "entries" / entries_path.name),
            (action_path, deletion_path / "records" / action_path.name),
        ]
        moved: list[tuple[Path, Path]] = []
        try:
            for source, destination in moves:
                if not source.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.replace(destination)
                moved.append((source, destination))
        except OSError as error:
            rollback_error = self._rollback_staged_deletion(moved, deletion_path)
            raise ActionStorageError("Failed to stage Action deletion") from (
                rollback_error or error
            )

        try:
            shutil.rmtree(deletion_path)
        except OSError as error:
            raise ActionStorageError("Failed to finalize Action deletion") from error
        return action, entry_count

    def save_draft(
        self, action_id: UUID, entry_id: UUID, request: SaveMoodEntryDraftRequest
    ) -> MoodEntry:
        with self._mutation_lock:
            return self._save_draft(action_id, entry_id, request)

    def _save_draft(
        self,
        action_id: UUID,
        entry_id: UUID,
        request: SaveMoodEntryDraftRequest,
    ) -> MoodEntry:
        action = self.load_action(action_id)
        self._validate_values(action, request.values, confirmed=False)
        existing = self._load_entry_if_present(action_id, entry_id)
        if existing is not None and existing.status == "confirmed":
            raise ActionConflictError("Confirmed entries must be amended")
        self._ensure_revision(existing, request.expected_revision)
        if existing is not None and existing.values == request.values:
            return existing

        now = datetime.now(UTC)
        entry = self._next_entry(
            existing=existing,
            entry_id=entry_id,
            action_id=action_id,
            values=request.values,
            status="draft",
            now=now,
            idempotency_key=None,
        )
        self._save_entry(entry)
        return entry

    def confirm_entry(
        self, action_id: UUID, entry_id: UUID, request: ConfirmMoodEntryRequest
    ) -> MoodEntry:
        with self._mutation_lock:
            return self._confirm_entry(action_id, entry_id, request)

    def _confirm_entry(
        self,
        action_id: UUID,
        entry_id: UUID,
        request: ConfirmMoodEntryRequest,
    ) -> MoodEntry:
        action = self.load_action(action_id)
        self._validate_values(action, request.values, confirmed=True)
        existing = self._load_entry_if_present(action_id, entry_id)
        if existing is not None and existing.status == "confirmed":
            if (
                existing.idempotency_key == request.idempotency_key
                and existing.values == request.values
            ):
                return existing
            raise ActionConflictError("Mood entry is already confirmed")
        self._ensure_revision(existing, request.expected_revision)

        now = datetime.now(UTC)
        entry = self._next_entry(
            existing=existing,
            entry_id=entry_id,
            action_id=action_id,
            values=request.values,
            status="confirmed",
            now=now,
            idempotency_key=request.idempotency_key,
        )
        self._save_entry(entry)
        return entry

    def amend_entry(
        self, action_id: UUID, entry_id: UUID, request: AmendMoodEntryRequest
    ) -> MoodEntry:
        with self._mutation_lock:
            return self._amend_entry(action_id, entry_id, request)

    def _amend_entry(
        self,
        action_id: UUID,
        entry_id: UUID,
        request: AmendMoodEntryRequest,
    ) -> MoodEntry:
        action = self.load_action(action_id)
        self._validate_values(action, request.values, confirmed=True)
        existing = self._load_entry_if_present(action_id, entry_id)
        if existing is None:
            raise ActionNotFoundError("Mood entry was not found")
        if existing.status != "confirmed":
            raise ActionConflictError("Draft entries must be saved or confirmed")
        self._ensure_revision(existing, request.expected_revision)
        if existing.values == request.values:
            return existing

        entry = self._next_entry(
            existing=existing,
            entry_id=entry_id,
            action_id=action_id,
            values=request.values,
            status="confirmed",
            now=datetime.now(UTC),
            idempotency_key=existing.idempotency_key,
        )
        self._save_entry(entry)
        return entry

    def list_entries(
        self,
        action_id: UUID,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[MoodEntry]:
        if not isinstance(self.load_action(action_id), MoodAction):
            raise ActionValidationError(
                "This Action does not accept mood entries", reason="wrong_action_type"
            )
        directory = self._entry_directory(action_id)
        if not directory.is_dir():
            return []
        entries = [
            self._load_model(path, MoodEntry, "Failed to load mood entry")
            for path in directory.glob("*.json")
        ]
        return sorted(
            [
                entry
                for entry in entries
                if (date_from is None or entry.values.relevant_date >= date_from)
                and (date_to is None or entry.values.relevant_date <= date_to)
            ],
            key=lambda item: (item.values.relevant_date, item.values.entry_time),
            reverse=True,
        )

    def delete_entry(self, action_id: UUID, entry_id: UUID) -> None:
        with self._mutation_lock:
            self._delete_entry(action_id, entry_id)

    def _delete_entry(self, action_id: UUID, entry_id: UUID) -> None:
        if not isinstance(self.load_action(action_id), MoodAction):
            raise ActionValidationError(
                "This Action does not accept mood entries", reason="wrong_action_type"
            )
        path = self._entry_path(action_id, entry_id)
        if not path.is_file():
            raise ActionNotFoundError("Mood entry was not found")
        try:
            path.unlink()
        except OSError as error:
            raise ActionStorageError("Failed to delete mood entry") from error

    def export_entries(self, action_id: UUID, entry_ids: list[UUID]) -> list[MoodEntry]:
        if len(entry_ids) != len(set(entry_ids)):
            raise ActionValidationError(
                "Export entry IDs are duplicated",
                reason="duplicate_export_entries",
            )
        entries_by_id = {
            entry.id: entry for entry in self.list_entries(action_id) if entry.status == "confirmed"
        }
        if not set(entry_ids) <= set(entries_by_id):
            raise ActionValidationError(
                "Export contains missing or unconfirmed entries",
                reason="invalid_export_entries",
            )
        return [entries_by_id[entry_id] for entry_id in entry_ids]

    @staticmethod
    def _validate_reviewed_settings(
        candidate: RecordMoodCandidate,
        settings: RecordMoodActionSettings,
    ) -> None:
        if candidate.state != "review_ready":
            raise ActionValidationError(
                "Record mood candidate is not ready for review",
                reason="candidate_not_reviewable",
            )
        if (
            candidate.exact_task != settings.exact_task
            or candidate.frequency != settings.frequency
            or candidate.timing != settings.timing
            or candidate.reference_period != settings.reference_period
            or candidate.configuration != settings.configuration
        ):
            raise ActionValidationError(
                "Only measurements and schedule can be changed before activation",
                reason="clinical_settings_changed",
            )

    @staticmethod
    def _validate_values(
        action: MoodAction | ActivityAction,
        values: MoodEntryValues,
        *,
        confirmed: bool,
    ) -> None:
        if not isinstance(action, MoodAction):
            raise ActionValidationError(
                "This Action does not accept mood entries", reason="wrong_action_type"
            )
        configuration = action.settings.configuration
        if action.settings.start_date and values.relevant_date < action.settings.start_date:
            raise ActionValidationError(
                "Mood entry is before this Action starts",
                reason="entry_before_action",
            )
        if action.settings.end_date and values.relevant_date > action.settings.end_date:
            raise ActionValidationError(
                "Mood entry is after this Action ends",
                reason="entry_after_action",
            )
        for measurement in ("energy", "sleep_quality", "motivation"):
            if (
                getattr(values, f"{measurement}_value") is not None
                and measurement not in action.settings.what_to_record
            ):
                raise ActionValidationError(
                    "Measurement is not enabled for this Action",
                    reason="unsupported_measurement",
                )
        if configuration.input_kind == "numeric_0_10":
            if values.mood_label is not None:
                raise ActionValidationError(
                    "This Action accepts a numeric mood rating",
                    reason="unsupported_mood_label",
                )
            if values.mood_value is not None and (
                configuration.scale is None or values.scale_id != configuration.scale.id
            ):
                raise ActionValidationError(
                    "Mood rating scale does not match the Action",
                    reason="scale_mismatch",
                )
        else:
            if values.mood_value is not None or values.scale_id is not None:
                raise ActionValidationError(
                    "This Action accepts a mood label",
                    reason="unsupported_mood_rating",
                )
        if confirmed and not any(
            value is not None
            for value in (
                values.mood_value,
                values.mood_label,
                values.context,
                values.thoughts,
                values.actions_taken,
                values.energy_value,
                values.sleep_quality_value,
                values.motivation_value,
                values.custom_feeling,
                values.feelings or None,
            )
        ):
            raise ActionValidationError(
                "Confirmed mood entry cannot be empty",
                reason="empty_confirmed_entry",
            )

    @staticmethod
    def _ensure_revision(existing: MoodEntry | None, expected_revision: int | None) -> None:
        if existing is None:
            if expected_revision is not None:
                raise ActionConflictError("Mood entry no longer matches the reviewed revision")
            return
        if expected_revision is not None and expected_revision != existing.revision:
            raise ActionConflictError("Mood entry changed; review the latest version")

    @staticmethod
    def _next_entry(
        *,
        existing: MoodEntry | None,
        entry_id: UUID,
        action_id: UUID,
        values: MoodEntryValues,
        status: str,
        now: datetime,
        idempotency_key: str | None,
    ) -> MoodEntry:
        history = list(existing.revision_history) if existing is not None else []
        if existing is not None:
            history.append(
                MoodEntryRevision(
                    revision=existing.revision,
                    values=existing.values,
                    status=existing.status,
                    changed_at=existing.updated_at,
                )
            )
        return MoodEntry(
            id=entry_id,
            action_id=action_id,
            status=status,
            values=values,
            revision=1 if existing is None else existing.revision + 1,
            revision_history=history,
            idempotency_key=idempotency_key,
            created_at=now if existing is None else existing.created_at,
            updated_at=now,
            confirmed_at=(
                (existing.confirmed_at if existing is not None else None) or now
                if status == "confirmed"
                else None
            ),
        )

    def _load_current_draft(self, recording_id: UUID) -> CompletedSessionInsights:
        try:
            draft = self.insights_service.load_completed(recording_id)
        except Exception as error:
            raise ActionStorageError("Failed to load session insights") from error
        if draft is None:
            raise ActionNotFoundError("Generate session insights before reviewing Actions")
        return draft

    def _load_action_if_present(self, action_id: UUID) -> MoodAction | ActivityAction | None:
        path = self._action_path(action_id)
        if not path.is_file():
            return None
        return self._load_stored_action(path)

    @staticmethod
    def _load_stored_action(path: Path) -> MoodAction | ActivityAction:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            model = (
                ActivityAction
                if payload.get("type") == "do_enjoyable_or_meaningful_activity"
                else MoodAction
            )
            return model.model_validate(payload)
        except Exception as error:
            raise ActionStorageError("Failed to load Action") from error

    def _load_review(
        self, recording_id: UUID, candidate_id: str
    ) -> MoodCandidateReviewResponse | None:
        path = self._review_path(recording_id, candidate_id)
        if not path.is_file():
            return None
        return self._load_model(path, MoodCandidateReviewResponse, "Failed to load Action review")

    def _load_entry_if_present(self, action_id: UUID, entry_id: UUID) -> MoodEntry | None:
        path = self._entry_path(action_id, entry_id)
        if not path.is_file():
            return None
        return self._load_model(path, MoodEntry, "Failed to load mood entry")

    def _find_active_ongoing_action(self) -> MoodAction | None:
        active = (
            action
            for action in self.list_actions()
            if action.type == "record_mood"
            and action.state == "active"
            and action.settings.duration_kind in ("ongoing", "unscheduled")
        )
        return min(active, key=lambda action: (action.created_at, str(action.id)), default=None)

    @staticmethod
    def _candidate_matches_action(
        candidate: RecordMoodCandidate,
        action: MoodAction,
    ) -> bool:
        candidate_configuration = candidate.configuration
        action_configuration = action.settings.configuration
        if candidate_configuration is None:
            return False
        candidate_scale = candidate_configuration.scale
        action_scale = action_configuration.scale
        return (
            candidate_configuration.input_kind == action_configuration.input_kind
            and candidate_scale == action_scale
        )

    def _review_with_current_action(
        self, review: MoodCandidateReviewResponse
    ) -> MoodCandidateReviewResponse:
        if review.action is None:
            return review
        current = self._load_action_if_present(review.action.id)
        return review.model_copy(update={"action": current}) if current is not None else review

    def _review_paths_for_action(self, action_id: UUID) -> list[Path]:
        if not self.reviews_directory.is_dir():
            return []
        try:
            paths = list(self.reviews_directory.glob("*.json"))
        except OSError as error:
            raise ActionStorageError("Failed to list Action reviews") from error
        linked: list[Path] = []
        for path in paths:
            review = self._load_model(
                path,
                MoodCandidateReviewResponse,
                "Failed to load Action review",
            )
            if review.action is not None and review.action.id == action_id:
                linked.append(path)
        return linked

    def _confirmation_paths_for_action(self, action_id: UUID) -> list[Path]:
        if not self.confirmations_directory.is_dir():
            return []
        try:
            paths = list(self.confirmations_directory.glob("*.json"))
        except OSError as error:
            raise ActionStorageError("Failed to list Action confirmations") from error
        return [
            path
            for path in paths
            if action_id
            in self._load_model(
                path, ActionReviewConfirmationResponse, "Failed to load Action confirmation"
            ).action_ids
        ]

    @staticmethod
    def _load_model(path: Path, model: type, message: str):
        try:
            return model.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise ActionStorageError(message) from error

    def _save_entry(self, entry: MoodEntry) -> None:
        directory = self._entry_directory(entry.action_id)
        directory.mkdir(parents=True, exist_ok=True)
        self._atomic_write(
            self._entry_path(entry.action_id, entry.id),
            entry.model_dump(mode="json"),
        )

    def _action_path(self, action_id: UUID) -> Path:
        return self.actions_directory / f"{action_id}.json"

    def _review_path(self, recording_id: UUID, candidate_id: str) -> Path:
        return self.reviews_directory / f"{recording_id}.{candidate_id}.json"

    def _entry_directory(self, action_id: UUID) -> Path:
        return self.entries_directory / str(action_id)

    def _entry_path(self, action_id: UUID, entry_id: UUID) -> Path:
        return self._entry_directory(action_id) / f"{entry_id}.json"

    def _deletion_path(self, action_id: UUID) -> Path:
        return self.deletions_directory / str(action_id)

    def _recover_pending_deletions(self) -> None:
        if not self.deletions_directory.is_dir():
            return
        try:
            paths = list(self.deletions_directory.iterdir())
        except OSError as error:
            raise ActionStorageError("Failed to inspect pending Action deletions") from error
        for path in paths:
            if not path.is_dir():
                continue
            try:
                action_id = UUID(path.name)
            except ValueError:
                continue
            if self._action_path(action_id).is_file():
                self._restore_staged_deletion(path)
            else:
                try:
                    shutil.rmtree(path)
                except OSError as error:
                    raise ActionStorageError("Failed to finalize Action deletion") from error

    def _restore_staged_deletion(self, deletion_path: Path) -> None:
        destinations = {
            "records": self.actions_directory,
            "reviews": self.reviews_directory,
            "confirmations": self.confirmations_directory,
            "entries": self.entries_directory,
        }
        try:
            for group, destination_directory in destinations.items():
                staged_directory = deletion_path / group
                if not staged_directory.is_dir():
                    continue
                for staged in staged_directory.iterdir():
                    destination = destination_directory / staged.name
                    if destination.exists():
                        raise OSError(f"Cannot restore staged Action data: {destination}")
                    staged.replace(destination)
            shutil.rmtree(deletion_path)
        except OSError as error:
            raise ActionStorageError("Failed to recover Action deletion") from error

    @staticmethod
    def _rollback_staged_deletion(
        moved: list[tuple[Path, Path]], deletion_path: Path
    ) -> OSError | None:
        rollback_error: OSError | None = None
        for source, staged in reversed(moved):
            try:
                if staged.exists():
                    staged.replace(source)
            except OSError as error:
                rollback_error = rollback_error or error
        if rollback_error is None:
            try:
                shutil.rmtree(deletion_path)
            except OSError as error:
                rollback_error = error
        return rollback_error

    @staticmethod
    def _atomic_write(destination: Path, payload: dict[str, object]) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
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
            raise ActionStorageError("Failed to save Action data") from error
