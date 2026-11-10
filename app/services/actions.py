import json
import os
import re
from contextlib import suppress
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from app.schemas.actions import (
    AmendMoodEntryRequest,
    ConfirmMoodEntryRequest,
    MoodAction,
    MoodCandidateReviewRequest,
    MoodCandidateReviewResponse,
    MoodEntry,
    MoodEntryRevision,
    MoodEntryValues,
    RecordMoodActionSettings,
    SaveMoodEntryDraftRequest,
)
from app.schemas.session_insights import RecordMoodCandidate
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
        self.entries_directory = self.root / "entries"
        self.insights_service = insights_service

    def prepare(self) -> None:
        self.actions_directory.mkdir(parents=True, exist_ok=True)
        self.reviews_directory.mkdir(parents=True, exist_ok=True)
        self.entries_directory.mkdir(parents=True, exist_ok=True)

    def review_candidate(
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

        existing = self._load_review(recording_id, candidate_id)
        if existing is not None:
            if (
                existing.draft_id != request.draft_id
                or existing.source_version != request.source_version
            ):
                raise ActionConflictError(
                    "The candidate was already reviewed from a different insights draft"
                )
            requested_disposition = "activated" if request.disposition == "activate" else "rejected"
            if existing.disposition != requested_disposition:
                raise ActionConflictError("The candidate has already been reviewed")
            return existing

        reviewed_at = datetime.now(UTC)
        action: MoodAction | None = None
        if request.disposition == "activate":
            settings = request.settings
            if settings is None:
                raise ActionValidationError(
                    "Activation requires reviewed settings",
                    reason="missing_reviewed_settings",
                )
            start_date = settings.start_date or reviewed_at.date()
            settings = settings.model_copy(
                update={
                    "start_date": start_date,
                    "end_date": settings.end_date
                    or start_date + timedelta(days=settings.duration_days - 1),
                }
            )
            settings = RecordMoodActionSettings.model_validate(settings.model_dump())
            self._validate_reviewed_settings(candidate, settings)
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

        review = MoodCandidateReviewResponse(
            recording_id=recording_id,
            candidate_id=candidate_id,
            draft_id=draft.draft_id,
            source_version=draft.source_version,
            disposition="activated" if action is not None else "rejected",
            reviewed_at=reviewed_at,
            action=action,
        )
        self._atomic_write(
            self._review_path(recording_id, candidate_id),
            review.model_dump(mode="json"),
        )
        return review

    def load_candidate_review(
        self, recording_id: UUID, candidate_id: str
    ) -> MoodCandidateReviewResponse:
        review = self._load_review(recording_id, candidate_id)
        if review is None:
            raise ActionNotFoundError("Action candidate review was not found")
        draft = self._load_current_draft(recording_id)
        if review.draft_id != draft.draft_id or review.source_version != draft.source_version:
            raise ActionConflictError("Session insights changed; review the latest candidate")
        return review

    def list_actions(self, recording_id: UUID | None = None) -> list[MoodAction]:
        actions: list[MoodAction] = []
        if not self.actions_directory.is_dir():
            return actions
        try:
            paths = list(self.actions_directory.glob("*.json"))
        except OSError as error:
            raise ActionStorageError("Failed to list Actions") from error
        for path in paths:
            action = self._load_model(path, MoodAction, "Failed to load Action")
            if recording_id is None or action.recording_id == recording_id:
                actions.append(action)
        return sorted(actions, key=lambda item: item.created_at, reverse=True)

    def load_action(self, action_id: UUID) -> MoodAction:
        action = self._load_action_if_present(action_id)
        if action is None:
            raise ActionNotFoundError("Mood Action was not found")
        return action

    def save_draft(
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
        self.load_action(action_id)
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
        self.load_action(action_id)
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
        if candidate.state == "review_ready" and candidate.configuration != settings.configuration:
            raise ActionValidationError(
                "Reviewed mood configuration does not match the session candidate",
                reason="configuration_changed",
            )

    @staticmethod
    def _validate_values(
        action: MoodAction,
        values: MoodEntryValues,
        *,
        confirmed: bool,
    ) -> None:
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
        if values.context is not None and not configuration.context_enabled:
            raise ActionValidationError(
                "Context is not enabled for this Action",
                reason="unsupported_context",
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
            value is not None for value in (values.mood_value, values.mood_label, values.context)
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
            confirmed_at=now if status == "confirmed" else None,
        )

    def _load_current_draft(self, recording_id: UUID) -> CompletedSessionInsights:
        try:
            draft = self.insights_service.load_completed(recording_id)
        except Exception as error:
            raise ActionStorageError("Failed to load session insights") from error
        if draft is None:
            raise ActionNotFoundError("Generate session insights before reviewing Actions")
        return draft

    def _load_action_if_present(self, action_id: UUID) -> MoodAction | None:
        path = self._action_path(action_id)
        if not path.is_file():
            return None
        return self._load_model(path, MoodAction, "Failed to load Action")

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
