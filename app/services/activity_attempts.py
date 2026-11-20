import json
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4

from app.schemas.actions import ActivityAction
from app.schemas.activity_attempts import (
    ActivityAttempt,
    ActivityAttemptData,
    ActivityAttemptVersion,
    ActivitySelection,
    ActivitySuggestion,
    SaveActivityAttemptRequest,
)
from app.services.actions import ActionService

SUGGESTIONS = (
    ActivitySuggestion(
        id="listen_to_an_album", label="Listen to an album", source="moodi_prototype_v1"
    ),
    ActivitySuggestion(
        id="spend_a_few_minutes_drawing",
        label="Spend a few minutes drawing",
        source="moodi_prototype_v1",
    ),
    ActivitySuggestion(
        id="call_someone_you_enjoy_talking_to",
        label="Call someone you enjoy talking to",
        source="moodi_prototype_v1",
    ),
)


class ActivityAttemptError(Exception):
    """Base error for activity attempt access."""


class ActivityAttemptNotFound(ActivityAttemptError):
    """The requested attempt does not exist."""


class ActivityAttemptConflict(ActivityAttemptError):
    """The action or attempt changed since the client loaded it."""


class ActivityAttemptInvalid(ActivityAttemptError):
    """A selection is not supported by the reviewed action."""


class ActivityAttemptStorageError(ActivityAttemptError):
    """An attempt could not be read or saved durably."""


class ActivityAttemptService:
    def __init__(self, directory: Path, action_service: ActionService) -> None:
        self.root = directory / "actions" / "entries"
        self.action_service = action_service
        self._lock = RLock()

    def suggestions(self, action_id: UUID) -> list[ActivitySuggestion]:
        action = self._activity_action(action_id)
        if action.state != "active" or action.settings.assignment_kind != "general":
            return []
        return list(SUGGESTIONS)

    def list_attempts(self, action_id: UUID) -> list[ActivityAttempt]:
        self._activity_action(action_id)
        directory = self.root / str(action_id)
        if not directory.is_dir():
            return []
        try:
            attempts = [
                self._read(path, action_id, UUID(path.name.split(".")[0]))
                for path in directory.glob("*.activity.json")
            ]
        except (OSError, ValueError) as error:
            raise ActivityAttemptStorageError("Could not list activity attempts") from error
        return sorted(attempts, key=lambda item: item.updated_at, reverse=True)

    def get_attempt(self, action_id: UUID, attempt_id: UUID) -> ActivityAttempt:
        self._activity_action(action_id)
        path = self._path(action_id, attempt_id)
        if not path.is_file():
            raise ActivityAttemptNotFound("Activity attempt was not found")
        return self._read(path, action_id, attempt_id)

    def save_attempt(
        self, action_id: UUID, attempt_id: UUID, request: SaveActivityAttemptRequest
    ) -> ActivityAttempt:
        with self._lock:
            action = self._activity_action(action_id)
            if action.revision != request.action_revision:
                raise ActivityAttemptConflict("The activity Action changed; reload before saving")
            path = self._path(action_id, attempt_id)
            existing = self._read(path, action_id, attempt_id) if path.is_file() else None
            if existing is None and action.state != "active":
                raise ActivityAttemptConflict("Only active activity Actions can start an attempt")
            self._validate_selection(action, request.selection)
            values = request.model_dump(exclude={"expected_revision", "action_revision"})
            if existing is not None:
                current_values = existing.model_dump(include=set(ActivityAttemptData.model_fields))
                if values == current_values:
                    return existing
                if existing.revision != request.expected_revision:
                    raise ActivityAttemptConflict("Activity attempt changed; reload before saving")
                if existing.status == "confirmed" and request.status == "draft":
                    raise ActivityAttemptConflict("A confirmed attempt cannot become a draft")
                history = [
                    *existing.history,
                    ActivityAttemptVersion(
                        **current_values, revision=existing.revision, saved_at=existing.updated_at
                    ),
                ]
                created_at = existing.created_at
                revision = existing.revision + 1
            else:
                if request.expected_revision != 0:
                    raise ActivityAttemptConflict(
                        "Activity attempt does not exist at that revision"
                    )
                history = []
                created_at = datetime.now(UTC)
                revision = 1
            now = datetime.now(UTC)
            attempt = ActivityAttempt(
                **values,
                id=attempt_id,
                action_id=action_id,
                action_revision=action.revision,
                revision=revision,
                created_at=created_at,
                updated_at=now,
                history=history,
            )
            self._write(path, attempt)
            return attempt

    def _activity_action(self, action_id: UUID) -> ActivityAction:
        action = self.action_service.load_action(action_id)
        if not isinstance(action, ActivityAction):
            raise ActivityAttemptInvalid("This is not an enjoyable activity Action")
        return action

    @staticmethod
    def _validate_selection(action: ActivityAction, selection: ActivitySelection) -> None:
        if action.settings.assignment_kind == "specific":
            if (
                selection.origin != "agreed"
                or selection.label != action.settings.agreed_activity
                or selection.source_ids != action.source_ids
                or selection.suggestion_id is not None
            ):
                raise ActivityAttemptInvalid("Use the specific activity agreed in the session")
            return
        if selection.origin == "agreed":
            raise ActivityAttemptInvalid("The session did not specify an activity")
        if selection.origin == "session_mention":
            if selection.suggestion_id is not None or not any(
                selection.label == mention.label and selection.source_ids == mention.source_ids
                for mention in action.activity_mentions
            ):
                raise ActivityAttemptInvalid("The activity was not mentioned in this session")
        elif selection.origin == "predefined":
            if selection.source_ids or not any(
                selection.suggestion_id == item.id and selection.label == item.label
                for item in SUGGESTIONS
            ):
                raise ActivityAttemptInvalid("Choose an available activity suggestion")
        elif selection.origin == "own_idea" and (
            selection.source_ids or selection.suggestion_id is not None
        ):
            raise ActivityAttemptInvalid("An own idea cannot claim a suggestion source")

    def _path(self, action_id: UUID, attempt_id: UUID) -> Path:
        return self.root / str(action_id) / f"{attempt_id}.activity.json"

    @staticmethod
    def _read(path: Path, action_id: UUID, attempt_id: UUID) -> ActivityAttempt:
        try:
            result = ActivityAttempt.model_validate_json(path.read_text(encoding="utf-8"))
            if result.action_id != action_id or result.id != attempt_id:
                raise ValueError("Activity attempt identity mismatch")
            return result
        except Exception as error:
            raise ActivityAttemptStorageError("Could not read activity attempt") from error

    @staticmethod
    def _write(path: Path, attempt: ActivityAttempt) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("x", encoding="utf-8") as output:
                json.dump(attempt.model_dump(mode="json"), output, ensure_ascii=False)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(path)
            with suppress(OSError):
                directory_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except Exception as error:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            raise ActivityAttemptStorageError("Could not save activity attempt") from error
