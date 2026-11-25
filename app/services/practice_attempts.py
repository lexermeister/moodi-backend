import json
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4

from app.schemas.actions import PracticeAction
from app.schemas.practice_attempts import (
    PracticeAttempt,
    PracticeAttemptData,
    PracticeAttemptVersion,
    SavePracticeAttemptRequest,
)
from app.services.actions import ActionService


class PracticeAttemptError(Exception):
    """Base error for practice attempt access."""


class PracticeAttemptNotFound(PracticeAttemptError):
    """The requested attempt does not exist."""


class PracticeAttemptConflict(PracticeAttemptError):
    """The action or attempt changed since the client loaded it."""


class PracticeAttemptInvalid(PracticeAttemptError):
    """The Action does not accept practice attempts."""


class PracticeAttemptStorageError(PracticeAttemptError):
    """An attempt could not be read or saved durably."""


class PracticeAttemptService:
    def __init__(self, directory: Path, action_service: ActionService) -> None:
        self.root = directory / "actions" / "entries"
        self.action_service = action_service
        self._lock = RLock()

    def list_attempts(self, action_id: UUID) -> list[PracticeAttempt]:
        self._practice_action(action_id)
        directory = self.root / str(action_id)
        if not directory.is_dir():
            return []
        try:
            attempts = [
                self._read(path, action_id, UUID(path.name.split(".")[0]))
                for path in directory.glob("*.practice.json")
            ]
        except (OSError, ValueError) as error:
            raise PracticeAttemptStorageError("Could not list practice attempts") from error
        return sorted(attempts, key=lambda item: item.updated_at, reverse=True)

    def get_attempt(self, action_id: UUID, attempt_id: UUID) -> PracticeAttempt:
        self._practice_action(action_id)
        path = self._path(action_id, attempt_id)
        if not path.is_file():
            raise PracticeAttemptNotFound("Practice attempt was not found")
        return self._read(path, action_id, attempt_id)

    def save_attempt(
        self, action_id: UUID, attempt_id: UUID, request: SavePracticeAttemptRequest
    ) -> PracticeAttempt:
        with self._lock:
            action = self._practice_action(action_id)
            if action.revision != request.action_revision:
                raise PracticeAttemptConflict("The practice Action changed; reload before saving")
            path = self._path(action_id, attempt_id)
            existing = self._read(path, action_id, attempt_id) if path.is_file() else None
            if existing is None and action.state != "active":
                raise PracticeAttemptConflict("Only active practice Actions can start an attempt")
            values = request.model_dump(exclude={"expected_revision", "action_revision"})
            if existing is not None:
                current_values = existing.model_dump(include=set(PracticeAttemptData.model_fields))
                if values == current_values:
                    return existing
                if existing.revision != request.expected_revision:
                    raise PracticeAttemptConflict("Practice attempt changed; reload before saving")
                if existing.status == "confirmed" and request.status == "draft":
                    raise PracticeAttemptConflict("A confirmed attempt cannot become a draft")
                history = [
                    *existing.history,
                    PracticeAttemptVersion(
                        **current_values, revision=existing.revision, saved_at=existing.updated_at
                    ),
                ]
                created_at = existing.created_at
                revision = existing.revision + 1
            else:
                if request.expected_revision != 0:
                    raise PracticeAttemptConflict(
                        "Practice attempt does not exist at that revision"
                    )
                history = []
                created_at = datetime.now(UTC)
                revision = 1
            now = datetime.now(UTC)
            attempt = PracticeAttempt(
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

    def _practice_action(self, action_id: UUID) -> PracticeAction:
        action = self.action_service.load_action(action_id)
        if not isinstance(action, PracticeAction):
            raise PracticeAttemptInvalid("This is not a relaxation/breathing practice Action")
        return action

    def _path(self, action_id: UUID, attempt_id: UUID) -> Path:
        return self.root / str(action_id) / f"{attempt_id}.practice.json"

    @staticmethod
    def _read(path: Path, action_id: UUID, attempt_id: UUID) -> PracticeAttempt:
        try:
            result = PracticeAttempt.model_validate_json(path.read_text(encoding="utf-8"))
            if result.action_id != action_id or result.id != attempt_id:
                raise ValueError("Practice attempt identity mismatch")
            return result
        except Exception as error:
            raise PracticeAttemptStorageError("Could not read practice attempt") from error

    @staticmethod
    def _write(path: Path, attempt: PracticeAttempt) -> None:
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
            raise PracticeAttemptStorageError("Could not save practice attempt") from error
