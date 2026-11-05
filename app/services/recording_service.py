from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models import Recording
from app.services.recording_storage import RecordingStorage, RecordingStorageError, SavedRecording

logger = get_logger("services.recordings")


class RecordingNotFoundError(Exception):
    """Raised when recording metadata or its stored audio cannot be found."""


class RecordingPersistenceError(Exception):
    """Raised when recording metadata cannot be saved."""


class RecordingService:
    def __init__(self, storage: RecordingStorage, session: Session) -> None:
        self.storage = storage
        self.session = session

    def save(self, upload: UploadFile) -> SavedRecording:
        saved = self.storage.save(upload)
        recording = Recording(
            id=saved.recording_id,
            storage_key=saved.filename,
            format=Path(saved.filename).suffix.lstrip("."),
            size_bytes=saved.size_bytes,
        )
        self.session.add(recording)

        try:
            self.session.commit()
        except SQLAlchemyError as error:
            self.session.rollback()
            try:
                self.storage.delete(saved.filename)
            except (OSError, RecordingStorageError) as cleanup_error:
                logger.error(
                    "recording_cleanup_failed recording_id=%s error_type=%s",
                    saved.recording_id,
                    type(cleanup_error).__name__,
                )
            raise RecordingPersistenceError("Failed to save recording metadata") from error

        return saved

    def find(self, recording_id: UUID) -> tuple[Recording, Path]:
        recording = self.session.get(Recording, recording_id)
        if recording is None:
            raise RecordingNotFoundError(f"Recording {recording_id} was not found")

        try:
            path = self.storage.resolve(recording.storage_key)
        except RecordingStorageError as error:
            raise RecordingNotFoundError(f"Recording {recording_id} was not found") from error

        return recording, path
