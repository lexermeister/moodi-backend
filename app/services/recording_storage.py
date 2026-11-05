import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import UploadFile

CHUNK_SIZE_BYTES = 1024 * 1024

ALLOWED_CONTENT_TYPES_BY_EXTENSION = {
    ".aac": {"audio/aac", "audio/x-aac", "audio/vnd.dlna.adts"},
    ".m4a": {"audio/m4a", "audio/mp4", "audio/x-m4a"},
    ".wav": {"audio/vnd.wave", "audio/wav", "audio/wave", "audio/x-wav"},
}


class RecordingStorageError(Exception):
    """Base error raised while validating or storing an upload."""


class InvalidRecordingTypeError(RecordingStorageError):
    """Raised when an upload fails extension or MIME screening."""


class EmptyRecordingError(RecordingStorageError):
    """Raised when an upload contains no bytes."""


class RecordingTooLargeError(RecordingStorageError):
    """Raised when an upload exceeds the configured limit."""


@dataclass(frozen=True)
class SavedRecording:
    recording_id: UUID
    filename: str
    size_bytes: int


class RecordingStorage:
    def __init__(self, directory: Path, max_size_bytes: int) -> None:
        self.directory = directory
        self.max_size_bytes = max_size_bytes

    def prepare(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, upload: UploadFile) -> SavedRecording:
        extension = self._validated_extension(upload)
        recording_id = uuid4()
        filename = f"{recording_id}{extension}"
        destination = self.directory / filename
        temporary = self.directory / f".{recording_id}.tmp"
        size_bytes = 0

        try:
            with temporary.open("xb") as output:
                while chunk := upload.file.read(CHUNK_SIZE_BYTES):
                    size_bytes += len(chunk)
                    if size_bytes > self.max_size_bytes:
                        raise RecordingTooLargeError(
                            f"Recording exceeds the {self.max_size_bytes}-byte upload limit"
                        )
                    output.write(chunk)

                if size_bytes == 0:
                    raise EmptyRecordingError("Recording must not be empty")

                output.flush()
                os.fsync(output.fileno())

            temporary.replace(destination)
            self._fsync_directory()
        except (EmptyRecordingError, RecordingTooLargeError):
            self._remove_partial_file(temporary)
            raise
        except Exception as error:
            self._remove_partial_file(temporary)
            self._remove_partial_file(destination)
            raise RecordingStorageError("Failed to save recording") from error

        return SavedRecording(
            recording_id=recording_id,
            filename=filename,
            size_bytes=size_bytes,
        )

    def resolve(self, storage_key: str) -> Path:
        path = (self.directory / storage_key).resolve()
        if path.parent != self.directory.resolve() or path.name != storage_key:
            raise RecordingStorageError("Invalid recording storage key")
        if not path.is_file():
            raise RecordingStorageError("Stored recording file was not found")
        return path

    def delete(self, storage_key: str) -> None:
        path = (self.directory / storage_key).resolve()
        if path.parent != self.directory.resolve() or path.name != storage_key:
            raise RecordingStorageError("Invalid recording storage key")
        path.unlink(missing_ok=True)
        self._fsync_directory()

    def _fsync_directory(self) -> None:
        directory_fd = os.open(self.directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @staticmethod
    def _remove_partial_file(path: Path) -> None:
        # Preserve the original storage failure if cleanup itself also fails.
        with suppress(OSError):
            path.unlink(missing_ok=True)

    @staticmethod
    def _validated_extension(upload: UploadFile) -> str:
        extension = Path(upload.filename or "").suffix.lower()
        content_type = (upload.content_type or "").split(";", maxsplit=1)[0].strip().lower()
        allowed_content_types = ALLOWED_CONTENT_TYPES_BY_EXTENSION.get(extension)

        if allowed_content_types is None or content_type not in allowed_content_types:
            raise InvalidRecordingTypeError(
                "Only M4A, AAC, and WAV uploads with a matching audio MIME type are supported"
            )

        return extension
