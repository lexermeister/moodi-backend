from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.datastructures import Headers

from app.models import Recording
from app.services.recording_service import RecordingPersistenceError, RecordingService
from app.services.recording_storage import RecordingStorage


def test_upload_saves_actual_bytes_and_metadata(
    client: TestClient,
    recordings_dir: Path,
    database_engine: Engine,
) -> None:
    audio = b"RIFF" + bytes(range(256)) * 8

    response = client.post(
        "/recordings",
        files={"file": ("voice.wav", audio, "audio/wav")},
    )

    assert response.status_code == 201
    payload = response.json()
    recording_id = UUID(payload["id"])
    assert payload == {
        "id": str(recording_id),
        "filename": f"{recording_id}.wav",
        "size_bytes": len(audio),
        "status": "saved",
    }
    assert (recordings_dir / payload["filename"]).read_bytes() == audio

    with Session(database_engine) as session:
        recording = session.get(Recording, recording_id)
        assert recording is not None
        assert recording.storage_key == payload["filename"]
        assert recording.format == "wav"
        assert recording.size_bytes == len(audio)


def test_upload_requires_file(client: TestClient) -> None:
    response = client.post("/recordings")

    assert response.status_code == 422


def test_upload_rejects_empty_file(client: TestClient, recordings_dir: Path) -> None:
    response = client.post(
        "/recordings",
        files={"file": ("empty.m4a", b"", "audio/mp4")},
    )

    assert response.status_code == 400
    assert list(recordings_dir.iterdir()) == []


def test_upload_rejects_oversized_file_and_removes_partial_file(
    client: TestClient,
    recordings_dir: Path,
) -> None:
    response = client.post(
        "/recordings",
        files={"file": ("large.aac", b"a" * (1024 * 1024 + 1), "audio/aac")},
    )

    assert response.status_code == 413
    assert list(recordings_dir.iterdir()) == []


def test_client_filename_cannot_escape_storage_directory(
    client: TestClient,
    recordings_dir: Path,
) -> None:
    audio = b"safe audio bytes"

    response = client.post(
        "/recordings",
        files={"file": ("../../escaped.wav", audio, "audio/wav")},
    )

    assert response.status_code == 201
    payload = response.json()
    saved_path = recordings_dir / payload["filename"]
    assert saved_path.parent == recordings_dir
    assert saved_path.read_bytes() == audio
    assert not (recordings_dir.parent / "escaped.wav").exists()


def test_metadata_failure_removes_finalized_recording(recordings_dir: Path) -> None:
    storage = RecordingStorage(recordings_dir, max_size_bytes=1024)
    storage.prepare()
    session = MagicMock(spec=Session)
    session.commit.side_effect = SQLAlchemyError("database unavailable")
    upload = UploadFile(
        file=BytesIO(b"audio"),
        filename="voice.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )

    with pytest.raises(RecordingPersistenceError):
        RecordingService(storage, session).save(upload)

    assert list(recordings_dir.iterdir()) == []
    session.rollback.assert_called_once()
