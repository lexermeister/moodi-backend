from pathlib import Path
from uuid import UUID

from fastapi.testclient import TestClient


def test_upload_saves_actual_bytes(client: TestClient, recordings_dir: Path) -> None:
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
