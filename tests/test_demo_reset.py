from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_restart_demo_clears_generated_data_and_preserves_fixture(tmp_path: Path) -> None:
    directory = tmp_path / "recordings"
    directory.mkdir()
    recording_id = uuid4()
    fixture_id = uuid4()
    generated = [
        directory / f"{recording_id}.m4a",
        directory / f"{recording_id}.transcript.json",
        directory / f"{recording_id}.insights.json",
        directory / f"{recording_id}.insights-review.json",
        directory / f".{recording_id}.transcript.tmp",
    ]
    for path in generated:
        path.write_text("test data")
    actions = directory / "actions" / "entries" / str(uuid4())
    actions.mkdir(parents=True)
    (actions / "entry.json").write_text("test data")
    fixture = directory / f"{fixture_id}.transcript.json"
    fixture.write_text("fixture")
    unrelated = directory / "notes.txt"
    unrelated.write_text("keep")

    settings = Settings(recordings_dir=directory, transcription_fixture_path=fixture)
    with TestClient(create_app(settings)) as client:
        assert client.delete("/demo/data").status_code == 204
        assert client.delete("/demo/data").status_code == 204
        assert client.get("/actions").json() == {"actions": []}
        assert client.post(f"/recordings/{recording_id}/transcription").status_code == 404

    assert all(not path.exists() for path in generated)
    assert not (directory / "actions").exists()
    assert fixture.read_text() == "fixture"
    assert unrelated.read_text() == "keep"


def test_restart_demo_is_unavailable_outside_development(tmp_path: Path) -> None:
    directory = tmp_path / "recordings"
    directory.mkdir()
    recording = directory / f"{uuid4()}.m4a"
    recording.write_bytes(b"audio")
    with TestClient(create_app(Settings(app_env="production", recordings_dir=directory))) as client:
        assert client.delete("/demo/data").status_code == 404
    assert recording.read_bytes() == b"audio"
