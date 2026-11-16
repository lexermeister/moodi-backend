import json
from pathlib import Path
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.services.transcript_store import TranscriptStore
from app.services.transcription import TranscriptionFixtureError, TranscriptionService


def test_fixture_reloads_and_persists_without_provider(tmp_path: Path):
    fixture = tmp_path / "fixture.json"
    service = TranscriptionService(
        TranscriptStore(tmp_path), None, "unused", "unused", fixture_path=fixture
    )
    recording_id = uuid4()
    turns = [
        {"role": "therapist", "text": "Record mood?"},
        {"role": "client", "text": "Yes, mood and motivation."},
    ]
    fixture.write_text(json.dumps({"turns": turns}))
    first = service.transcribe(recording_id, tmp_path / "unused.m4a")
    assert first.model == "development-fixture"
    assert {turn.source_speaker for turn in first.turns} == {"speaker_1", "speaker_2"}
    turns[1]["text"] = "Yes, mood and sleep quality."
    fixture.write_text(json.dumps({"turns": turns}))
    second = service.transcribe(recording_id, tmp_path / "unused.m4a")
    assert second.text != first.text
    assert service.transcript_store.load(recording_id) == second
    fixture.unlink()
    with pytest.raises(TranscriptionFixtureError):
        service.transcribe(recording_id, tmp_path / "unused.m4a")


@pytest.mark.parametrize(
    "payload",
    [
        "broken json",
        "{}",
        '{"turns": []}',
        '{"turns": [{"role": "client", "text": "alone"}]}',
        '{"turns": [{"role": "therapist", "text": " "}]}',
    ],
)
def test_invalid_fixture_fails_without_provider(tmp_path: Path, payload):
    fixture = tmp_path / "fixture.json"
    fixture.write_text(payload)
    service = TranscriptionService(
        TranscriptStore(tmp_path), None, "unused", "unused", fixture_path=fixture
    )
    with pytest.raises(TranscriptionFixtureError):
        service.transcribe(uuid4(), tmp_path / "unused.m4a")


def test_fixture_not_available_in_production():
    settings = Settings(
        _env_file=None, app_env="production", transcription_fixture_path="test.json"
    )
    with pytest.raises(ValueError, match="only available in development"):
        _ = settings.fixture_path
