import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.session_insights import (
    InsightBullet,
    InsightFact,
    InsightSection,
    MoodScaleDefinition,
    RecordMoodCandidate,
    RecordMoodConfiguration,
    SessionInsightsResult,
)
from app.services.transcription import RoleAttributionDecision


class FakeTranscriptions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        audio_file = kwargs.pop("file")
        self.calls.append({**kwargs, "audio_bytes": audio_file.read()})
        return SimpleNamespace(
            text="What would be useful today? I want to understand my anxiety.",
            segments=[
                SimpleNamespace(speaker="speaker-a", text="What would be useful today?"),
                SimpleNamespace(speaker="speaker-b", text="I want to understand my anxiety."),
            ],
            languages=[SimpleNamespace(code="en")],
        )


class FakeResponses:
    def __init__(
        self,
        *,
        invalid_insights_count: int = 0,
        invalid_bullet_references: bool = False,
        record_mood_candidate: bool = False,
        inexact_candidate_excerpt: bool = False,
        insights_error: Exception | None = None,
    ) -> None:
        self.invalid_insights_count = invalid_insights_count
        self.invalid_bullet_references = invalid_bullet_references
        self.record_mood_candidate = record_mood_candidate
        self.inexact_candidate_excerpt = inexact_candidate_excerpt
        self.insights_error = insights_error
        self.calls: list[dict[str, Any]] = []
        self.insights_call_count = 0

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs["text_format"] is RoleAttributionDecision:
            return SimpleNamespace(
                output_parsed=RoleAttributionDecision(therapist_speaker="speaker_1"),
                _request_id="role-request-id",
            )

        assert kwargs["text_format"] is SessionInsightsResult
        self.insights_call_count += 1
        if self.insights_error is not None:
            raise self.insights_error

        input_data = json.loads(kwargs["input"][1]["content"].split("\n", 1)[1])
        source_id = "S999" if self.insights_call_count <= self.invalid_insights_count else "S2"
        return SimpleNamespace(
            output_parsed=SessionInsightsResult(
                session_id=input_data["session_id"],
                source_version=input_data["source_version"],
                status="draft",
                coverage_note=None,
                primary_variant="P01",
                secondary_variant=None,
                selection_note="The session focused on understanding an experience.",
                facts=[
                    InsightFact(
                        id="F1",
                        kind="client_report",
                        text="You wanted to understand your anxiety.",
                        source_ids=[source_id],
                    )
                ],
                action_candidates=(
                    [
                        RecordMoodCandidate(
                            id="C1",
                            type="record_mood",
                            state="review_ready",
                            actor="client",
                            exact_task="Record mood",
                            source_excerpt=(
                                "A paraphrase that is not in the transcript."
                                if self.inexact_candidate_excerpt
                                else "I want to understand my anxiety."
                            ),
                            frequency="once daily",
                            timing="evening",
                            reference_period="whole day",
                            start_date="2026-11-10",
                            end_date="2026-11-23",
                            duration_days=14,
                            configuration=RecordMoodConfiguration(
                                input_kind="numeric_0_10",
                                context_enabled=False,
                                scale_origin="moodi_default",
                                scale=MoodScaleDefinition(
                                    id="overall-day-mood-0-10-v1",
                                    construct="overall_day_mood",
                                    minimum=0,
                                    maximum=10,
                                    minimum_label="worst",
                                    maximum_label="best",
                                    direction="low_to_high",
                                    version="1",
                                ),
                            ),
                            unknowns=[],
                            source_ids=["S2"],
                        )
                    ]
                    if self.record_mood_candidate or self.inexact_candidate_excerpt
                    else []
                ),
                sections=[
                    InsightSection(
                        key="today_focus",
                        heading="Today's focus",
                        bullets=[
                            InsightBullet(
                                text="You wanted to understand your anxiety.",
                                fact_ids=["F999"] if self.invalid_bullet_references else ["F1"],
                                candidate_ids=[],
                                module=None,
                            )
                        ],
                    )
                ],
                review_flags=[],
            ),
            _request_id="insights-request-id",
        )


class FakeOpenAIClient:
    def __init__(
        self,
        *,
        invalid_insights_count: int = 0,
        invalid_bullet_references: bool = False,
        record_mood_candidate: bool = False,
        inexact_candidate_excerpt: bool = False,
        insights_error: Exception | None = None,
    ) -> None:
        self.transcriptions = FakeTranscriptions()
        self.audio = SimpleNamespace(transcriptions=self.transcriptions)
        self.responses = FakeResponses(
            invalid_insights_count=invalid_insights_count,
            invalid_bullet_references=invalid_bullet_references,
            record_mood_candidate=record_mood_candidate,
            inexact_candidate_excerpt=inexact_candidate_excerpt,
            insights_error=insights_error,
        )


def upload_recording(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/recordings",
        files={"file": ("session.m4a", b"audio test bytes", "audio/mp4")},
    )
    assert response.status_code == 201
    return response.json()


@contextmanager
def configured_client(
    recordings_dir: Path,
    fake_openai: FakeOpenAIClient,
) -> Iterator[TestClient]:
    settings = Settings(
        recordings_dir=recordings_dir,
        max_upload_mb=30,
        openai_api_key="test-key",
        openai_insights_model="gpt-6-luna",
        openai_insights_reasoning_effort="none",
    )
    with TestClient(create_app(settings, transcription_client=fake_openai)) as client:
        yield client


@pytest.fixture
def fake_openai() -> FakeOpenAIClient:
    return FakeOpenAIClient()


@pytest.fixture
def insights_client(
    recordings_dir: Path,
    fake_openai: FakeOpenAIClient,
) -> Iterator[TestClient]:
    with configured_client(recordings_dir, fake_openai) as client:
        yield client


def test_insights_require_completed_transcript(
    insights_client: TestClient,
    fake_openai: FakeOpenAIClient,
) -> None:
    recording = upload_recording(insights_client)

    response = insights_client.post(f"/recordings/{recording['id']}/insights", json={})

    assert response.status_code == 409
    assert response.json() == {"detail": "Generate a transcript before requesting session insights"}
    assert fake_openai.responses.insights_call_count == 0


def test_insights_are_structured_saved_and_cached(
    insights_client: TestClient,
    recordings_dir: Path,
    fake_openai: FakeOpenAIClient,
) -> None:
    recording = upload_recording(insights_client)
    first_transcription = insights_client.post(f"/recordings/{recording['id']}/transcription")
    cached_transcription = insights_client.post(f"/recordings/{recording['id']}/transcription")
    assert first_transcription.status_code == cached_transcription.status_code == 200
    assert first_transcription.json() == cached_transcription.json()
    assert len(fake_openai.transcriptions.calls) == 1

    endpoint = f"/recordings/{recording['id']}/insights"
    first = insights_client.post(endpoint, json={"session_date": "2026-11-06"})
    cached = insights_client.post(endpoint, json={"session_date": "2026-11-06"})

    assert first.status_code == cached.status_code == 200
    assert first.json() == cached.json()
    payload = first.json()
    assert payload["recording_id"] == recording["id"]
    assert payload["draft_id"].startswith("sha256:")
    assert payload["source_version"].startswith("sha256:")
    assert payload["prompt_version"] == "v0.4.2"
    assert payload["model"] == "gpt-6-luna"
    assert payload["result"]["status"] == "draft"
    assert payload["result"]["sections"][0]["heading"] == "Today's focus"
    assert payload["result"]["sections"][0]["bullets"][0]["item_id"] == "B1"
    assert fake_openai.responses.insights_call_count == 1

    insight_calls = [
        call for call in fake_openai.responses.calls if call["text_format"] is SessionInsightsResult
    ]
    assert len(insight_calls) == 1
    assert insight_calls[0]["store"] is False
    assert insight_calls[0]["model"] == "gpt-6-luna"
    assert insight_calls[0]["reasoning"] == {"effort": "none"}
    source_input = json.loads(insight_calls[0]["input"][1]["content"].split("\n", 1)[1])
    assert source_input["segments"] == [
        {
            "id": "S1",
            "origin": "transcript",
            "role": "therapist",
            "text": "What would be useful today?",
        },
        {
            "id": "S2",
            "origin": "transcript",
            "role": "client",
            "text": "I want to understand my anxiety.",
        },
    ]

    stored = json.loads(
        (recordings_dir / f"{recording['id']}.insights.json").read_text(encoding="utf-8")
    )
    assert stored["request_fingerprint"]
    assert stored["result"] == payload["result"]


def test_invalid_evidence_is_regenerated_once(
    recordings_dir: Path,
) -> None:
    fake_openai = FakeOpenAIClient(invalid_insights_count=1)
    with configured_client(recordings_dir, fake_openai) as client:
        recording = upload_recording(client)
        assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
        response = client.post(f"/recordings/{recording['id']}/insights", json={})

    assert response.status_code == 200
    assert fake_openai.responses.insights_call_count == 2
    assert response.json()["result"]["facts"][0]["source_ids"] == ["S2"]


def test_stale_prompt_cache_is_regenerated(
    insights_client: TestClient,
    recordings_dir: Path,
    fake_openai: FakeOpenAIClient,
) -> None:
    recording = upload_recording(insights_client)
    assert insights_client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
    endpoint = f"/recordings/{recording['id']}/insights"
    first = insights_client.post(endpoint, json={})
    assert first.status_code == 200

    path = recordings_dir / f"{recording['id']}.insights.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["prompt_version"] = "v0.3"
    path.write_text(json.dumps(stored), encoding="utf-8")

    regenerated = insights_client.post(endpoint, json={})

    assert regenerated.status_code == 200
    assert regenerated.json()["prompt_version"] == "v0.4.2"
    assert fake_openai.responses.insights_call_count == 2


def test_repeated_invalid_evidence_is_not_saved(
    recordings_dir: Path,
) -> None:
    fake_openai = FakeOpenAIClient(invalid_insights_count=2)
    with configured_client(recordings_dir, fake_openai) as client:
        recording = upload_recording(client)
        assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
        response = client.post(f"/recordings/{recording['id']}/insights", json={})

    assert response.status_code == 502
    assert response.json() == {"detail": "Could not generate session insights"}
    assert fake_openai.responses.insights_call_count == 2
    assert not (recordings_dir / f"{recording['id']}.insights.json").exists()


def test_invalid_bullet_references_do_not_reject_insights(
    recordings_dir: Path,
) -> None:
    fake_openai = FakeOpenAIClient(invalid_bullet_references=True)
    with configured_client(recordings_dir, fake_openai) as client:
        recording = upload_recording(client)
        assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
        response = client.post(f"/recordings/{recording['id']}/insights", json={})

    assert response.status_code == 200
    assert fake_openai.responses.insights_call_count == 1
    bullet = response.json()["result"]["sections"][0]["bullets"][0]
    assert bullet["fact_ids"] == ["F999"]


def test_inexact_candidate_excerpt_is_rejected(
    recordings_dir: Path,
) -> None:
    fake_openai = FakeOpenAIClient(inexact_candidate_excerpt=True)
    with configured_client(recordings_dir, fake_openai) as client:
        recording = upload_recording(client)
        assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
        response = client.post(f"/recordings/{recording['id']}/insights", json={})

    assert response.status_code == 502
    assert response.json() == {"detail": "Could not generate session insights"}
    assert fake_openai.responses.insights_call_count == 2


def test_provider_failure_does_not_expose_details(
    recordings_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_openai = FakeOpenAIClient(
        insights_error=RuntimeError("sensitive provider failure containing transcript data")
    )
    with configured_client(recordings_dir, fake_openai) as client:
        recording = upload_recording(client)
        assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
        response = client.post(f"/recordings/{recording['id']}/insights", json={})

    assert response.status_code == 502
    assert response.json() == {"detail": "Could not generate session insights"}
    assert "insights_provider_failed" in caplog.text
    assert "sensitive provider failure" not in caplog.text
    assert not (recordings_dir / f"{recording['id']}.insights.json").exists()
