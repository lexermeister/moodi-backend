from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.schemas.actions import PracticeActionSettings
from app.schemas.practice_attempts import SavePracticeAttemptRequest
from app.schemas.session_insights import PracticeCandidate
from app.services.practice_attempts import PracticeAttemptConflict, PracticeAttemptService
from tests.test_session_insights import FakeOpenAIClient, configured_client, upload_recording


def candidate(**changes):
    return PracticeCandidate.model_validate(
        {
            "id": "C1",
            "type": "practise_relaxation_breathing",
            "state": "review_ready",
            "actor": "client",
            "exact_task": "Repeat the breathing practice we did in session",
            "technique": "Breathing practice from session",
            "source_excerpt": "Repeat the breathing practice",
            "source_ids": ["S1"],
            **changes,
        }
    )


def test_review_confirmation_attempt_api_and_history(recordings_dir: Path):
    item = candidate()
    with configured_client(recordings_dir, FakeOpenAIClient()) as client:
        recording = upload_recording(client)
        client.app.state.action_service.insights_service = SimpleNamespace(
            load_completed=lambda _: SimpleNamespace(
                draft_id="draft",
                source_version="source",
                result=SimpleNamespace(action_candidates=[item]),
            )
        )
        review_path = f"/recordings/{recording['id']}/action-candidates/C1"
        assert client.get(f"{review_path}/resolution").json()["status"] == "needs_setup"
        settings = PracticeActionSettings(exact_task=item.exact_task, technique=item.technique)
        review = {
            "draft_id": "draft",
            "source_version": "source",
            "disposition": "activate",
            "settings": settings.model_dump(mode="json"),
        }
        response = client.put(f"{review_path}/review", json=review)
        assert response.status_code == 200, response.text
        action = response.json()["action"]
        assert action["settings"]["frequency"] == "Every day"
        assert action["settings"]["reminder_time"] is None
        assert client.put(f"{review_path}/review", json=review).json()["action"] == action
        assert client.get(f"{review_path}/resolution").json()["status"] == "activated"
        assert client.get(f"/actions/{action['id']}").json() == action
        assert client.get("/actions").json()["actions"] == [action]
        assert client.put(
            f"/recordings/{recording['id']}/actions/confirmation",
            json={"draft_id": "draft", "source_version": "source"},
        ).json()["action_ids"] == [action["id"]]
        attempt_id = str(uuid4())
        path = f"/actions/{action['id']}/practice-attempts/{attempt_id}"
        payload = {
            "action_revision": 1,
            "expected_revision": 0,
            "status": "draft",
            "relevant_date": "2026-11-25",
        }
        draft = client.put(path, json=payload)
        assert draft.status_code == 200, draft.text
        assert client.put(path, json=payload).json() == draft.json()
        payload.update(
            status="confirmed",
            outcome="stopped_early",
            tension_before=0,
            tension_after=8,
            expected_revision=1,
        )
        saved = client.put(path, json=payload)
        assert saved.status_code == 200, saved.text
        assert saved.json()["actual_minutes"] is None
        assert saved.json()["tension_before"] == 0
        assert client.put(path, json=payload).json() == saved.json()
        assert client.get(path).json() == saved.json()
        assert client.get(f"/actions/{action['id']}/practice-attempts").json()["attempts"] == [
            saved.json()
        ]
        payload.update(notes="Interrupted", expected_revision=2)
        amended = client.put(path, json=payload).json()
        assert amended["history"][-1]["tension_after"] == 8
        assert amended["revision"] == 3
        payload.update(notes="stale", expected_revision=1)
        assert client.put(path, json=payload).status_code == 409
        assert client.delete(f"/actions/{action['id']}").status_code == 204
        assert client.get(path).status_code == 404


@pytest.mark.parametrize("outcome", ["completed", "partly_completed", "skipped", "stopped_early"])
def test_optional_values_and_outcomes(outcome):
    attempt = SavePracticeAttemptRequest(
        action_revision=1,
        expected_revision=0,
        status="confirmed",
        outcome=outcome,
        relevant_date="2026-11-25",
    )
    assert attempt.tension_before is None and attempt.notes is None
    with pytest.raises(ValidationError):
        SavePracticeAttemptRequest.model_validate({**attempt.model_dump(), "tension_after": 11})


def test_timer_does_not_confirm_practice():
    with pytest.raises(ValidationError):
        SavePracticeAttemptRequest(
            action_revision=1,
            expected_revision=0,
            status="confirmed",
            relevant_date="2026-11-25",
            timer_elapsed_seconds=180,
        )


def test_inactive_actions_cannot_start_attempt(tmp_path):
    from app.schemas.actions import MoodCandidateReviewRequest
    from app.services.actions import ActionService

    item = candidate()
    service = ActionService(
        tmp_path,
        SimpleNamespace(
            load_completed=lambda _: SimpleNamespace(
                draft_id="draft",
                source_version="source",
                result=SimpleNamespace(action_candidates=[item]),
            )
        ),
    )
    active = service.review_candidate(
        uuid4(),
        item.id,
        MoodCandidateReviewRequest(
            draft_id="draft",
            source_version="source",
            disposition="activate",
            settings=PracticeActionSettings(exact_task=item.exact_task, technique=item.technique),
        ),
    ).action
    inactive = active.model_copy(update={"state": "paused"})
    attempts = PracticeAttemptService(tmp_path, SimpleNamespace(load_action=lambda _: inactive))
    with pytest.raises(PracticeAttemptConflict):
        attempts.save_attempt(
            active.id,
            uuid4(),
            SavePracticeAttemptRequest(
                action_revision=1, expected_revision=0, status="draft", relevant_date="2026-11-25"
            ),
        )


def test_mocked_provider_practice_survives_generation_and_cache(recordings_dir: Path):
    import json

    from app.schemas.session_insights import SessionInsightsResult

    fake = FakeOpenAIClient()
    original_parse = fake.responses.parse
    commitment = "I will repeat the breathing practice we did in session."
    fake.transcriptions.create = lambda **_: SimpleNamespace(
        text=commitment,
        segments=[
            SimpleNamespace(speaker="a", text="What will you practice?"),
            SimpleNamespace(speaker="b", text=commitment),
        ],
        languages=[SimpleNamespace(code="en")],
    )

    def parse(**kwargs):
        response = original_parse(**kwargs)
        if kwargs["text_format"] is SessionInsightsResult:
            response.output_parsed.action_candidates = [
                candidate(exact_task=commitment, source_excerpt=commitment, source_ids=["S2"])
            ]
        return response

    fake.responses.parse = parse
    with configured_client(recordings_dir, fake) as client:
        recording = upload_recording(client)
        assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
        path = f"/recordings/{recording['id']}/insights"
        generated = client.post(path, json={})
        assert generated.status_code == 200, generated.text
        assert (
            generated.json()["result"]["action_candidates"][0]["type"]
            == "practise_relaxation_breathing"
        )
        assert client.post(path, json={}).json() == generated.json()
        assert fake.responses.insights_call_count == 1
        # Existing session drafts remain readable after the prompt version change.
        stored_path = recordings_dir / f"{recording['id']}.insights.json"
        stored = json.loads(stored_path.read_text())
        stored["prompt_version"] = "v0.6.4"
        stored_path.write_text(json.dumps(stored))
        assert (
            client.app.state.session_insights_service.load_completed(
                UUID(recording["id"])
            ).prompt_version
            == "v0.6.4"
        )
        assert client.post(path, json={}).json()["prompt_version"] == "v0.8.0"


def test_failed_durable_write_preserves_previous_draft(tmp_path, monkeypatch):
    from app.schemas.actions import MoodCandidateReviewRequest
    from app.services.actions import ActionService
    from app.services.practice_attempts import PracticeAttemptStorageError

    item = candidate()
    actions = ActionService(
        tmp_path,
        SimpleNamespace(
            load_completed=lambda _: SimpleNamespace(
                draft_id="draft",
                source_version="source",
                result=SimpleNamespace(action_candidates=[item]),
            )
        ),
    )
    active = actions.review_candidate(
        uuid4(),
        item.id,
        MoodCandidateReviewRequest(
            draft_id="draft",
            source_version="source",
            disposition="activate",
            settings=PracticeActionSettings(exact_task=item.exact_task, technique=item.technique),
        ),
    ).action
    service = PracticeAttemptService(tmp_path, actions)
    attempt_id = uuid4()
    request = SavePracticeAttemptRequest(
        action_revision=1, expected_revision=0, status="draft", relevant_date="2026-11-25"
    )
    draft = service.save_attempt(active.id, attempt_id, request)

    def fail(_):
        raise OSError("Disk failure")

    monkeypatch.setattr("app.services.practice_attempts.os.fsync", fail)
    with pytest.raises(PracticeAttemptStorageError):
        service.save_attempt(
            active.id,
            attempt_id,
            request.model_copy(update={"expected_revision": 1, "notes": "new answer"}),
        )
    assert service.get_attempt(active.id, attempt_id) == draft
    assert not list((tmp_path / "actions" / "entries" / str(active.id)).glob(".*.tmp"))
