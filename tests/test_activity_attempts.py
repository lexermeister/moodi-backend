from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.schemas.actions import ActivityAction, ActivityActionSettings
from app.schemas.activity_attempts import ActivitySelection, SaveActivityAttemptRequest
from app.schemas.session_insights import ActivityMention
from app.services.activity_attempts import (
    ActivityAttemptConflict,
    ActivityAttemptInvalid,
    ActivityAttemptService,
)
from tests.test_activity_actions import candidate
from tests.test_session_insights import FakeOpenAIClient, configured_client, upload_recording


def action(kind: str = "general") -> ActivityAction:
    specific = kind == "specific"
    return ActivityAction(
        id=uuid4(),
        type="do_enjoyable_or_meaningful_activity",
        state="active",
        recording_id=uuid4(),
        transcript_version="source",
        candidate_id="C1",
        source_ids=["S1"],
        source_excerpt="Do something enjoyable this week",
        agreed_frequency="this week",
        agreed_timing=None,
        agreed_reference_period="this week",
        activity_mentions=[ActivityMention(label="Call a friend", source_ids=["S1"])],
        settings=ActivityActionSettings(
            exact_task="Do something enjoyable this week",
            assignment_kind="specific" if specific else "general",
            agreed_activity="Call a friend" if specific else None,
            frequency="this week",
        ),
        created_at=datetime.now(UTC),
        revision=1,
    )


def request(
    selection: ActivitySelection,
    *,
    status: str = "draft",
    revision: int = 0,
    outcome: str | None = None,
) -> SaveActivityAttemptRequest:
    return SaveActivityAttemptRequest(
        action_revision=1,
        expected_revision=revision,
        status=status,
        selection=selection,
        planned_minutes=15,
        outcome=outcome,
    )


def test_attempt_draft_confirmation_retry_and_amendment(tmp_path: Path) -> None:
    active = action()
    service = ActivityAttemptService(tmp_path, SimpleNamespace(load_action=lambda _: active))
    attempt_id = uuid4()
    selection = ActivitySelection(
        origin="session_mention", label="Call a friend", source_ids=["S1"]
    )
    draft_request = request(selection)

    draft = service.save_attempt(active.id, attempt_id, draft_request)
    assert draft.status == "draft" and draft.revision == 1
    assert service.save_attempt(active.id, attempt_id, draft_request) == draft

    own_idea = ActivitySelection(origin="own_idea", label="Read a novel")
    changed_draft = service.save_attempt(active.id, attempt_id, request(own_idea, revision=1))
    assert changed_draft.revision == 2
    assert changed_draft.selection.label == "Read a novel"
    assert changed_draft.history[0].selection.label == "Call a friend"

    confirmed_request = request(
        own_idea, status="confirmed", revision=2, outcome="partly_completed"
    )
    confirmed = service.save_attempt(active.id, attempt_id, confirmed_request)
    assert confirmed.revision == 3 and confirmed.outcome == "partly_completed"
    assert confirmed.history[0].status == "draft"
    assert service.save_attempt(active.id, attempt_id, confirmed_request) == confirmed
    assert service.list_attempts(active.id) == [confirmed]
    assert service.get_attempt(active.id, attempt_id) == confirmed

    corrected = service.save_attempt(
        active.id,
        attempt_id,
        request(own_idea, status="confirmed", revision=3, outcome="completed"),
    )
    assert corrected.revision == 4
    assert corrected.history[-1].outcome == "partly_completed"
    with pytest.raises(ActivityAttemptConflict):
        service.save_attempt(
            active.id,
            attempt_id,
            request(own_idea, status="confirmed", revision=2, outcome="skipped"),
        )


def test_specific_assignment_cannot_be_replaced(tmp_path: Path) -> None:
    active = action("specific")
    service = ActivityAttemptService(tmp_path, SimpleNamespace(load_action=lambda _: active))
    assert service.suggestions(active.id) == []
    with pytest.raises(ActivityAttemptInvalid):
        service.save_attempt(
            active.id,
            uuid4(),
            request(ActivitySelection(origin="own_idea", label="Go swimming")),
        )
    saved = service.save_attempt(
        active.id,
        uuid4(),
        request(ActivitySelection(origin="agreed", label="Call a friend", source_ids=["S1"])),
    )
    assert saved.selection.origin == "agreed"


def test_predefined_and_own_idea_require_honest_sources(tmp_path: Path) -> None:
    active = action()
    service = ActivityAttemptService(tmp_path, SimpleNamespace(load_action=lambda _: active))
    suggestion = service.suggestions(active.id)[0]
    saved = service.save_attempt(
        active.id,
        uuid4(),
        request(
            ActivitySelection(
                origin="predefined", label=suggestion.label, suggestion_id=suggestion.id
            )
        ),
    )
    assert saved.selection.suggestion_id == suggestion.id
    with pytest.raises(ActivityAttemptInvalid):
        service.save_attempt(
            active.id,
            uuid4(),
            request(
                ActivitySelection(origin="predefined", label="Invented", suggestion_id="made_up")
            ),
        )
    own = service.save_attempt(
        active.id, uuid4(), request(ActivitySelection(origin="own_idea", label="Read a novel"))
    )
    assert own.selection.source_ids == []


def test_inactive_action_cannot_start_attempt(tmp_path: Path) -> None:
    inactive = action().model_copy(update={"state": "paused"})
    service = ActivityAttemptService(tmp_path, SimpleNamespace(load_action=lambda _: inactive))
    with pytest.raises(ActivityAttemptConflict):
        service.save_attempt(
            inactive.id, uuid4(), request(ActivitySelection(origin="own_idea", label="Read"))
        )


def test_attempt_requires_real_activity_and_confirmed_outcome() -> None:
    with pytest.raises(ValueError):
        ActivitySelection(origin="own_idea", label="   ")
    with pytest.raises(ValueError):
        request(ActivitySelection(origin="own_idea", label="Read"), status="confirmed")


def test_activity_attempt_api_round_trip(recordings_dir: Path) -> None:
    with configured_client(recordings_dir, FakeOpenAIClient()) as client:
        recording = upload_recording(client)
        item = candidate()
        draft = SimpleNamespace(
            draft_id="draft",
            source_version="source",
            result=SimpleNamespace(action_candidates=[item]),
        )
        client.app.state.action_service.insights_service = SimpleNamespace(
            load_completed=lambda _: draft
        )
        settings = ActivityActionSettings(
            exact_task=item.exact_task,
            assignment_kind="general",
            frequency=item.frequency,
            reference_period=item.reference_period,
        )
        reviewed = client.put(
            f"/recordings/{recording['id']}/action-candidates/C1/review",
            json={
                "draft_id": "draft",
                "source_version": "source",
                "disposition": "activate",
                "settings": settings.model_dump(mode="json"),
            },
        )
        assert reviewed.status_code == 200
        action_id = reviewed.json()["action"]["id"]
        suggestions = client.get(f"/actions/{action_id}/activity-suggestions")
        assert suggestions.status_code == 200
        assert len(suggestions.json()["suggestions"]) == 3
        attempt_id = str(uuid4())
        path = f"/actions/{action_id}/activity-attempts/{attempt_id}"
        payload = request(
            ActivitySelection(origin="own_idea", label="Read a novel"),
            status="confirmed",
            outcome="completed",
        ).model_dump(mode="json")
        first = client.put(path, json=payload)
        assert first.status_code == 200
        assert first.json()["revision"] == 1
        assert client.put(path, json=payload).json() == first.json()
        assert client.get(path).json() == first.json()
        assert client.get(f"/actions/{action_id}/activity-attempts").json()["attempts"] == [
            first.json()
        ]
        assert client.delete(f"/actions/{action_id}").status_code == 204
        assert not (recordings_dir / "actions" / "entries" / action_id).exists()
