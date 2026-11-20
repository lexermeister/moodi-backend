from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.schemas.actions import (
    ActionReviewConfirmationRequest,
    ActivityActionSettings,
    MoodCandidateReviewRequest,
)
from app.schemas.session_insights import (
    ActivityMention,
    EnjoyableActivityCandidate,
    InsightBullet,
    InsightFact,
    InsightSection,
    SessionInsightsResult,
)
from app.services.actions import ActionConflictError, ActionService, ActionValidationError
from app.services.session_insights import InsightsValidationError, SessionInsightsService
from tests.test_session_insights import FakeOpenAIClient, configured_client, upload_recording


def candidate(kind: str = "general") -> EnjoyableActivityCandidate:
    return EnjoyableActivityCandidate(
        id="C1",
        type="do_enjoyable_or_meaningful_activity",
        state="review_ready",
        actor="client",
        exact_task="Do something enjoyable this week",
        assignment_kind=kind,
        agreed_activity="Call a friend" if kind == "specific" else None,
        activity_mentions=[ActivityMention(label="Call a friend", source_ids=["S1"])],
        frequency="this week",
        timing=None,
        reference_period="this week",
        source_excerpt="Do something enjoyable this week",
        source_ids=["S1", "S2"],
    )


def result(item: EnjoyableActivityCandidate) -> SessionInsightsResult:
    return SessionInsightsResult(
        session_id="session",
        source_version="source",
        status="draft",
        coverage_note=None,
        primary_variant="P01",
        secondary_variant=None,
        selection_note="Activity planning",
        facts=[
            InsightFact(id="F1", kind="decision", text="Activity agreed", source_ids=["S1", "S2"])
        ],
        action_candidates=[item],
        sections=[
            InsightSection(
                key="agreements",
                heading="What was agreed",
                bullets=[
                    InsightBullet(
                        text="Do an activity", fact_ids=["F1"], candidate_ids=["C1"], module=None
                    ),
                ],
            )
        ],
        review_flags=[],
    )


def test_activity_candidate_requires_real_mention_evidence() -> None:
    item = candidate()
    SessionInsightsService._validate(
        result(item),
        {
            "session_id": "session",
            "source_version": "source",
            "segments": [
                {"id": "S1", "text": "Do something enjoyable this week; call a friend"},
                {"id": "S2", "text": "Yes, I will"},
            ],
        },
    )
    separate_mention = item.model_copy(
        update={"activity_mentions": [ActivityMention(label="Talk to someone", source_ids=["S3"])]}
    )
    SessionInsightsService._validate(
        result(separate_mention),
        {
            "session_id": "session",
            "source_version": "source",
            "segments": [
                {"id": "S1", "text": "Do something enjoyable this week"},
                {"id": "S2", "text": "Yes, I will"},
                {"id": "S3", "text": "I could call a friend"},
            ],
        },
    )
    unsupported = item.model_copy(
        update={
            "activity_mentions": [ActivityMention(label="Call a friend", source_ids=["S9"])],
        }
    )
    with pytest.raises(InsightsValidationError, match="mention"):
        SessionInsightsService._validate(
            result(unsupported),
            {
                "session_id": "session",
                "source_version": "source",
                "segments": [
                    {"id": "S1", "text": "Do something enjoyable this week; call a friend"},
                    {"id": "S2", "text": "Yes, I will"},
                ],
            },
        )
    cleaned, omitted = SessionInsightsService._omit_unlinked_activity_mentions(
        result(unsupported),
        {"segments": [{"id": "S1"}, {"id": "S2"}]},
    )
    assert omitted == 1
    assert cleaned.action_candidates[0].activity_mentions == []


def test_duplicate_activity_suggestions_are_collapsed() -> None:
    first = candidate()
    duplicate = first.model_copy(update={"id": "C2", "exact_task": "Make time for enjoyment"})
    generated = result(first)
    generated = generated.model_copy(
        update={
            "action_candidates": [first, duplicate],
            "sections": [
                generated.sections[0].model_copy(
                    update={
                        "bullets": [
                            generated.sections[0]
                            .bullets[0]
                            .model_copy(update={"candidate_ids": ["C1", "C2"]})
                        ]
                    }
                )
            ],
        }
    )
    cleaned, omitted = SessionInsightsService._omit_unsupported_activity_candidates(
        generated,
        {
            "segments": [
                {"id": "S1", "text": "Do something enjoyable this week"},
                {"id": "S2", "text": "Yes, I will"},
            ]
        },
    )
    assert omitted == 1
    assert [item.id for item in cleaned.action_candidates] == ["C1"]
    assert cleaned.sections[0].bullets[0].candidate_ids == ["C1"]
    assert not cleaned.review_flags


def test_specific_activity_can_be_paraphrased_when_source_quote_is_exact() -> None:
    item = candidate("specific").model_copy(
        update={
            "source_excerpt": "I’ll phone a friend this week",
            "activity_mentions": [],
        }
    )
    SessionInsightsService._validate(
        result(item),
        {
            "session_id": "session",
            "source_version": "source",
            "segments": [
                {"id": "S1", "text": "How about contacting a friend?"},
                {"id": "S2", "text": "I'll phone a friend this week."},
            ],
        },
    )


def test_source_quote_still_requires_matching_words() -> None:
    item = candidate().model_copy(update={"source_excerpt": "I agreed to go swimming"})
    with pytest.raises(InsightsValidationError, match="excerpt"):
        SessionInsightsService._validate(
            result(item),
            {
                "session_id": "session",
                "source_version": "source",
                "segments": [
                    {"id": "S1", "text": "Do something enjoyable this week"},
                    {"id": "S2", "text": "Yes, I will"},
                ],
            },
        )
    cleaned, omitted = SessionInsightsService._omit_unsupported_activity_candidates(
        result(item),
        {
            "segments": [
                {"id": "S1", "text": "Do something enjoyable this week"},
                {"id": "S2", "text": "Yes, I will"},
            ]
        },
    )
    assert omitted == 1
    assert cleaned.action_candidates == []
    assert cleaned.sections[0].bullets[0].candidate_ids == []
    assert cleaned.review_flags[0].code == "missing_action_detail"
    SessionInsightsService._validate(
        cleaned,
        {
            "session_id": "session",
            "source_version": "source",
            "segments": [
                {"id": "S1", "text": "Do something enjoyable this week"},
                {"id": "S2", "text": "Yes, I will"},
            ],
        },
    )


@pytest.mark.parametrize("kind", ["general", "specific"])
def test_activity_review_activation_and_confirmation(tmp_path: Path, kind: str) -> None:
    recording_id = uuid4()
    item = candidate(kind)
    draft = SimpleNamespace(
        draft_id="draft", source_version="source", result=SimpleNamespace(action_candidates=[item])
    )
    service = ActionService(tmp_path, SimpleNamespace(load_completed=lambda _: draft))
    settings = ActivityActionSettings(
        exact_task=item.exact_task,
        assignment_kind=item.assignment_kind,
        agreed_activity=item.agreed_activity,
        frequency=item.frequency,
        timing=item.timing,
        reference_period=item.reference_period,
        reminder_time=None,
        reminder_time_zone=None,
    )
    request = MoodCandidateReviewRequest(
        draft_id="draft", source_version="source", disposition="activate", settings=settings
    )
    review = service.review_candidate(recording_id, item.id, request)
    assert review.action is not None
    assert review.action.type == "do_enjoyable_or_meaningful_activity"
    assert review.action.agreed_frequency == "this week"
    assert review.action.activity_mentions[0].label == "Call a friend"
    assert service.review_candidate(recording_id, item.id, request).action == review.action
    assert service.resolve_candidate(recording_id, item.id).status == "activated"
    assert service.list_actions() == [review.action]
    confirmation = service.confirm_action_reviews(
        recording_id, ActionReviewConfirmationRequest(draft_id="draft", source_version="source")
    )
    assert confirmation.action_ids == [review.action.id]
    assert service.load_action_review_confirmation(recording_id) == confirmation


def test_activity_review_rejects_changed_assignment(tmp_path: Path) -> None:
    recording_id = uuid4()
    item = candidate("specific")
    draft = SimpleNamespace(
        draft_id="draft", source_version="source", result=SimpleNamespace(action_candidates=[item])
    )
    service = ActionService(tmp_path, SimpleNamespace(load_completed=lambda _: draft))
    settings = ActivityActionSettings(
        exact_task=item.exact_task,
        assignment_kind="specific",
        agreed_activity="Go running",
        frequency=None,
        timing=None,
        reference_period=None,
    )
    with pytest.raises(ActionValidationError, match="cannot be changed"):
        service.review_candidate(
            recording_id,
            item.id,
            MoodCandidateReviewRequest(
                draft_id="draft", source_version="source", disposition="activate", settings=settings
            ),
        )
    rejected = service.review_candidate(
        recording_id,
        item.id,
        MoodCandidateReviewRequest(draft_id="draft", source_version="source", disposition="reject"),
    )
    assert rejected.disposition == "rejected"
    with pytest.raises(ActionConflictError):
        service.review_candidate(
            recording_id,
            item.id,
            MoodCandidateReviewRequest(
                draft_id="draft", source_version="source", disposition="activate", settings=settings
            ),
        )


def test_activity_review_api_accepts_activity_settings(recordings_dir: Path) -> None:
    item = candidate()
    with configured_client(recordings_dir, FakeOpenAIClient()) as client:
        recording = upload_recording(client)
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
            assignment_kind=item.assignment_kind,
            agreed_activity=item.agreed_activity,
            frequency=item.frequency,
            timing=item.timing,
            reference_period=item.reference_period,
        )
        response = client.put(
            f"/recordings/{recording['id']}/action-candidates/C1/review",
            json={
                "draft_id": "draft",
                "source_version": "source",
                "disposition": "activate",
                "settings": settings.model_dump(mode="json"),
            },
        )
        assert response.status_code == 200
        action_id = response.json()["action"]["id"]
        assert client.get(f"/actions/{action_id}").json()["type"] == item.type
        assert client.get("/actions").json()["actions"][0]["id"] == action_id
        assert client.put(
            f"/recordings/{recording['id']}/actions/confirmation",
            json={"draft_id": "draft", "source_version": "source"},
        ).json()["action_ids"] == [action_id]
