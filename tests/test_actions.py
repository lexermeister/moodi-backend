from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tests.test_session_insights import FakeOpenAIClient, configured_client, upload_recording


def generate_candidate(client: TestClient) -> tuple[dict[str, Any], dict[str, Any]]:
    recording = upload_recording(client)
    assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
    response = client.post(f"/recordings/{recording['id']}/insights", json={})
    assert response.status_code == 200
    return recording, response.json()


def reviewed_settings(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_task": candidate["exact_task"],
        "frequency": candidate["frequency"],
        "timing": candidate["timing"],
        "reference_period": candidate["reference_period"],
        "start_date": candidate["start_date"],
        "end_date": candidate["end_date"],
        "duration_days": candidate["duration_days"],
        "configuration": candidate["configuration"],
    }


def activate_candidate(
    client: TestClient,
    recording: dict[str, Any],
    insights: dict[str, Any],
) -> dict[str, Any]:
    candidate = insights["result"]["action_candidates"][0]
    response = client.put(
        f"/recordings/{recording['id']}/action-candidates/{candidate['id']}/review",
        json={
            "draft_id": insights["draft_id"],
            "source_version": insights["source_version"],
            "disposition": "activate",
            "settings": reviewed_settings(candidate),
        },
    )
    assert response.status_code == 200
    return response.json()


def entry_values(*, mood_value: int = 0, context: str | None = None) -> dict[str, Any]:
    return {
        "relevant_date": "2026-11-10",
        "entry_time": "2026-11-10T20:00:00+01:00",
        "timezone_offset_minutes": 60,
        "day_key": "2026-11-10",
        "mood_value": mood_value,
        "mood_label": None,
        "context": context,
        "scale_id": "overall-day-mood-0-10-v1",
    }


def test_candidate_activation_is_reviewed_persisted_and_idempotent(
    recordings_dir: Path,
) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_candidate(client)
        first = activate_candidate(client, recording, insights)
        second = activate_candidate(client, recording, insights)

        assert first == second
        assert first["disposition"] == "activated"
        assert first["action"]["settings"]["configuration"]["input_kind"] == "numeric_0_10"
        assert first["action"]["settings"]["configuration"]["scale_origin"] == "moodi_default"
        assert first["action"]["settings"]["frequency"] == "once daily"
        assert first["action"]["settings"]["timing"] == "evening"
        assert first["action"]["settings"]["duration_days"] == 14
        assert first["action"]["settings"]["start_date"] == "2026-11-10"
        assert first["action"]["settings"]["end_date"] == "2026-11-23"

        listed = client.get("/actions", params={"recording_id": recording["id"]})
        assert listed.status_code == 200
        assert listed.json()["actions"] == [first["action"]]

    action_files = list((recordings_dir / "actions" / "records").glob("*.json"))
    assert len(action_files) == 1


def test_rejected_candidate_creates_no_action(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_candidate(client)
        candidate = insights["result"]["action_candidates"][0]
        response = client.put(
            f"/recordings/{recording['id']}/action-candidates/{candidate['id']}/review",
            json={
                "draft_id": insights["draft_id"],
                "source_version": insights["source_version"],
                "disposition": "reject",
                "settings": None,
            },
        )

        assert response.status_code == 200
        assert response.json()["disposition"] == "rejected"
        assert response.json()["action"] is None
        assert client.get("/actions").json() == {"actions": []}


def test_mood_entry_draft_confirm_amend_export_and_delete(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    entry_id = "2d063c5f-86a1-42a7-8fa4-f51d17594fcc"
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_candidate(client)
        action = activate_candidate(client, recording, insights)["action"]
        action_id = action["id"]

        draft = client.put(
            f"/actions/{action_id}/entries/{entry_id}/draft",
            json={"values": entry_values(), "expected_revision": None},
        )
        assert draft.status_code == 200
        assert draft.json()["status"] == "draft"
        assert draft.json()["values"]["mood_value"] == 0

        confirm_payload = {
            "values": entry_values(),
            "expected_revision": 1,
            "idempotency_key": "confirm-entry-1",
        }
        confirmed = client.post(
            f"/actions/{action_id}/entries/{entry_id}/confirm",
            json=confirm_payload,
        )
        retried = client.post(
            f"/actions/{action_id}/entries/{entry_id}/confirm",
            json=confirm_payload,
        )
        assert confirmed.status_code == retried.status_code == 200
        assert confirmed.json() == retried.json()
        assert confirmed.json()["revision"] == 2
        assert len(confirmed.json()["revision_history"]) == 1

        amended = client.patch(
            f"/actions/{action_id}/entries/{entry_id}",
            json={"values": entry_values(mood_value=7), "expected_revision": 2},
        )
        assert amended.status_code == 200
        assert amended.json()["values"]["mood_value"] == 7
        assert amended.json()["revision"] == 3
        assert len(amended.json()["revision_history"]) == 2

        listed = client.get(f"/actions/{action_id}/entries")
        assert listed.status_code == 200
        assert listed.json()["entries"] == [amended.json()]

        exported = client.post(
            f"/actions/{action_id}/entries/export",
            json={"entry_ids": [entry_id]},
        )
        assert exported.status_code == 200
        assert exported.json()["entries"] == [amended.json()]

        deleted = client.delete(f"/actions/{action_id}/entries/{entry_id}")
        assert deleted.status_code == 204
        assert client.get(f"/actions/{action_id}/entries").json() == {"entries": []}


def test_action_configuration_rejects_unassigned_context(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_candidate(client)
        action_id = activate_candidate(client, recording, insights)["action"]["id"]
        response = client.post(
            f"/actions/{action_id}/entries/2d063c5f-86a1-42a7-8fa4-f51d17594fcc/confirm",
            json={
                "values": entry_values(mood_value=6, context="Not part of the assignment"),
                "expected_revision": None,
                "idempotency_key": "confirm-with-context",
            },
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "Context is not enabled for this Action"}


def test_entry_day_key_must_match_captured_offset(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_candidate(client)
        action_id = activate_candidate(client, recording, insights)["action"]["id"]
        values = entry_values(mood_value=6)
        values["day_key"] = "2026-11-11"
        response = client.put(
            f"/actions/{action_id}/entries/2d063c5f-86a1-42a7-8fa4-f51d17594fcc/draft",
            json={"values": values, "expected_revision": None},
        )

    assert response.status_code == 422
