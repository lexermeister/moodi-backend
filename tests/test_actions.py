import json
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


def activate_ongoing_candidate(
    client: TestClient,
    recording: dict[str, Any],
    insights: dict[str, Any],
) -> dict[str, Any]:
    candidate = insights["result"]["action_candidates"][0]
    settings = reviewed_settings(candidate)
    settings.update(
        {
            "duration_kind": "ongoing",
            "duration_days": None,
            "end_date": None,
            "preferred_check_in_time": "20:00",
            "preferred_time_zone": "Europe/Belgrade",
            "schedule_origin": "user_selected",
        }
    )
    response = client.put(
        f"/recordings/{recording['id']}/action-candidates/{candidate['id']}/review",
        json={
            "draft_id": insights["draft_id"],
            "source_version": insights["source_version"],
            "disposition": "activate",
            "settings": settings,
        },
    )
    assert response.status_code == 200
    return response.json()


def entry_values(
    *,
    mood_value: int = 0,
    context: str | None = None,
    relevant_date: str = "2026-11-10",
    entry_time: str = "2026-11-10T20:00:00+01:00",
) -> dict[str, Any]:
    return {
        "relevant_date": relevant_date,
        "entry_time": entry_time,
        "timezone_offset_minutes": 60,
        "day_key": relevant_date,
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
        assert first["action"]["settings"]["duration_kind"] == "fixed"
        assert first["action"]["settings"]["duration_days"] == 14
        assert first["action"]["settings"]["start_date"] == "2026-11-10"
        assert first["action"]["settings"]["end_date"] == "2026-11-23"
        assert first["action"]["settings"]["preferred_check_in_time"] is None
        assert first["action"]["settings"]["preferred_time_zone"] is None
        assert first["action"]["settings"]["schedule_origin"] == "moodi_default"

        listed = client.get("/actions", params={"recording_id": recording["id"]})
        assert listed.status_code == 200
        assert listed.json()["actions"] == [first["action"]]

    action_files = list((recordings_dir / "actions" / "records").glob("*.json"))
    assert len(action_files) == 1


def test_candidate_activation_accepts_user_selected_fixed_schedule(
    recordings_dir: Path,
) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_candidate(client)
        candidate = insights["result"]["action_candidates"][0]
        settings = reviewed_settings(candidate)
        settings.update(
            {
                "duration_kind": "fixed",
                "duration_days": 30,
                "end_date": "2026-12-09",
                "preferred_check_in_time": "20:30",
                "preferred_time_zone": "Europe/Belgrade",
                "schedule_origin": "user_selected",
            }
        )

        response = client.put(
            f"/recordings/{recording['id']}/action-candidates/{candidate['id']}/review",
            json={
                "draft_id": insights["draft_id"],
                "source_version": insights["source_version"],
                "disposition": "activate",
                "settings": settings,
            },
        )

        assert response.status_code == 200
        activated = response.json()["action"]["settings"]
        assert activated["duration_kind"] == "fixed"
        assert activated["duration_days"] == 30
        assert activated["end_date"] == "2026-12-09"
        assert activated["preferred_check_in_time"] == "20:30"
        assert activated["preferred_time_zone"] == "Europe/Belgrade"
        assert activated["schedule_origin"] == "user_selected"


def test_candidate_activation_accepts_ongoing_schedule(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_candidate(client)
        candidate = insights["result"]["action_candidates"][0]
        settings = reviewed_settings(candidate)
        settings.update(
            {
                "duration_kind": "ongoing",
                "duration_days": None,
                "end_date": None,
                "preferred_check_in_time": "19:45",
                "preferred_time_zone": "Europe/Belgrade",
                "schedule_origin": "user_selected",
            }
        )

        response = client.put(
            f"/recordings/{recording['id']}/action-candidates/{candidate['id']}/review",
            json={
                "draft_id": insights["draft_id"],
                "source_version": insights["source_version"],
                "disposition": "activate",
                "settings": settings,
            },
        )

        assert response.status_code == 200
        activated = response.json()["action"]["settings"]
        assert activated["duration_kind"] == "ongoing"
        assert activated["duration_days"] is None
        assert activated["end_date"] is None

        action_id = response.json()["action"]["id"]
        entry_id = "58ab0b60-afcf-40ac-9734-c7390cf8f15f"
        confirmed = client.post(
            f"/actions/{action_id}/entries/{entry_id}/confirm",
            json={
                "values": entry_values(
                    mood_value=8,
                    relevant_date="2027-05-01",
                    entry_time="2027-05-01T20:00:00+01:00",
                ),
                "expected_revision": None,
                "idempotency_key": "ongoing-entry",
            },
        )
        assert confirmed.status_code == 200


def test_candidate_reuses_compatible_active_ongoing_action(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        first_recording, first_insights = generate_candidate(client)
        first_candidate = first_insights["result"]["action_candidates"][0]
        first_review = activate_ongoing_candidate(client, first_recording, first_insights)
        action = first_review["action"]
        entry_id = "2d063c5f-86a1-42a7-8fa4-f51d17594fcc"
        saved_entry = client.put(
            f"/actions/{action['id']}/entries/{entry_id}/draft",
            json={"values": entry_values(mood_value=6), "expected_revision": None},
        )
        assert saved_entry.status_code == 200

        second_recording, second_insights = generate_candidate(client)
        second_candidate = second_insights["result"]["action_candidates"][0]
        resolution_url = (
            f"/recordings/{second_recording['id']}/action-candidates/"
            f"{second_candidate['id']}/resolution"
        )
        resolution = client.get(resolution_url)

        assert resolution.status_code == 200
        assert resolution.json()["status"] == "already_active"
        assert resolution.json()["action"] == action

        linked = activate_candidate(client, second_recording, second_insights)
        retried = activate_candidate(client, second_recording, second_insights)

        assert linked == retried
        assert linked["disposition"] == "linked_existing"
        assert linked["action"] == action
        assert client.get("/actions").json() == {"actions": [action]}
        assert client.get(f"/actions/{action['id']}/entries").json() == {
            "entries": [saved_entry.json()]
        }
        assert client.get(resolution_url).json()["status"] == "linked_existing"

        deleted = client.delete(f"/actions/{action['id']}")

        assert deleted.status_code == 204
        for recording, candidate in (
            (first_recording, first_candidate),
            (second_recording, second_candidate),
        ):
            review_url = f"/recordings/{recording['id']}/action-candidates/{candidate['id']}/review"
            candidate_resolution_url = (
                f"/recordings/{recording['id']}/action-candidates/{candidate['id']}/resolution"
            )
            assert client.get(review_url).status_code == 404
            assert client.get(candidate_resolution_url).json()["status"] == "needs_setup"


def test_fixed_action_does_not_block_another_action(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        first_recording, first_insights = generate_candidate(client)
        first_action = activate_candidate(client, first_recording, first_insights)["action"]
        second_recording, second_insights = generate_candidate(client)
        second_candidate = second_insights["result"]["action_candidates"][0]

        resolution = client.get(
            f"/recordings/{second_recording['id']}/action-candidates/"
            f"{second_candidate['id']}/resolution"
        )
        second_action = activate_candidate(client, second_recording, second_insights)["action"]

        assert resolution.status_code == 200
        assert resolution.json()["status"] == "needs_setup"
        assert second_action["id"] != first_action["id"]
        assert len(client.get("/actions").json()["actions"]) == 2


def test_incompatible_ongoing_action_blocks_activation(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        first_recording, first_insights = generate_candidate(client)
        action = activate_ongoing_candidate(client, first_recording, first_insights)["action"]
        action_path = recordings_dir / "actions" / "records" / f"{action['id']}.json"
        stored_action = json.loads(action_path.read_text(encoding="utf-8"))
        stored_action["settings"]["timing"] = "morning"
        action_path.write_text(json.dumps(stored_action), encoding="utf-8")

        second_recording, second_insights = generate_candidate(client)
        second_candidate = second_insights["result"]["action_candidates"][0]
        resolution = client.get(
            f"/recordings/{second_recording['id']}/action-candidates/"
            f"{second_candidate['id']}/resolution"
        )
        activation = client.put(
            f"/recordings/{second_recording['id']}/action-candidates/"
            f"{second_candidate['id']}/review",
            json={
                "draft_id": second_insights["draft_id"],
                "source_version": second_insights["source_version"],
                "disposition": "activate",
                "settings": reviewed_settings(second_candidate),
            },
        )

        assert resolution.status_code == 200
        assert resolution.json()["status"] == "conflict"
        assert resolution.json()["reason"] == "incompatible_ongoing_action"
        assert resolution.json()["action"]["id"] == action["id"]
        assert activation.status_code == 409
        assert activation.json() == {
            "detail": "An active ongoing mood Action already has different settings"
        }
        assert len(client.get("/actions").json()["actions"]) == 1


def test_candidate_activation_rejects_clinical_setting_edits(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_candidate(client)
        candidate = insights["result"]["action_candidates"][0]
        settings = reviewed_settings(candidate)
        settings["exact_task"] = "Track something else"

        response = client.put(
            f"/recordings/{recording['id']}/action-candidates/{candidate['id']}/review",
            json={
                "draft_id": insights["draft_id"],
                "source_version": insights["source_version"],
                "disposition": "activate",
                "settings": settings,
            },
        )

        assert response.status_code == 422
        assert (
            response.json()["detail"] == "Only the Action schedule can be changed before activation"
        )


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


def test_delete_action_removes_entries_and_review_and_can_be_set_up_again(
    recordings_dir: Path,
) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    draft_entry_id = "2d063c5f-86a1-42a7-8fa4-f51d17594fcc"
    confirmed_entry_id = "58ab0b60-afcf-40ac-9734-c7390cf8f15f"
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_candidate(client)
        candidate = insights["result"]["action_candidates"][0]
        action = activate_candidate(client, recording, insights)["action"]
        action_id = action["id"]

        draft = client.put(
            f"/actions/{action_id}/entries/{draft_entry_id}/draft",
            json={"values": entry_values(mood_value=4), "expected_revision": None},
        )
        confirmed = client.post(
            f"/actions/{action_id}/entries/{confirmed_entry_id}/confirm",
            json={
                "values": entry_values(mood_value=7),
                "expected_revision": None,
                "idempotency_key": "delete-action-confirmed-entry",
            },
        )
        assert draft.status_code == confirmed.status_code == 200

        deleted = client.delete(f"/actions/{action_id}")

        assert deleted.status_code == 204
        assert client.get(f"/actions/{action_id}").status_code == 404
        assert client.get(f"/actions/{action_id}/entries").status_code == 404
        assert client.get("/actions").json() == {"actions": []}
        review = client.get(
            f"/recordings/{recording['id']}/action-candidates/{candidate['id']}/review"
        )
        assert review.status_code == 404

        reactivated = activate_candidate(client, recording, insights)
        assert reactivated["action"]["id"] == action_id
        assert client.get(f"/actions/{action_id}/entries").json() == {"entries": []}

    assert not (recordings_dir / "actions" / "entries" / action_id).exists()
    assert not list((recordings_dir / "actions" / ".deleting").iterdir())


def test_delete_action_keeps_other_actions_and_entries(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        first_recording, first_insights = generate_candidate(client)
        first_action = activate_candidate(client, first_recording, first_insights)["action"]
        second_recording, second_insights = generate_candidate(client)
        second_action = activate_candidate(client, second_recording, second_insights)["action"]
        second_entry_id = "2d063c5f-86a1-42a7-8fa4-f51d17594fcc"
        saved = client.put(
            f"/actions/{second_action['id']}/entries/{second_entry_id}/draft",
            json={"values": entry_values(mood_value=5), "expected_revision": None},
        )
        assert saved.status_code == 200

        deleted = client.delete(f"/actions/{first_action['id']}")

        assert deleted.status_code == 204
        assert client.get(f"/actions/{second_action['id']}").json() == second_action
        assert client.get(f"/actions/{second_action['id']}/entries").json() == {
            "entries": [saved.json()]
        }


def test_delete_action_returns_not_found_for_unknown_action(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        response = client.delete("/actions/2d063c5f-86a1-42a7-8fa4-f51d17594fcc")

    assert response.status_code == 404
    assert response.json() == {"detail": "Mood Action was not found"}


def test_delete_action_rolls_back_if_staging_fails(
    recordings_dir: Path,
    monkeypatch,
) -> None:
    fake_openai = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_candidate(client)
        candidate = insights["result"]["action_candidates"][0]
        action = activate_candidate(client, recording, insights)["action"]
        action_id = action["id"]
        entry_id = "2d063c5f-86a1-42a7-8fa4-f51d17594fcc"
        saved = client.put(
            f"/actions/{action_id}/entries/{entry_id}/draft",
            json={"values": entry_values(mood_value=5), "expected_revision": None},
        )
        assert saved.status_code == 200

        action_path = recordings_dir / "actions" / "records" / f"{action_id}.json"
        original_replace = Path.replace

        def fail_action_move(path: Path, target: Path) -> Path:
            if path == action_path:
                raise OSError("simulated Action move failure")
            return original_replace(path, target)

        monkeypatch.setattr(Path, "replace", fail_action_move)
        response = client.delete(f"/actions/{action_id}")

        assert response.status_code == 502
        assert client.get(f"/actions/{action_id}").json() == action
        assert client.get(f"/actions/{action_id}/entries").json() == {"entries": [saved.json()]}
        review = client.get(
            f"/recordings/{recording['id']}/action-candidates/{candidate['id']}/review"
        )
        assert review.status_code == 200
        assert not list((recordings_dir / "actions" / ".deleting").iterdir())


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
