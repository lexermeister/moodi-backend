import json
from pathlib import Path

import pytest

from tests.test_actions import entry_values, generate_candidate, reviewed_settings
from tests.test_session_insights import FakeOpenAIClient, configured_client
from tests.test_session_insights_review import review_payload


def setup_request(insights, measurements):
    candidate = insights["result"]["action_candidates"][0]
    settings = reviewed_settings(candidate) | {
        "what_to_record": measurements,
        "duration_kind": "unscheduled",
        "duration_days": None,
        "start_date": None,
        "end_date": None,
    }
    return {
        "draft_id": insights["draft_id"],
        "source_version": insights["source_version"],
        "disposition": "activate",
        "settings": settings,
    }


def test_duplicate_review_updates_first_action_and_keeps_entries(recordings_dir: Path):
    with configured_client(recordings_dir, FakeOpenAIClient(record_mood_candidate=True)) as client:
        recording, insights = generate_candidate(client)
        path = f"/recordings/{recording['id']}/action-candidates/C1"
        payload = setup_request(insights, ["mood", "motivation"])
        response = client.put(f"{path}/review", json=payload)
        assert response.status_code == 200
        action = response.json()["action"]
        assert action["settings"]["duration_kind"] == "unscheduled"
        assert action["settings"]["start_date"] is None
        assert action["settings"]["end_date"] is None
        assert action["settings"]["preferred_check_in_time"] is None
        assert client.put(f"{path}/review", json=payload).json() == response.json()
        assert client.get(f"/actions/{action['id']}").json() == action
        assert client.get(f"{path}/resolution").json()["status"] == "activated"
        different = setup_request(insights, ["mood", "energy"])
        assert client.put(f"{path}/review", json=different).status_code == 409

        entry_id = "2d063c5f-86a1-42a7-8fa4-f51d17594fcc"
        saved_entry = client.post(
            f"/actions/{action['id']}/entries/{entry_id}/confirm",
            json={"values": entry_values(), "idempotency_key": "first-session-entry"},
        )
        assert saved_entry.status_code == 200

        second, second_insights = generate_candidate(client)
        second_path = f"/recordings/{second['id']}/action-candidates/C1"
        assert client.get(f"{second_path}/resolution").json()["status"] == "already_active"
        second_payload = setup_request(second_insights, ["mood", "energy"])
        linked = client.put(f"{second_path}/review", json=second_payload)
        assert linked.status_code == 200
        assert linked.json()["disposition"] == "linked_existing"
        updated = linked.json()["action"]
        assert updated["id"] == action["id"]
        assert updated["revision"] == action["revision"] + 1
        assert updated["settings"]["what_to_record"] == ["mood", "energy"]
        assert client.put(f"{second_path}/review", json=second_payload).json() == linked.json()
        assert client.get(f"/actions/{action['id']}").json() == updated
        assert client.get(f"{path}/review").json()["action"] == updated
        assert client.get(f"/actions/{action['id']}/entries").json()["entries"] == [
            saved_entry.json()
        ]
        assert len(client.get("/actions").json()["actions"]) == 1

        entry = client.post(
            f"/actions/{action['id']}/entries/2d063c5f-86a1-42a7-8fa4-f51d17594fcd/confirm",
            json={
                "values": entry_values(
                    relevant_date="2027-01-01", entry_time="2027-01-01T20:00:00+01:00"
                ),
                "idempotency_key": "no-expiry",
            },
        )
        assert entry.status_code == 200


def test_duplicate_review_updates_schedule_without_resetting_action(recordings_dir: Path):
    with configured_client(recordings_dir, FakeOpenAIClient(record_mood_candidate=True)) as client:
        first_recording, first_insights = generate_candidate(client)
        first_payload = setup_request(first_insights, ["mood"])
        first_payload["settings"].update(
            duration_kind="ongoing",
            cadence="daily",
            preferred_check_in_time="20:00",
            preferred_time_zone="Europe/Belgrade",
            schedule_origin="user_selected",
        )
        first = client.put(
            f"/recordings/{first_recording['id']}/action-candidates/C1/review",
            json=first_payload,
        )
        assert first.status_code == 200
        original = first.json()["action"]
        entry = client.post(
            f"/actions/{original['id']}/entries/2d063c5f-86a1-42a7-8fa4-f51d17594fcc/confirm",
            json={
                "values": entry_values(
                    relevant_date="2027-01-01", entry_time="2027-01-01T20:00:00+01:00"
                ),
                "idempotency_key": "existing-mood-entry",
            },
        )
        assert entry.status_code == 200

        second_recording, second_insights = generate_candidate(client)
        second_payload = setup_request(second_insights, ["mood", "energy"])
        second_payload["settings"].update(
            duration_kind="ongoing",
            cadence="weekdays",
            preferred_check_in_time="21:00",
            preferred_time_zone="Europe/Belgrade",
            schedule_origin="user_selected",
        )
        path = f"/recordings/{second_recording['id']}/action-candidates/C1"
        response = client.put(f"{path}/review", json=second_payload)
        assert response.status_code == 200
        updated = response.json()["action"]
        assert updated["id"] == original["id"]
        assert updated["created_at"] == original["created_at"]
        assert updated["settings"]["start_date"] == original["settings"]["start_date"]
        assert updated["settings"]["cadence"] == "weekdays"
        assert updated["settings"]["preferred_check_in_time"] == "21:00"
        assert updated["settings"]["what_to_record"] == ["mood", "energy"]
        assert updated["revision"] == original["revision"] + 1
        assert client.get(f"/actions/{original['id']}/entries").json()["entries"] == [entry.json()]
        assert client.put(f"{path}/review", json=second_payload).json()["action"] == updated
        assert len(client.get("/actions").json()["actions"]) == 1


@pytest.mark.parametrize("measurements", [[], ["energy"], ["mood", "mood"], ["mood", "stress"]])
def test_invalid_measurement_selection(recordings_dir: Path, measurements):
    with configured_client(recordings_dir, FakeOpenAIClient(record_mood_candidate=True)) as client:
        recording, insights = generate_candidate(client)
        response = client.put(
            f"/recordings/{recording['id']}/action-candidates/C1/review",
            json=setup_request(insights, measurements),
        )
        assert response.status_code == 422
        assert client.get("/actions").json()["actions"] == []


def test_unscheduled_setup_rejects_hidden_schedule(recordings_dir: Path):
    with configured_client(recordings_dir, FakeOpenAIClient(record_mood_candidate=True)) as client:
        recording, insights = generate_candidate(client)
        payload = setup_request(insights, ["mood", "energy"])
        payload["settings"]["duration_days"] = 14
        assert (
            client.put(
                f"/recordings/{recording['id']}/action-candidates/C1/review", json=payload
            ).status_code
            == 422
        )


def test_generated_combined_measurements_survive_review(recordings_dir: Path):
    provider = FakeOpenAIClient(record_mood_candidate=True)
    original = provider.responses.parse

    def parse(**kwargs):
        result = original(**kwargs)
        if hasattr(result.output_parsed, "action_candidates"):
            result.output_parsed.action_candidates[0].what_to_record = [
                "mood",
                "energy",
                "sleep_quality",
                "motivation",
            ]
        return result

    provider.responses.parse = parse
    with configured_client(recordings_dir, provider) as client:
        recording, insights = generate_candidate(client)
        candidate = insights["result"]["action_candidates"][0]
        assert candidate["what_to_record"] == ["mood", "energy", "sleep_quality", "motivation"]
        response = client.put(
            f"/recordings/{recording['id']}/action-candidates/C1/review",
            json=setup_request(insights, candidate["what_to_record"]),
        )
        assert response.status_code == 200
        assert (
            response.json()["action"]["settings"]["what_to_record"] == candidate["what_to_record"]
        )


def test_reviewed_previous_prompt_draft_remains_available_for_action_review(recordings_dir: Path):
    provider = FakeOpenAIClient(record_mood_candidate=True)
    with configured_client(recordings_dir, provider) as client:
        recording, insights = generate_candidate(client)
        review_url = f"/recordings/{recording['id']}/insights/review"
        bullet = insights["result"]["sections"][0]["bullets"][0]
        reviewed = client.put(
            review_url,
            json=review_payload(insights, disposition="retained", displayed_text=bullet["text"]),
        )
        assert reviewed.status_code == 200

        draft_path = recordings_dir / f"{recording['id']}.insights.json"
        stored = json.loads(draft_path.read_text(encoding="utf-8"))
        stored["prompt_version"] = "v0.5.0"
        draft_path.write_text(json.dumps(stored), encoding="utf-8")

        assert client.get(review_url).status_code == 200
        candidate_url = f"/recordings/{recording['id']}/action-candidates/C1"
        assert client.get(f"{candidate_url}/resolution").json()["status"] == "needs_setup"
        activated = client.put(f"{candidate_url}/review", json=setup_request(insights, ["mood"]))
        assert activated.status_code == 200
        assert activated.json()["disposition"] == "activated"
        assert provider.responses.insights_call_count == 1


@pytest.mark.parametrize(
    ("cadence", "weekday"),
    [("daily", None), ("weekdays", None), ("weekly", 2)],
)
def test_review_saves_exact_cadence_and_time(recordings_dir: Path, cadence, weekday):
    with configured_client(recordings_dir, FakeOpenAIClient(record_mood_candidate=True)) as client:
        recording, insights = generate_candidate(client)
        candidate = insights["result"]["action_candidates"][0]
        assert candidate["what_to_record"] == ["mood"]
        payload = setup_request(insights, ["mood", "energy"])
        payload["settings"].update(
            duration_kind="ongoing",
            duration_days=None,
            cadence=cadence,
            weekly_weekday=weekday,
            preferred_check_in_time="20:15",
            preferred_time_zone="Europe/Belgrade",
            schedule_origin="user_selected",
        )
        response = client.put(
            f"/recordings/{recording['id']}/action-candidates/C1/review", json=payload
        )
        assert response.status_code == 200
        settings = response.json()["action"]["settings"]
        assert settings["cadence"] == cadence
        assert settings["weekly_weekday"] == weekday
        assert settings["preferred_check_in_time"] == "20:15"
        assert settings["what_to_record"] == ["mood", "energy"]
        assert (
            client.get(f"/actions/{response.json()['action']['id']}").json()["settings"] == settings
        )


@pytest.mark.parametrize(
    ("cadence", "weekday", "time"),
    [("weekly", None, "20:00"), ("daily", 1, "20:00"), ("weekdays", None, "8 pm")],
)
def test_review_rejects_invalid_schedule(recordings_dir: Path, cadence, weekday, time):
    with configured_client(recordings_dir, FakeOpenAIClient(record_mood_candidate=True)) as client:
        recording, insights = generate_candidate(client)
        payload = setup_request(insights, ["mood"])
        payload["settings"].update(
            duration_kind="ongoing",
            duration_days=None,
            cadence=cadence,
            weekly_weekday=weekday,
            preferred_check_in_time=time,
            preferred_time_zone="Europe/Belgrade",
        )
        response = client.put(
            f"/recordings/{recording['id']}/action-candidates/C1/review", json=payload
        )
        assert response.status_code == 422


def test_multiple_generated_candidates_are_rejected(recordings_dir: Path):
    provider = FakeOpenAIClient(record_mood_candidate=True)
    original = provider.responses.parse

    def parse(**kwargs):
        result = original(**kwargs)
        if hasattr(result.output_parsed, "action_candidates"):
            candidate = result.output_parsed.action_candidates[0]
            result.output_parsed.action_candidates.append(candidate.model_copy(update={"id": "C2"}))
        return result

    provider.responses.parse = parse
    from tests.test_session_insights import upload_recording

    with configured_client(recordings_dir, provider) as client:
        recording = upload_recording(client)
        assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
        assert client.post(f"/recordings/{recording['id']}/insights", json={}).status_code == 502
        assert provider.responses.insights_call_count == 2
