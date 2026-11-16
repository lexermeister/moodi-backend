from pathlib import Path

import pytest

from tests.test_actions import entry_values, generate_candidate, reviewed_settings
from tests.test_session_insights import FakeOpenAIClient, configured_client


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


def test_unscheduled_setup_retry_reload_and_conflict(recordings_dir: Path):
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

        # A separate session suggests different measurements; no silent replacement.
        second, second_insights = generate_candidate(client)
        second_path = f"/recordings/{second['id']}/action-candidates/C1"
        assert client.get(f"{second_path}/resolution").json()["status"] == "conflict"
        assert (
            client.put(
                f"{second_path}/review", json=setup_request(second_insights, ["mood", "energy"])
            ).status_code
            == 409
        )
        linked = client.put(
            f"{second_path}/review", json=setup_request(second_insights, ["mood", "motivation"])
        )
        assert linked.status_code == 200
        assert linked.json()["disposition"] == "linked_existing"
        assert linked.json()["action"]["id"] == action["id"]
        assert len(client.get("/actions").json()["actions"]) == 1

        # There is no hidden 14-day expiry on the new flow.
        entry = client.post(
            f"/actions/{action['id']}/entries/2d063c5f-86a1-42a7-8fa4-f51d17594fcc/confirm",
            json={
                "values": entry_values(
                    relevant_date="2027-01-01", entry_time="2027-01-01T20:00:00+01:00"
                ),
                "idempotency_key": "no-expiry",
            },
        )
        assert entry.status_code == 200


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
