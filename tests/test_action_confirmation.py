from pathlib import Path

from tests.test_actions import generate_candidate
from tests.test_mood_setup import setup_request
from tests.test_session_insights import FakeOpenAIClient, configured_client


def test_action_list_confirmation_requires_activation_and_is_persisted(recordings_dir: Path):
    with configured_client(recordings_dir, FakeOpenAIClient(record_mood_candidate=True)) as client:
        recording, insights = generate_candidate(client)
        endpoint = f"/recordings/{recording['id']}/actions/confirmation"
        payload = {"draft_id": insights["draft_id"], "source_version": insights["source_version"]}

        assert client.get(endpoint).status_code == 404
        incomplete = client.put(endpoint, json=payload)
        assert incomplete.status_code == 422
        assert incomplete.json()["detail"] == "Activate every proposed action before confirming"

        activated = client.put(
            f"/recordings/{recording['id']}/action-candidates/C1/review",
            json=setup_request(insights, ["mood"]),
        )
        assert activated.status_code == 200
        action_id = activated.json()["action"]["id"]
        confirmed = client.put(endpoint, json=payload)
        assert confirmed.status_code == 200
        assert confirmed.json()["status"] == "confirmed"
        assert confirmed.json()["candidate_ids"] == ["C1"]
        assert confirmed.json()["action_ids"] == [action_id]
        assert client.get(endpoint).json() == confirmed.json()
        assert client.put(endpoint, json=payload).json() == confirmed.json()
        assert client.put(endpoint, json={**payload, "draft_id": "stale"}).status_code == 409

        assert client.delete(f"/actions/{action_id}").status_code == 204
        assert client.get(endpoint).status_code == 404
        confirmation_path = recordings_dir / "actions" / "confirmations" / f"{recording['id']}.json"
        assert not confirmation_path.exists()


def test_new_insights_draft_requires_new_action_review_before_confirmation(recordings_dir: Path):
    with configured_client(recordings_dir, FakeOpenAIClient(record_mood_candidate=True)) as client:
        recording, first_insights = generate_candidate(client)
        endpoint = f"/recordings/{recording['id']}/actions/confirmation"
        assert (
            client.put(
                f"/recordings/{recording['id']}/action-candidates/C1/review",
                json=setup_request(first_insights, ["mood"]),
            ).status_code
            == 200
        )
        regenerated = client.post(
            f"/recordings/{recording['id']}/insights", json={"session_date": "2026-11-17"}
        )
        assert regenerated.status_code == 200
        assert regenerated.json()["draft_id"] != first_insights["draft_id"]
        response = client.put(
            endpoint,
            json={
                "draft_id": regenerated.json()["draft_id"],
                "source_version": regenerated.json()["source_version"],
            },
        )
        assert response.status_code == 422
        assert response.json()["detail"] == "Activate every proposed action before confirming"
