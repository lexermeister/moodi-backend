import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from tests.test_session_insights import FakeOpenAIClient, configured_client, upload_recording


def generate_insights(client: TestClient) -> tuple[dict[str, Any], dict[str, Any]]:
    recording = upload_recording(client)
    assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
    response = client.post(f"/recordings/{recording['id']}/insights", json={})
    assert response.status_code == 200
    return recording, response.json()


def review_payload(
    insights: dict[str, Any],
    *,
    disposition: str,
    displayed_text: str | None,
) -> dict[str, Any]:
    bullet = insights["result"]["sections"][0]["bullets"][0]
    return {
        "draft_id": insights["draft_id"],
        "source_version": insights["source_version"],
        "items": [
            {
                "item_id": bullet["item_id"],
                "disposition": disposition,
                "displayed_text": displayed_text,
            }
        ],
    }


def test_review_is_saved_loaded_and_does_not_mutate_generated_draft(
    recordings_dir: Path,
) -> None:
    fake_openai = FakeOpenAIClient()
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_insights(client)
        draft_path = recordings_dir / f"{recording['id']}.insights.json"
        draft_before = draft_path.read_bytes()
        endpoint = f"/recordings/{recording['id']}/insights/review"
        payload = review_payload(
            insights,
            disposition="edited",
            displayed_text="You wanted to better understand your anxiety.",
        )

        saved = client.put(endpoint, json=payload)
        loaded = client.get(endpoint)

    assert saved.status_code == loaded.status_code == 200
    assert saved.json() == loaded.json()
    review = saved.json()
    assert review["recording_id"] == recording["id"]
    assert review["draft_id"] == insights["draft_id"]
    assert review["status"] == "reviewed"
    assert review["sections"][0]["items"] == [
        {
            "item_id": "B1",
            "disposition": "edited",
            "original_text": "You wanted to understand your anxiety.",
            "displayed_text": "You wanted to better understand your anxiety.",
            "provenance": "client_edited",
        }
    ]
    assert draft_path.read_bytes() == draft_before
    stored = json.loads(
        (recordings_dir / f"{recording['id']}.insights-review.json").read_text(encoding="utf-8")
    )
    assert stored == review


def test_review_can_remove_generated_item(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient()
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_insights(client)
        response = client.put(
            f"/recordings/{recording['id']}/insights/review",
            json=review_payload(insights, disposition="removed", displayed_text=None),
        )

    assert response.status_code == 200
    item = response.json()["sections"][0]["items"][0]
    assert item["disposition"] == "removed"
    assert item["displayed_text"] is None
    assert item["provenance"] == "source_backed"


def test_review_rejects_stale_draft(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient()
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_insights(client)
        payload = review_payload(
            insights,
            disposition="retained",
            displayed_text="You wanted to understand your anxiety.",
        )
        payload["draft_id"] = "sha256:stale"
        response = client.put(
            f"/recordings/{recording['id']}/insights/review",
            json=payload,
        )

    assert response.status_code == 409
    assert response.json() == {
        "detail": "Session insights changed; review the latest draft before saving"
    }


def test_review_requires_every_generated_item(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient()
    with configured_client(recordings_dir, fake_openai) as client:
        recording, insights = generate_insights(client)
        response = client.put(
            f"/recordings/{recording['id']}/insights/review",
            json={
                "draft_id": insights["draft_id"],
                "source_version": insights["source_version"],
                "items": [],
            },
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "Review items do not match the generated draft"}


def test_review_requires_generated_insights(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient()
    with configured_client(recordings_dir, fake_openai) as client:
        recording = upload_recording(client)
        response = client.put(
            f"/recordings/{recording['id']}/insights/review",
            json={
                "draft_id": "sha256:missing",
                "source_version": "sha256:missing",
                "items": [],
            },
        )

    assert response.status_code == 409
    assert response.json() == {"detail": "Generate session insights before reviewing them"}


def test_missing_review_returns_not_found(recordings_dir: Path) -> None:
    fake_openai = FakeOpenAIClient()
    with configured_client(recordings_dir, fake_openai) as client:
        recording, _ = generate_insights(client)
        response = client.get(f"/recordings/{recording['id']}/insights/review")

    assert response.status_code == 404
    assert response.json() == {"detail": "No reviewed session insights were found"}
