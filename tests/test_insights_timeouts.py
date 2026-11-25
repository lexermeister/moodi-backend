from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from openai import APITimeoutError

from app.services.transcription import OpenAISessionInsightsGenerator
from tests.test_session_insights import FakeOpenAIClient, configured_client, upload_recording


def test_insights_client_disables_hidden_retries_and_bounds_network_time(monkeypatch):
    captured = {}

    def create_client(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr("app.services.transcription.OpenAI", create_client)
    generator = OpenAISessionInsightsGenerator("test-key", "gpt-6-luna", "none")
    generator._get_client()
    assert captured["max_retries"] == 0
    assert captured["timeout"].read == 60
    assert captured["timeout"].connect == 5


def test_provider_timeout_returns_504_without_retry_or_sensitive_content(recordings_dir, caplog):
    timeout = APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    fake = FakeOpenAIClient(insights_error=timeout)
    with configured_client(recordings_dir, fake) as client:
        recording = upload_recording(client)
        assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
        response = client.post(f"/recordings/{recording['id']}/insights", json={})
    assert response.status_code == 504
    assert response.json() == {"detail": "Session insights took too long. Please try again."}
    assert fake.responses.insights_call_count == 1
    assert "insights_provider_timeout" in caplog.text
    assert "client_request_id=" in caplog.text
    assert not (recordings_dir / f"{recording['id']}.insights.json").exists()


def test_concurrent_insights_requests_do_not_start_duplicate_provider_calls(recordings_dir):
    entered = Event()
    release = Event()
    fake = FakeOpenAIClient()
    original = fake.responses.parse

    def parse(**kwargs):
        if kwargs["text_format"].__name__ == "SessionInsightsResult":
            entered.set()
            assert release.wait(5)
        return original(**kwargs)

    fake.responses.parse = parse
    with configured_client(recordings_dir, fake) as client:
        recording = upload_recording(client)
        assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
        path = f"/recordings/{recording['id']}/insights"
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(client.post, path, json={})
            assert entered.wait(5)
            try:
                assert client.post(path, json={}).status_code == 409
            finally:
                release.set()
            assert first.result(timeout=5).status_code == 200
        assert client.post(path, json={}).status_code == 200
    assert fake.responses.insights_call_count == 1


def test_timeout_releases_the_recording_for_retry(recordings_dir):
    fake = FakeOpenAIClient(
        insights_error=APITimeoutError(
            request=httpx.Request("POST", "https://api.openai.com/v1/responses")
        )
    )
    with configured_client(recordings_dir, fake) as client:
        recording = upload_recording(client)
        assert client.post(f"/recordings/{recording['id']}/transcription").status_code == 200
        path = f"/recordings/{recording['id']}/insights"
        assert client.post(path, json={}).status_code == 504
        fake.responses.insights_error = None
        assert client.post(path, json={}).status_code == 200


def test_real_sdk_transport_timeout_is_not_retried_and_has_a_trace_id(caplog):
    from openai import OpenAI

    from app.services.session_insights import InsightsTimeoutError

    requests = []

    def transport(request):
        requests.append(request)
        raise httpx.ReadTimeout("sensitive upstream failure", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(transport)) as http_client,
        OpenAI(api_key="secret-test-key", max_retries=0, http_client=http_client) as client,
    ):
        generator = OpenAISessionInsightsGenerator(None, "gpt-6-luna", "none", client=client)
        with pytest.raises(InsightsTimeoutError):
            generator.generate(
                uuid4(), "private prompt", {"segments": [{"text": "private transcript"}]}
            )
    assert len(requests) == 1
    assert requests[0].extensions["timeout"]["read"] == 60
    assert requests[0].extensions["timeout"]["connect"] == 5
    assert requests[0].headers["X-Client-Request-Id"]
    for sensitive in (
        "secret-test-key",
        "sensitive upstream failure",
        "private prompt",
        "private transcript",
    ):
        assert sensitive not in caplog.text
