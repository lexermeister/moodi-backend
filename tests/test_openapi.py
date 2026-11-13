from fastapi.testclient import TestClient


def test_api_documentation_includes_action_deletion(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200

    response = client.get("/openapi.json")

    assert response.status_code == 200
    operation = response.json()["paths"]["/actions/{action_id}"]["delete"]
    assert operation["summary"] == "Permanently delete a rating Action"
    assert "session insights are preserved" in operation["description"]
    assert set(operation["responses"]) >= {"204", "404", "409", "502"}
    resolution = response.json()["paths"][
        "/recordings/{recording_id}/action-candidates/{candidate_id}/resolution"
    ]["get"]
    assert resolution["summary"] == "Resolve a Record rating candidate against current Actions"
