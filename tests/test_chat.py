"""Tests for chat routes with a mocked NVIDIA provider."""

import json
from unittest.mock import patch

from app.ai_service import AIServiceError


def _auth_headers(client):
    client.post(
        "/api/auth/register",
        data=json.dumps({"name": "Bob", "email": "bob@example.com", "password": "password123"}),
        content_type="application/json",
    )
    resp = client.post(
        "/api/auth/login",
        data=json.dumps({"email": "bob@example.com", "password": "password123"}),
        content_type="application/json",
    )
    token = resp.get_json()["data"]["token"]
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def _mock_reply(content="The capital of Kenya is Nairobi."):
    return {"content": content, "model": "test-model", "usage": {"total_tokens": 10}}


def test_send_message_calls_nvidia_and_persists(client):
    headers = _auth_headers(client)
    # Create a conversation first.
    create = client.post("/api/chat", data=json.dumps({"title": "Math chat"}), headers=headers)
    conv_id = create.get_json()["data"]["id"]

    with patch(
        "app.routes.chat.ai_service.generate_chat_reply",
        return_value=_mock_reply(),
    ) as mock_generate:
        resp = client.post(
            f"/api/chat/{conv_id}/message",
            data=json.dumps({"content": "What is the capital of Kenya?"}),
            headers=headers,
        )

    assert resp.status_code == 201, resp.get_json()
    data = resp.get_json()
    assert data["success"] is True
    assert data["data"]["assistant_message"]["content"] == "The capital of Kenya is Nairobi."
    assert data["data"]["user_message"]["content"] == "What is the capital of Kenya?"
    mock_generate.assert_called_once()
    # System prompt must be part of the messages sent to NVIDIA.
    sent_messages = mock_generate.call_args[0][0]
    assert any(m["role"] == "system" for m in sent_messages)
    assert any(m["role"] == "user" and m["content"] == "What is the capital of Kenya?" for m in sent_messages)


def test_send_message_requires_auth(client):
    resp = client.post(
        "/api/chat/1/message",
        data=json.dumps({"content": "hi"}),
        content_type="application/json",
    )
    assert resp.status_code == 401


def test_send_message_conversation_not_found(client):
    headers = _auth_headers(client)
    resp = client.post(
        "/api/chat/99999/message",
        data=json.dumps({"content": "hi"}),
        headers=headers,
    )
    assert resp.status_code == 404


def test_conversation_ownership_enforced(client):
    headers = _auth_headers(client)
    create = client.post("/api/chat", data=json.dumps({"title": "Private"}), headers=headers)
    conv_id = create.get_json()["data"]["id"]

    # Second user registers and tries to read the first user's conversation.
    client.post(
        "/api/auth/register",
        data=json.dumps({"name": "Eve", "email": "eve@example.com", "password": "password123"}),
        content_type="application/json",
    )
    eve_login = client.post(
        "/api/auth/login",
        data=json.dumps({"email": "eve@example.com", "password": "password123"}),
        content_type="application/json",
    )
    eve_token = eve_login.get_json()["data"]["token"]
    eve_headers = {"Authorization": f"Bearer {eve_token}"}

    resp = client.get(f"/api/chat/{conv_id}", headers=eve_headers)
    assert resp.status_code == 404


def test_send_message_empty_content(client):
    headers = _auth_headers(client)
    create = client.post("/api/chat", data=json.dumps({"title": "x"}), headers=headers)
    conv_id = create.get_json()["data"]["id"]
    resp = client.post(
        f"/api/chat/{conv_id}/message",
        data=json.dumps({"content": "   "}),
        headers=headers,
    )
    assert resp.status_code == 400


def test_provider_failure_returned_honestly(client):
    headers = _auth_headers(client)
    create = client.post("/api/chat", data=json.dumps({"title": "x"}), headers=headers)
    conv_id = create.get_json()["data"]["id"]

    with patch(
        "app.routes.chat.ai_service.generate_chat_reply",
        side_effect=AIServiceError("Invalid NVIDIA API key.", status_code=401, kind="auth"),
    ):
        resp = client.post(
            f"/api/chat/{conv_id}/message",
            data=json.dumps({"content": "hi"}),
            headers=headers,
        )
    assert resp.status_code == 401
    assert resp.get_json()["success"] is False


def test_list_conversations_requires_auth(client):
    resp = client.get("/api/chat")
    assert resp.status_code == 401
