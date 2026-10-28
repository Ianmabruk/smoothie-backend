"""Tests for the NVIDIA AI service (provider responses are mocked)."""

import json
import urllib.error
from unittest.mock import patch

import pytest

from app import ai_service
from app.ai_service import AIServiceError

SUCCESS_BODY = json.dumps(
    {
        "id": "cmpl-test",
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": "The answer is 42."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
)


class _FakeResponse:
    def __init__(self, status, body):
        self.status = status
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, code, body=b"{}"):
        self.code = code
        self._body = body
        super().__init__("http://example.com/v1/chat/completions", code, "msg", {}, None)

    def read(self):
        return self._body


def _mock_urlopen(response=None, error=None):
    def opener(req, timeout=None):
        if error is not None:
            raise error
        return response
    return opener


def test_successful_chat_reply_is_normalized():
    with patch("app.ai_service.urllib.request.urlopen", _mock_urlopen(response=_FakeResponse(200, SUCCESS_BODY))):
        result = ai_service.generate_chat_reply(
            [{"role": "user", "content": "What is 6 times 7?"}]
        )
    assert result["content"] == "The answer is 42."
    assert result["model"] == "test-model"
    assert result["usage"]["total_tokens"] == 15


def test_invalid_api_key_returns_auth_error():
    with patch(
        "app.ai_service.urllib.request.urlopen",
        _mock_urlopen(error=_FakeHTTPError(401, b'{"error":{"message":"Invalid API key"}}')),
    ):
        with pytest.raises(AIServiceError) as exc:
            ai_service.generate_chat_reply([{"role": "user", "content": "hi"}])
    assert exc.value.kind == "auth"
    assert exc.value.status_code == 401


def test_invalid_model_returns_invalid_model_error():
    with patch(
        "app.ai_service.urllib.request.urlopen",
        _mock_urlopen(error=_FakeHTTPError(404, b'{"error":{"message":"model not found"}}')),
    ):
        with pytest.raises(AIServiceError) as exc:
            ai_service.generate_chat_reply([{"role": "user", "content": "hi"}])
    assert exc.value.kind == "invalid_model"
    assert exc.value.status_code == 404


def test_rate_limit_is_retried_then_fails():
    calls = {"n": 0}

    def opener(req, timeout=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _FakeHTTPError(429, b'{"error":{"message":"rate limited"}}')
        return _FakeResponse(200, SUCCESS_BODY)

    with patch("app.ai_service.urllib.request.urlopen", opener):
        result = ai_service.generate_chat_reply([{"role": "user", "content": "hi"}])
    assert result["content"] == "The answer is 42."
    assert calls["n"] == 3


def test_permanent_errors_are_not_retried():
    calls = {"n": 0}

    def opener(req, timeout=None):
        calls["n"] += 1
        raise _FakeHTTPError(401, b"unauthorized")

    with patch("app.ai_service.urllib.request.urlopen", opener):
        with pytest.raises(AIServiceError) as exc:
            ai_service.generate_chat_reply([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1  # no retry for 401
    assert exc.value.kind == "auth"


def test_network_timeout_fails_after_retries():
    import urllib.error

    def opener(req, timeout=None):
        raise urllib.error.URLError("timed out")

    with patch("app.ai_service.urllib.request.urlopen", opener):
        with pytest.raises(AIServiceError) as exc:
            ai_service.generate_chat_reply([{"role": "user", "content": "hi"}])
    assert exc.value.kind in ("network", "timeout")


def test_empty_response_raises():
    with patch(
        "app.ai_service.urllib.request.urlopen",
        _mock_urlopen(response=_FakeResponse(200, json.dumps({"choices": []}))),
    ):
        with pytest.raises(AIServiceError) as exc:
            ai_service.generate_chat_reply([{"role": "user", "content": "hi"}])
    assert exc.value.kind == "empty_response"


def test_malformed_response_raises():
    with patch(
        "app.ai_service.urllib.request.urlopen",
        _mock_urlopen(response=_FakeResponse(200, "not json")),
    ):
        with pytest.raises(AIServiceError) as exc:
            ai_service.generate_chat_reply([{"role": "user", "content": "hi"}])
    assert exc.value.kind == "empty_response"


def test_no_mock_or_fallback_path():
    # Configuration status reports the real provider only.
    status = ai_service.configuration_status()
    assert status["provider"] == "nvidia"
    assert "gemini" not in status["provider"]
    assert "ollama" not in status["provider"]
    assert "groq" not in status["provider"]
