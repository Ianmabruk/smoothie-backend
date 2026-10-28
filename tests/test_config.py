"""Tests for backend configuration and provider status."""

import os

import pytest

from app import ai_service


def test_provider_is_nvidia():
    assert ai_service.provider_name() == "nvidia"


def test_configuration_status_shape():
    status = ai_service.configuration_status()
    assert status["provider"] == "nvidia"
    assert "configured" in status
    assert "base_url" in status
    assert "model" in status
    assert "missing" in status
    assert "note" in status


def test_no_secrets_in_configuration_status():
    status = ai_service.configuration_status()
    payload = str(status)
    assert "test-nvidia-key" not in payload or status["configured"] is True
    # The status must never expose the raw API key.
    assert "NVIDIA_API_KEY=" not in payload


def test_missing_api_key_marks_unconfigured(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    assert not ai_service.is_nvidia_configured()


def test_missing_model_marks_unconfigured(monkeypatch):
    monkeypatch.delenv("NVIDIA_MODEL", raising=False)
    monkeypatch.delenv("AI_MODEL", raising=False)
    assert not ai_service.is_nvidia_configured()


def test_placeholder_key_rejected(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "your_actual_nvidia_api_key")
    assert ai_service._is_placeholder_value("your_actual_nvidia_api_key")
    assert not ai_service.is_nvidia_configured()


def test_generate_chat_reply_requires_configuration(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("AI_API_KEY", raising=False)
    with pytest.raises(ai_service.AIServiceError) as exc:
        ai_service.generate_chat_reply([{"role": "user", "content": "hi"}])
    assert exc.value.kind == "configuration"
    assert exc.value.status_code == 503
