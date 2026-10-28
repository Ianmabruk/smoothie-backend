"""Tests for authentication routes."""

import json

from app.models import User


def _register(client, email="alice@example.com", name="Alice", password="password123"):
    return client.post(
        "/api/auth/register",
        data=json.dumps({"name": name, "email": email, "password": password}),
        content_type="application/json",
    )


def _login(client, email="alice@example.com", password="password123"):
    return client.post(
        "/api/auth/login",
        data=json.dumps({"email": email, "password": password}),
        content_type="application/json",
    )


def _auth_headers(client, email="alice@example.com", password="password123"):
    resp = _login(client, email, password)
    token = resp.get_json()["data"]["token"]
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def test_register_creates_user(client):
    resp = _register(client)
    assert resp.status_code == 201
    data = resp.get_json()
    assert data["success"] is True
    assert data["data"]["user"]["email"] == "alice@example.com"
    assert data["data"]["user"]["name"] == "Alice"
    assert "token" in data["data"]


def test_register_duplicate_email(client):
    _register(client)
    resp = _register(client)
    assert resp.status_code == 409
    assert resp.get_json()["success"] is False


def test_register_missing_fields(client):
    resp = client.post(
        "/api/auth/register",
        data=json.dumps({"name": "Bob"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_login_success(client):
    _register(client)
    resp = _login(client)
    assert resp.status_code == 200
    assert resp.get_json()["success"] is True


def test_login_invalid_credentials(client):
    _register(client)
    resp = _login(client, password="wrongpassword")
    assert resp.status_code == 401


def test_me_requires_auth(client):
    resp = client.get("/api/auth/me")
    assert resp.status_code == 401


def test_me_returns_user(client):
    _register(client)
    headers = _auth_headers(client)
    resp = client.get("/api/auth/me", headers=headers)
    assert resp.status_code == 200
    assert resp.get_json()["data"]["email"] == "alice@example.com"
