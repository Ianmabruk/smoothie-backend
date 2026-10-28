"""Pytest fixtures and configuration for the Smoothie AI backend tests."""

import os
import tempfile

import pytest

os.environ.setdefault("FLASK_ENV", "testing")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret-key")
os.environ.setdefault("AI_PROVIDER", "nvidia")
os.environ.setdefault("NVIDIA_API_KEY", "test-nvidia-key-12345")
os.environ.setdefault("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
os.environ.setdefault("NVIDIA_MODEL", "test-model")
os.environ.setdefault("AI_REQUEST_TIMEOUT_SECONDS", "5")
os.environ.setdefault("AI_MAX_RETRIES", "3")
os.environ.setdefault("AI_RETRY_BACKOFF_SECONDS", "0")
os.environ.setdefault("CORS_ORIGINS", "http://localhost:5173")


@pytest.fixture()
def app():
    from app import create_app, db as _db

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    application = create_app()
    application.config["TESTING"] = True
    application.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    with application.app_context():
        _db.create_all()
    yield application
    with application.app_context():
        _db.session.remove()
        _db.drop_all()
    os.unlink(db_path)


@pytest.fixture()
def client(app):
    return app.test_client()
