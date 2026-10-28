"""Health and provider-status routes for Smoothie AI."""

from flask import Blueprint, jsonify

from app import ai_service

bp = Blueprint("health", __name__)


@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Smoothie AI API"})


@bp.route("/health/ready", methods=["GET"])
def ready():
    return jsonify({"status": "ok", "service": "Smoothie AI API", "ai_provider": "nvidia"})


@bp.route("/api/ai/provider-status", methods=["GET"])
def provider_status():
    """Server-side provider configuration and connectivity status.

    Never returns secrets. A successful configuration check is not proof that
    every model request will succeed.
    """
    return jsonify(ai_service.configuration_status() | ai_service.nvidia_connectivity_check())
