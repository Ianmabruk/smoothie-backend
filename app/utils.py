"""Response helpers and current-user lookup for Smoothie AI."""

from flask import current_app
from flask_jwt_extended import get_jwt_identity


def success(message="", data=None, status=200):
    body = {"success": True}
    if isinstance(message, (list, dict)):
        body["data"] = message
    else:
        if message:
            body["message"] = message
        if data is not None:
            body["data"] = data
    return body, status


def error(message, status=400):
    return {"success": False, "error": message}, status


def get_current_user():
    """Return the logged-in User for the current JWT, or None."""
    from app import db
    from app.models import User

    identity = get_jwt_identity()
    if identity is None:
        return None
    try:
        user_id = int(identity)
    except (TypeError, ValueError):
        return None
    return db.session.get(User, user_id)
