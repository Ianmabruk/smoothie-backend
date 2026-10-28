"""Authentication routes for Smoothie AI."""

from flask import Blueprint, request
from flask_jwt_extended import create_access_token, jwt_required

from app import db
from app.models import User
from app.utils import success, error, get_current_user

bp = Blueprint("auth", __name__)


def _validate_register_payload(data):
    errors = []
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not name:
        errors.append("name")
    if not email or "@" not in email:
        errors.append("email")
    if not password or len(password) < 6:
        errors.append("password")
    return name, email, password, errors


@bp.route("/auth/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    name, email, password, errors = _validate_register_payload(data)
    if errors:
        return error(f"Missing or invalid fields: {', '.join(errors)}", 400)

    if User.query.filter_by(email=email).first():
        return error("That email address is already registered.", 409)

    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    from flask_jwt_extended import create_access_token

    token = create_access_token(identity=str(user.id))
    return success(
        "Account created successfully",
        {"token": token, "user": user.to_dict()},
        201,
    )


@bp.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return error("Email and password are required.", 400)

    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return error("Invalid email or password.", 401)

    from flask_jwt_extended import create_access_token

    token = create_access_token(identity=str(user.id))
    return success(
        "Signed in successfully",
        {"token": token, "user": user.to_dict()},
    )


@bp.route("/auth/me", methods=["GET"])
@jwt_required()
def me():
    user = get_current_user()
    if not user:
        return error("Not authenticated.", 401)
    return success(data=user.to_dict())
