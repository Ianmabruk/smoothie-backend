"""Flask application factory for Smoothie AI."""

import logging
import os
import time

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix

from config import (
    Config,
    DEFAULT_DEV_ORIGINS,
    DEV_JWT_SECRET_KEY,
    DEV_SECRET_KEY,
    build_engine_options,
    is_production_environment,
    resolve_database_url,
)

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()

logger = logging.getLogger("smoothieai.api")


def _validate_production_secrets(app):
    if not is_production_environment():
        return
    missing = []
    if app.config.get("SECRET_KEY") in {None, "", DEV_SECRET_KEY}:
        missing.append("SECRET_KEY")
    if app.config.get("JWT_SECRET_KEY") in {None, "", DEV_JWT_SECRET_KEY}:
        missing.append("JWT_SECRET_KEY")
    if missing:
        raise RuntimeError(
            "Refusing to start in production with default or missing secrets: "
            + ", ".join(missing)
            + ". Set these environment variables in the production dashboard."
        )


def _initialize_database(app):
    if not app.config.get("AUTO_CREATE_TABLES", False):
        return
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            db.create_all()
            return
        except SQLAlchemyError:
            logger.warning(
                "Database initialization attempt %s/%s failed",
                attempt,
                attempts,
                exc_info=True,
            )
            if attempt < attempts:
                time.sleep(min(2 ** attempt, 5))
    logger.error("Database initialization failed after %s attempts", attempts)


def _register_error_handlers(app):
    @app.errorhandler(HTTPException)
    def handle_http_exception(exc):
        response = jsonify(
            {
                "success": False,
                "error": exc.description,
                "status": exc.code,
            }
        )
        response.status_code = exc.code or 500
        return response

    @app.errorhandler(SQLAlchemyError)
    def handle_database_error(exc):
        logger.exception("Database error while handling %s %s", request.method, request.path)
        db.session.rollback()
        return (
            jsonify(
                {
                    "success": False,
                    "error": "The database is temporarily unavailable. Please try again shortly.",
                    "status": 503,
                }
            ),
            503,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc):
        logger.exception("Unhandled error while handling %s %s", request.method, request.path)
        db.session.rollback()
        return (
            jsonify(
                {
                    "success": False,
                    "error": "An unexpected server error occurred.",
                    "status": 500,
                }
            ),
            500,
        )


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    database_url = (
        os.environ.get("DATABASE_URL")
        or app.config.get("SQLALCHEMY_DATABASE_URI")
        or ""
    )
    app.config["SQLALCHEMY_DATABASE_URI"] = resolve_database_url(database_url)
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = build_engine_options(
        app.config["SQLALCHEMY_DATABASE_URI"]
    )
    _validate_production_secrets(app)

    if app.config.get("TRUST_PROXY", True):
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

    _cors_origins = [
        o.strip() for o in (app.config.get("CORS_ORIGINS") or "").split(",") if o.strip()
    ]
    if not _cors_origins:
        _cors_origins = list(DEFAULT_DEV_ORIGINS)
    _supports_credentials = bool(
        app.config.get("CORS_SUPPORTS_CREDENTIALS", True) and _cors_origins
    )
    CORS(
        app,
        resources={r"/api/*": {"origins": _cors_origins}},
        supports_credentials=_supports_credentials,
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "Accept", "X-Requested-With", "X-Request-ID"],
        expose_headers=["Content-Type", "X-Request-ID"],
        max_age=app.config.get("CORS_MAX_AGE", 600),
        automatic_options=True,
    )

    @app.after_request
    def echo_request_id(response):
        request_id = request.headers.get("X-Request-ID")
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    db.init_app(app)
    migrate.init_app(app, db)
    jwt.init_app(app)

    from app import models  # noqa: F401

    with app.app_context():
        _initialize_database(app)

    from app.routes import auth, chat, health

    app.register_blueprint(auth.bp, url_prefix="/api")
    app.register_blueprint(chat.bp, url_prefix="/api")
    app.register_blueprint(health.bp)

    @app.route("/")
    def root():
        return jsonify({"status": "ok", "service": "Smoothie AI API"})

    _register_error_handlers(app)

    return app


app = create_app()
