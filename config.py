import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv

# Load backend-only environment variables from .env (never committed).
# The frontend must never read this file.
load_dotenv()

DEV_SECRET_KEY = "dev-secret-key-change-in-production"
DEV_JWT_SECRET_KEY = "dev-jwt-secret-change-in-production"

# Origins used by local development servers only.
DEFAULT_DEV_ORIGINS = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
)


def is_production_environment():
    return (
        (os.environ.get("FLASK_ENV") or "").lower() in {"production", "prod"}
        or (os.environ.get("APP_ENV") or "").lower() in {"production", "prod"}
        or bool(os.environ.get("RENDER"))
    )


def _env_bool(name, default=False):
    value = (os.environ.get(name) or "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _env_int(name, default):
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _normalize_postgresql_url(url):
    parsed = urlsplit(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("sslmode", "require")
    query.setdefault("connect_timeout", "10")
    return urlunsplit(parsed._replace(query=urlencode(query, doseq=True)))


def resolve_database_url(raw_url=None):
    """Return the SQLAlchemy database URI.

    The application must not silently fall back to SQLite in production. A
    missing or broken DATABASE_URL is a startup failure so the service does not
    appear healthy while pointed at the wrong database. A local SQLite URL is
    only permitted explicitly for isolated test runs when TESTING is enabled.
    """
    url = (raw_url if raw_url else os.environ.get("DATABASE_URL") or "").strip()

    if not url:
        if _env_bool("TESTING", False):
            base_dir = os.path.abspath(os.path.dirname(__file__))
            instance_dir = os.path.join(base_dir, "instance")
            os.makedirs(instance_dir, exist_ok=True)
            return f"sqlite:///{os.path.join(instance_dir, 'test_smoothie_ai.db')}"
        raise RuntimeError(
            "DATABASE_URL is not configured. Set DATABASE_URL to the Neon PostgreSQL "
            "connection string before starting the Flask backend."
        )

    if url.startswith("sqlite"):
        if _env_bool("TESTING", False):
            if url.startswith("sqlite://") and not url.startswith("sqlite:////"):
                relative_path = url.replace("sqlite:///", "", 1)
                if not os.path.isabs(relative_path):
                    base_dir = os.path.abspath(os.path.dirname(__file__))
                    resolved = os.path.join(base_dir, relative_path)
                    os.makedirs(os.path.dirname(resolved), exist_ok=True)
                    return f"sqlite:///{resolved}"
            return url
        raise RuntimeError(
            "SQLite is not supported for the application database. The project requires "
            "Neon PostgreSQL via DATABASE_URL and will refuse to start with a local SQLite URL."
        )

    if url.startswith(("postgresql://", "postgresql+psycopg://", "postgresql+psycopg2://")):
        return _normalize_postgresql_url(url)

    return url


def build_engine_options(database_uri):
    options = {"pool_pre_ping": True}
    if database_uri.startswith("sqlite"):
        if _env_bool("TESTING", False):
            options["connect_args"] = {"check_same_thread": False, "timeout": 30}
            return options
        raise RuntimeError(
            "SQLite engine configuration is not allowed for the application database. "
            "Use a PostgreSQL DATABASE_URL instead."
        )

    options.update(
        {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
            "pool_recycle": 280,
            "connect_args": {"connect_timeout": 10, "sslmode": "require"},
        }
    )
    return options


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY") or DEV_SECRET_KEY
    JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or DEV_JWT_SECRET_KEY
    JWT_ACCESS_TOKEN_EXPIRES = _env_int("JWT_ACCESS_TOKEN_EXPIRES_SECONDS", 60 * 60 * 24)
    JWT_TOKEN_LOCATION = ["headers"]
    JWT_HEADER_NAME = "Authorization"
    JWT_HEADER_TYPE = "Bearer"

    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or ""
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    CORS_ORIGINS = os.environ.get("CORS_ORIGINS") or ""
    CORS_SUPPORTS_CREDENTIALS = _env_bool("CORS_SUPPORTS_CREDENTIALS", True)
    CORS_MAX_AGE = _env_int("CORS_MAX_AGE_SECONDS", 600)

    TRUST_PROXY = _env_bool("TRUST_PROXY", True)
    AUTO_CREATE_TABLES = _env_bool("AUTO_CREATE_TABLES", False)

    # --- NVIDIA NIM (server-side only) ---
    AI_PROVIDER = (os.environ.get("AI_PROVIDER") or "nvidia").strip().lower()
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
    NVIDIA_BASE_URL = (os.environ.get("NVIDIA_BASE_URL") or "https://integrate.api.nvidia.com/v1").strip().rstrip("/")
    NVIDIA_MODEL = os.environ.get("NVIDIA_MODEL")

    # Compatibility aliases.
    AI_API_KEY = os.environ.get("AI_API_KEY")
    AI_API_BASE_URL = os.environ.get("AI_API_BASE_URL")
    AI_MODEL = os.environ.get("AI_MODEL")

    # Request behaviour.
    AI_REQUEST_TIMEOUT_SECONDS = _env_int("AI_REQUEST_TIMEOUT_SECONDS", 45)
    AI_MAX_RETRIES = _env_int("AI_MAX_RETRIES", 2)
    AI_RETRY_BACKOFF_SECONDS = _env_int("AI_RETRY_BACKOFF_SECONDS", 2)

    # Safety limits.
    MAX_MESSAGE_CHARS = _env_int("MAX_MESSAGE_CHARS", 4000)
    MAX_CONTEXT_MESSAGES = _env_int("MAX_CONTEXT_MESSAGES", 12)
