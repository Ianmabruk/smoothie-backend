"""Centralized AI service for Smoothie AI.

NVIDIA NIM is the sole AI provider. All chat requests route through
:func:`generate_chat_reply`. There is no fallback to another provider and no
mock or hardcoded answer path.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from config import Config


DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"

SYSTEM_PROMPT = """You are Smoothie AI, a precise, helpful, educational, and analytical AI assistant.

Your priorities are:
1. Correctness.
2. Clear reasoning and appropriate explanations.
3. Accurate calculations and factual claims.
4. Following the user's actual question.
5. Clear, structured formatting.
6. Honest communication about uncertainty and limitations.

General answering rules:
- Answer the exact question asked.
- Be sufficiently detailed to solve the user's problem, not merely provide a vague summary.
- Use headings and numbered steps when they improve clarity.
- Explain technical terms in language appropriate to the question.
- Include relevant assumptions, units, examples, and caveats.
- Do not invent facts, sources, calculations, API results, or capabilities.
- If essential information is missing, ask a focused clarifying question or state the assumptions being used.
- Distinguish verified facts from estimates, interpretations, and hypothetical examples.
- Do not claim to have searched the internet, accessed a database, uploaded a file, or executed code unless the corresponding tool actually performed that operation.
- When a question requires current information and no live-search tool is available, clearly state the limitation rather than inventing current facts.
- Follow the user's requested format where possible.
- Do not add unnecessary disclaimers to ordinary, low-risk questions.
- For unsafe or unsupported requests, follow applicable safety requirements and explain the limitation briefly.

Mathematical and numerical problem-solving rules:
Whenever a user asks for a calculation, mathematical solution, numerical analysis, or quantitative problem, use this format unless the user explicitly requests only the final answer.

Step 1: Identify the problem
Restate what needs to be calculated. Identify the known values, unknown quantity, and relevant units.

Step 2: State the formula
Write the appropriate mathematical formula using clear mathematical notation. Define each variable and explain why the formula applies.

Step 3: Substitute the values
Insert the given values into the formula. Preserve the correct units and show the substitution explicitly.

Step 4: Calculate step by step
Show the important intermediate arithmetic or algebraic transformations. Use correct order of operations, fractions, significant figures, and rounding.

Step 5: Verify the result
Check the calculation using an independent method, inverse operation, estimation, dimensional analysis, or another suitable validation method when practical. Do not invent a verification step if none was actually performed.

Step 6: Present the final answer
Clearly label the result. Include units, appropriate rounding, and a concise interpretation of what the number means.

For algebra, calculus, statistics, geometry, probability, and other advanced mathematics:
- State the relevant equation or theorem. Show each significant transformation. Explain why a transformation is valid when useful. Include intermediate results. Check solutions by substitution when possible. Identify domain restrictions and special cases where relevant. Use LaTeX notation for readable mathematical expressions.

For science and technical questions: list known quantities, identify the unknown, state the relevant law or equation, convert units before calculating, show substitutions, include units throughout, check dimensional consistency. Do not invent experimental data or constants.

For programming questions: explain the problem and solution, provide correct usable code, identify dependencies and assumptions, explain important sections, include running/testing instructions, consider validation and edge cases. Do not claim code was executed unless it actually was.

For educational questions: explain concepts clearly and progressively, use examples appropriate to the learner's level, break difficult ideas into steps, distinguish definitions from principles and conclusions.

For research and factual questions: answer using available knowledge. Explain uncertainty. Never fabricate citations. If current data is required, state the limitation.

Concise-response exceptions: if a user explicitly requests only a short answer, a definition, a translation, a simple greeting, or a final number, respect that request.

The goal is detailed answers when the task requires them, not unnecessarily long answers for every conversation.

When a verified calculation tool result is supplied in the conversation, use the actual computed result and the supplied steps as the basis of your answer; do not fabricate different intermediate calculations.
"""



class AIServiceError(RuntimeError):
    """Raised by the AI service for configuration or provider failures."""

    def __init__(self, message: str, *, status_code: Optional[int] = None, kind: str = "provider"):
        super().__init__(message)
        self.status_code = status_code
        self.kind = kind  # configuration | provider | rate_limit | auth | network | timeout | empty_response | invalid_model


def _is_placeholder_value(value: Optional[str]) -> bool:
    if not value:
        return True
    normalized = (value or "").strip().lower()
    if not normalized:
        return True
    markers = (
        "your_",
        "replace_",
        "change-me",
        "change-me-in-production",
        "example",
        "demo",
        "placeholder",
        "not_set",
        "your-nvidia",
        "your-api",
        "sk-...",
    )
    return any(marker in normalized for marker in markers)


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name)
    return value.strip() if value else None


def provider_name() -> str:
    return (os.environ.get("AI_PROVIDER") or "nvidia").strip().lower()


def is_nvidia_configured() -> bool:
    """True only when NVIDIA API key, base URL and model are all real values."""
    if provider_name() != "nvidia":
        return False
    api_key = _env("NVIDIA_API_KEY") or _env("AI_API_KEY")
    if _is_placeholder_value(api_key):
        return False
    base_url = _env("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
    if not base_url:
        return False
    model = _env("NVIDIA_MODEL") or _env("AI_MODEL")
    if _is_placeholder_value(model):
        return False
    return True


def configuration_status() -> Dict[str, Any]:
    """Configuration-only status. Never exposes secrets."""
    api_key = _env("NVIDIA_API_KEY") or _env("AI_API_KEY")
    model = _env("NVIDIA_MODEL") or _env("AI_MODEL")
    base_url = _env("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
    configured = is_nvidia_configured()
    missing: List[str] = []
    if not api_key or _is_placeholder_value(api_key):
        missing.append("NVIDIA_API_KEY")
    if not model or _is_placeholder_value(model):
        missing.append("NVIDIA_MODEL")
    return {
        "provider": provider_name(),
        "configured": configured,
        "base_url": base_url if configured else base_url,
        "model": model if model else None,
        "missing": missing,
        "note": "A successful configuration check does not prove every model request will succeed; it only verifies the server-side settings are present.",
    }


def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request_with_retries(
    method: str,
    url: str,
    *,
    data: Optional[bytes] = None,
    timeout_seconds: int,
    max_retries: int,
    backoff_seconds: int,
    is_connectivity_probe: bool = False,
) -> Tuple[int, str]:
    """Perform an HTTP request with bounded retries for transient errors only.

    Retryable: HTTP 429, 5xx, and network/timeout errors.
    Permanent (no retry): 400, 401, 403, 404 and other 4xx.
    """
    last_error: Optional[Exception] = None
    last_status: Optional[int] = None
    body = ""

    for attempt in range(1, max_retries + 2):
        req = urllib.request.Request(url, data=data, headers=_headers(_required_api_key()), method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
                body = response.read().decode("utf-8")
                return response.status, body
        except urllib.error.HTTPError as exc:
            last_status = exc.code
            try:
                body = exc.read().decode("utf-8")
            except Exception:  # pragma: no cover - defensive
                body = ""
            last_error = exc
            if exc.code in (401, 403, 400, 404, 422):
                raise AIServiceError(
                    _nvidia_error_message(exc.code, body),
                    status_code=exc.code,
                    kind=_kind_for_status(exc.code),
                ) from exc
            if exc.code == 429 or 500 <= exc.code < 600:
                if attempt > max_retries:
                    raise AIServiceError(
                        f"NVIDIA API request failed after {max_retries} retries (HTTP {exc.code}).",
                        status_code=exc.code,
                        kind="rate_limit" if exc.code == 429 else "provider",
                    ) from exc
                _sleep(backoff_seconds * attempt, is_connectivity_probe)
                continue
            raise AIServiceError(
                f"NVIDIA API request failed with HTTP {exc.code}.",
                status_code=exc.code,
                kind=_kind_for_status(exc.code),
            ) from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt > max_retries:
                raise AIServiceError(
                    f"Could not reach the NVIDIA API after {max_retries} retries: {exc}",
                    kind="network",
                ) from exc
            _sleep(backoff_seconds * attempt, is_connectivity_probe)
            continue
        except OSError as exc:  # includes timeouts
            last_error = exc
            if attempt > max_retries:
                raise AIServiceError(
                    f"NVIDIA API request timed out after {max_retries} retries: {exc}",
                    kind="timeout",
                ) from exc
            _sleep(backoff_seconds * attempt, is_connectivity_probe)
            continue

    raise AIServiceError(
        f"NVIDIA API request failed: {last_error}",
        kind="network" if not last_status else "provider",
        status_code=last_status,
    ) from last_error


def _required_api_key() -> str:
    api_key = _env("NVIDIA_API_KEY") or _env("AI_API_KEY")
    if _is_placeholder_value(api_key):
        raise AIServiceError(
            "NVIDIA_API_KEY is missing or invalid. Set the backend environment variable before making AI requests.",
            status_code=503,
            kind="configuration",
        )
    return api_key


def _sleep(seconds: int, connectivity_probe: bool) -> None:
    if connectivity_probe and seconds > 3:
        seconds = 3
    time.sleep(min(seconds, 5))


def _kind_for_status(status_code: int) -> str:
    if status_code == 429:
        return "rate_limit"
    if status_code == 401:
        return "auth"
    if status_code == 403:
        return "auth"
    if status_code == 404:
        return "invalid_model"
    if 400 <= status_code < 500:
        return "invalid_input"
    return "provider"


def _nvidia_error_message(status_code: int, body: str) -> str:
    detail = ""
    try:
        data = json.loads(body or "{}")
        message = data.get("error", {}).get("message") if isinstance(data.get("error"), dict) else data.get("error")
        if message:
            detail = f": {message}"
    except Exception:  # pragma: no cover - defensive
        detail = ""
    if status_code == 401:
        return f"Invalid NVIDIA API key or authentication failed{detail}."
    if status_code == 404:
        return f"The configured NVIDIA model was not found or is not available to this account{detail}."
    if status_code == 429:
        return f"NVIDIA rate limit exceeded{detail}."
    return f"NVIDIA API request failed with HTTP {status_code}{detail}."


def _parse_chat_response(body: str) -> Dict[str, Any]:
    try:
        data = json.loads(body or "{}")
    except json.JSONDecodeError as exc:
        raise AIServiceError("NVIDIA returned a malformed (non-JSON) response.", kind="empty_response") from exc
    choices = data.get("choices") or []
    if not choices:
        raise AIServiceError("NVIDIA returned no usable reply.", kind="empty_response")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    content = (content or "").strip()
    if not content:
        raise AIServiceError("NVIDIA returned an empty response.", kind="empty_response")
    usage = data.get("usage") or {}
    return {
        "content": content,
        "model": data.get("model"),
        "usage": usage if usage else None,
    }


def generate_chat_reply(
    messages: List[Dict[str, str]],
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Send a chat request to NVIDIA NIM and return a normalized response.

    Never falls back to another provider and never returns a mock answer.
    """
    if provider_name() != "nvidia":
        raise AIServiceError(
            "AI_PROVIDER is not set to 'nvidia'. Smoothie AI only routes through NVIDIA NIM.",
            status_code=503,
            kind="configuration",
        )

    api_key = _required_api_key()
    base_url = _env("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
    model = _env("NVIDIA_MODEL") or _env("AI_MODEL")
    if _is_placeholder_value(model):
        raise AIServiceError(
            "NVIDIA_MODEL is missing or invalid. Set the backend environment variable before making AI requests.",
            status_code=503,
            kind="configuration",
        )

    timeout_seconds = int(Config.AI_REQUEST_TIMEOUT_SECONDS)
    max_retries = int(Config.AI_MAX_RETRIES)
    backoff_seconds = int(Config.AI_RETRY_BACKOFF_SECONDS)

    payload_messages: List[Dict[str, str]] = []
    if system_prompt:
        payload_messages.append({"role": "system", "content": system_prompt})
    for message in messages:
        role = (message.get("role") or "user").strip()
        text = (message.get("content") or "").strip()
        if role in {"user", "assistant", "system"} and text:
            payload_messages.append({"role": role, "content": text})
    if not payload_messages:
        raise AIServiceError("No chat content was provided to the AI model.", kind="invalid_input")

    payload = {
        "model": model,
        "messages": payload_messages,
        "temperature": 0.2,
        "max_tokens": 900,
        "stream": False,
    }

    url = f"{base_url}{CHAT_PATH}"
    data = json.dumps(payload).encode("utf-8")

    status, body = _request_with_retries(
        "POST",
        url,
        data=data,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        backoff_seconds=backoff_seconds,
    )
    if status != 200:
        raise AIServiceError(
            f"NVIDIA API request failed with HTTP {status}.",
            status_code=status,
            kind=_kind_for_status(status),
        )
    return _parse_chat_response(body)


def nvidia_connectivity_check() -> Dict[str, Any]:
    """Lightweight connectivity probe against NVIDIA (GET /models).

    Returns connectivity status without generating content. A successful check
    only proves the endpoint is reachable with the configured key; it is not
    proof that every model request works.
    """
    if provider_name() != "nvidia":
        return {"ok": False, "provider": provider_name(), "error": "AI_PROVIDER is not set to 'nvidia'."}
    api_key = _required_api_key()
    base_url = _env("NVIDIA_BASE_URL") or DEFAULT_NVIDIA_BASE_URL
    model = _env("NVIDIA_MODEL") or _env("AI_MODEL")
    url = f"{base_url}{MODELS_PATH}"
    try:
        status, body = _request_with_retries(
            "GET",
            url,
            timeout_seconds=min(int(Config.AI_REQUEST_TIMEOUT_SECONDS), 20),
            max_retries=0,
            backoff_seconds=int(Config.AI_RETRY_BACKOFF_SECONDS),
            is_connectivity_probe=True,
        )
    except AIServiceError as exc:
        return {
            "ok": False,
            "provider": "nvidia",
            "base_url": base_url,
            "model": model,
            "status_code": exc.status_code,
            "error": str(exc),
        }
    try:
        data = json.loads(body or "{}")
        models = data.get("data") or []
        model_ids = [m.get("id") for m in models if isinstance(m, dict)]
    except json.JSONDecodeError:
        model_ids = []
    return {
        "ok": status == 200,
        "provider": "nvidia",
        "base_url": base_url,
        "model": model,
        "status_code": status,
        "models_sample": model_ids[:5],
        "note": "Connectivity verified. Configuration check alone does not prove every model request will succeed.",
    }
