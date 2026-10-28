"""Chat routes: conversations and AI-generated messages via NVIDIA NIM."""

import re

from flask import Blueprint, request
from flask_jwt_extended import jwt_required

from app import ai_service, db
from app.ai_service import AIServiceError, SYSTEM_PROMPT
from app import calc_tools
from app.models import Conversation, Message
from app.utils import success, error, get_current_user

bp = Blueprint("chat", __name__)

MAX_HISTORY_MESSAGES = 12


def _conversation_history(conversation, limit=MAX_HISTORY_MESSAGES):
    messages = (
        Message.query.filter_by(conversation_id=conversation.id)
        .order_by(Message.created_at.asc())
        .all()
    )
    history = [
        {"role": m.role, "content": m.content} for m in messages[-limit:]
    ]
    return history


def _detect_calculation(text: str):
    """Detect explicit calculation requests and return a structured result.

    Returns ``None`` when the message is not a calculation request. The
    detector is intentionally conservative: it only handles patterns it can
    solve exactly, so the AI never fabricates working.
    """
    lower = text.strip().lower()

    # --- Simple / compound interest ---
    m = re.search(
        r"(simple|compound)?\s*interest.*?([\d,.]+)\s*(?:at|@)?\s*([\d.]+)\s*%\s*(?:per\s+year|/year|p\.a\.|annum)?\s*(?:for\s+)?([\d.]+)\s*year",
        lower,
    )
    if m:
        kind = (m.group(1) or "simple").strip()
        principal = float(m.group(2).replace(",", ""))
        rate = float(m.group(3))
        years = float(m.group(4))
        if kind.startswith("comp"):
            return calc_tools.compound_interest(principal, rate, years)
        return calc_tools.simple_interest(principal, rate, years)

    # --- Linear equation: ax + b = c ---
    m = re.search(
        r"\s*([\d.]+)\s*x\s*([+-]\s*[\d.]+)\s*=\s*([\d.]+)\s*$",
        text.strip(),
    )
    if m:
        return calc_tools.linear_equation(
            float(m.group(1)), float(m.group(2)), float(m.group(3))
        )

    # --- Quadratic equation: ax^2 + bx + c = 0 ---
    m = re.search(
        r"\s*([\d.]+)\s*x\^2\s*([+-]\s*[\d.]+)\s*x\s*([+-]\s*[\d.]+)\s*=\s*0\s*$",
        text.strip(),
    )
    if m:
        return calc_tools.quadratic_equation(
            float(m.group(1)), float(m.group(2)), float(m.group(3))
        )

    # --- Unit conversion: convert X unit1 to unit2 ---
    m = re.search(
        r"convert\s+([\d.]+)\s+([a-zA-Z]+)\s+to\s+([a-zA-Z]+)", lower
    )
    if m:
        return calc_tools.convert_units(
            float(m.group(1)), m.group(2), m.group(3)
        )

    # --- Basic arithmetic expression ---
    if re.fullmatch(r"[\d\s\+\-\*/\^\(\)\.%,]+", text.strip()) and any(
        ch in text for ch in "+-*/^"
    ):
        return calc_tools.evaluate_expression(text.strip())

    # --- Percentage: X% of Y ---
    m = re.search(r"([\d.]+)\s*%\s*(?:of\s+)?([\d,.]+)", lower)
    if m and "interest" not in lower:
        percent = float(m.group(1))
        value = float(m.group(2).replace(",", ""))
        result = value * percent / 100
        steps = [
            {"label": "Identify the values", "detail": f"Value = {value}; percentage = {percent}%."},
            {"label": "Formula", "detail": "Result = value x percentage / 100"},
            {"label": "Substitute", "detail": f"Result = {value} x {percent} / 100"},
            {"label": "Calculate", "detail": f"Result = {value * percent} / 100 = {result}"},
            {"label": "Final answer", "detail": f"{percent}% of {value} is {result}."},
        ]
        return {"ok": True, "tool": "percentage", "result": result, "steps": steps}

    return None


def _build_ai_messages(conversation, user_content, calc_result=None):
    system_prompt = SYSTEM_PROMPT
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in _conversation_history(conversation):
        messages.append(item)
    if calc_result is not None:
        calc_summary = (
            "CALCULATION TOOL RESULT (verified, do not fabricate):\n"
            + "\n".join(f"{s['label']}: {s['detail']}" for s in calc_result.get("steps", []))
            + f"\nFinal result: {calc_result.get('result')}"
        )
        messages.append(
            {
                "role": "system",
                "content": calc_summary,
            }
        )
    messages.append({"role": "user", "content": user_content})
    return messages


@bp.route("/chat", methods=["GET"])
@jwt_required()
def list_conversations():
    user = get_current_user()
    if not user:
        return error("Unauthorized", 401)
    conversations = (
        Conversation.query.filter_by(user_id=user.id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )
    return success([c.to_dict() for c in conversations])


@bp.route("/chat", methods=["POST"])
@jwt_required()
def create_conversation():
    user = get_current_user()
    if not user:
        return error("Unauthorized", 401)
    data = request.get_json(silent=True) or {}
    title = (data.get("title") or "New chat").strip() or "New chat"
    conversation = Conversation(user_id=user.id, title=title)
    db.session.add(conversation)
    db.session.commit()
    return success("Conversation created", conversation.to_dict(), 201)


@bp.route("/chat/<int:conversation_id>", methods=["GET"])
@jwt_required()
def get_conversation(conversation_id):
    user = get_current_user()
    if not user:
        return error("Unauthorized", 401)
    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=user.id
    ).first()
    if not conversation:
        return error("Conversation not found", 404)
    return success(
        {
            "conversation": conversation.to_dict(),
            "messages": [m.to_dict() for m in conversation.messages],
        }
    )


@bp.route("/chat/<int:conversation_id>/message", methods=["POST"])
@jwt_required()
def send_message(conversation_id):
    user = get_current_user()
    if not user:
        return error("Unauthorized", 401)

    conversation = Conversation.query.filter_by(
        id=conversation_id, user_id=user.id
    ).first()
    if not conversation:
        return error("Conversation not found", 404)

    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    if not content:
        return error("content is required", 400)
    if len(content) > 4000:
        return error("Message is too long (max 4000 characters).", 400)

    user_message = Message(conversation_id=conversation.id, role="user", content=content)
    db.session.add(user_message)

    if not conversation.title or conversation.title == "New chat":
        title = content[:30].strip() or "New chat"
        if len(title) >= 30:
            title = title[:27].rstrip() + "..."
        conversation.title = title
    db.session.commit()

    calc_result = _detect_calculation(content)
    messages = _build_ai_messages(conversation, content, calc_result)

    try:
        reply = ai_service.generate_chat_reply(messages)
    except AIServiceError as exc:
        return error(str(exc), exc.status_code or 503)
    except Exception as exc:  # noqa: BLE001 - never leak internals to the client
        return error("The AI request failed. Please try again.", 502)

    assistant_message = Message(
        conversation_id=conversation.id, role="assistant", content=reply["content"]
    )
    db.session.add(assistant_message)
    db.session.commit()

    return success(
        "Message saved",
        {
            "user_message": user_message.to_dict(),
            "assistant_message": assistant_message.to_dict(),
            "conversation": conversation.to_dict(),
            "calculation": calc_result,
        },
        201,
    )


@bp.route("/calc", methods=["POST"])
def calculate():
    """Standalone safe calculation endpoint used to verify numerical results."""
    data = request.get_json(silent=True) or {}
    expression = data.get("expression") or data.get("query") or ""
    if not expression or not str(expression).strip():
        return error("expression is required", 400)
    result = calc_tools.evaluate_expression(str(expression))
    if not result.get("ok"):
        return error(result.get("error", "Calculation failed."), 400)
    return success(data=result)
