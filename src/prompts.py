#!/usr/bin/env python3
"""Prompt templates for CLR methods, extraction, and repair."""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPTS = {
    "direct": (
        "You solve multi-turn logical constraint satisfaction problems. "
        "Track all prior commitments and keep the final assignment consistent with every active constraint. "
        "Return only the final JSON solution."
    ),
    "cot": (
        "You solve multi-turn logical constraint satisfaction problems. "
        "Track all prior commitments and reason briefly before answering. "
        "Output at most 3 short bullets, then the final JSON solution."
    ),
    "ledger_only": (
        "You solve a multi-turn logical constraint problem with an explicit ledger. "
        "Treat the ledger as committed state and keep your answer consistent with it and the new turn constraints. "
        "Return only JSON."
    ),
    "mus_repair": (
        "You solve a multi-turn logical constraint problem with formal consistency checks and targeted repair signals. "
        "Use the ledger as committed state, address any repair signal directly, and provide a complete consistent solution. "
        "Return only JSON."
    ),
}


def _domain_schema_hint(domain: str, entities: list[str], context: dict[str, Any] | None = None) -> str:
    context = context or {}
    if domain == "seating":
        return (
            "Output JSON as {\"solution\": {\"Entity\": position_int, ...}} "
            "with positions as 1-indexed seat numbers. "
            "Include all entities and do not add extra text."
        )

    if domain == "scheduling":
        return (
            "Output JSON as {\"solution\": {\"Event\": {\"start\": int, \"duration\": int}, ...}}. "
            "If duration is unknown, you may still provide start only per event. "
            "Include all entities and do not add extra text."
        )

    if domain == "logic_grid":
        cats = ", ".join(context.get("categories", {}).keys())
        return (
            "Output JSON as {\"solution\": {\"Entity\": {\"category\": \"value\", ...}, ...}}. "
            f"Expected categories include: {cats}. "
            "Include all entities and do not add extra text."
        )

    return "Output JSON as {\"solution\": {...}}. Do not add extra text."


def _render_ledger(ledger: list[dict[str, Any]]) -> str:
    if not ledger:
        return "(empty ledger)"
    return json.dumps({"constraints": ledger}, ensure_ascii=True, indent=2)


def _problem_setup(domain: str, entities: list[str], context: dict[str, Any] | None = None) -> str:
    context = context or {}
    lines = [
        f"Domain: {domain}",
        f"Entity count: {len(entities)}",
        f"Entities: {', '.join(entities)}",
    ]

    if domain == "seating":
        table_shape = context.get("table_shape", "round")
        num_entities = context.get("num_entities", len(entities))
        lines.append(f"Table shape: {table_shape}")
        lines.append(f"Seat positions: 1 through {num_entities}")
    elif domain == "scheduling":
        num_slots = context.get("num_slots", max(6, len(entities) + 1))
        max_duration = context.get("max_duration", 3)
        lines.append(f"Time slots available: 1 through {num_slots}")
        lines.append(f"Maximum duration per event: {max_duration}")
    elif domain == "logic_grid":
        categories = context.get("categories", {})
        if isinstance(categories, dict) and categories:
            lines.append("Categories:")
            for cat, values in categories.items():
                if isinstance(values, list):
                    lines.append(f"- {cat}: {', '.join(str(v) for v in values)}")

    return "\n".join(lines)


def build_turn_messages(
    method: str,
    conversation_history: list[dict[str, str]],
    new_user_message: str,
    domain: str,
    entities: list[str],
    context: dict[str, Any] | None = None,
    ledger: list[dict[str, Any]] | None = None,
    repair_signal: str | None = None,
) -> list[dict[str, str]]:
    """Build chat-completions messages for the main turn response."""
    if method not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported method: {method}")

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPTS[method]}]
    messages.extend(conversation_history)

    schema_hint = _domain_schema_hint(domain, entities, context)
    first_turn = len(conversation_history) == 0
    user_chunks: list[str] = []
    if first_turn:
        user_chunks.append("Problem setup:\n" + _problem_setup(domain, entities, context))
    else:
        user_chunks.append(f"Domain: {domain}")
        user_chunks.append(f"Entities: {', '.join(entities)}")

    if method in {"ledger_only", "mus_repair"}:
        user_chunks.append("Current ledger:\n" + _render_ledger(ledger or []))

    user_chunks.append(f"New constraints from user:\n{new_user_message}")

    if repair_signal:
        user_chunks.append("Repair signal:\n" + repair_signal)

    user_chunks.append(schema_hint)

    messages.append({"role": "user", "content": "\n\n".join(user_chunks)})
    return messages


def build_answer_retry_messages(
    prior_assistant_response: str,
    domain: str,
    entities: list[str],
    context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    schema_hint = _domain_schema_hint(domain, entities, context)
    context = context or {}
    entity_list = ", ".join(entities)
    extra = ""
    if domain == "logic_grid":
        cats = list(context.get("categories", {}).keys())
        if cats:
            extra = f"Required categories per entity: {', '.join(cats)}.\n"
    return [
        {
            "role": "system",
            "content": (
                "You are a strict JSON formatter. "
                "Output one valid JSON object only. No markdown or prose."
            ),
        },
        {
            "role": "user",
            "content": (
                "Reformat the following response as JSON only. "
                "Do not add commentary.\n"
                f"Include these entities exactly as keys: {entity_list}.\n"
                f"{extra}"
                "If any value is unknown, still emit valid JSON with your best assignment.\n\n"
                f"Original response:\n{prior_assistant_response}\n\n"
                f"{schema_hint}"
            ),
        },
    ]


def build_truncation_retry_messages(
    base_messages: list[dict[str, str]],
    domain: str,
    entities: list[str],
    context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Retry prompt when the main generation was clipped at max_tokens."""
    schema_hint = _domain_schema_hint(domain, entities, context)
    retry_messages = list(base_messages)
    retry_messages.append(
        {
            "role": "user",
            "content": (
                "Your previous answer was truncated by the output token limit. "
                "Retry now and return only one compact JSON object. "
                "No bullets, no prose, no analysis.\n\n"
                f"{schema_hint}"
            ),
        }
    )
    return retry_messages


def build_constraint_extraction_messages(
    assistant_response: str,
    domain: str,
    turn_number: int,
    user_message: str | None = None,
    entities: list[str] | None = None,
    context: dict[str, Any] | None = None,
    retry: bool = False,
) -> list[dict[str, str]]:
    context = context or {}
    entities = entities or []
    strict = (
        "Return valid JSON only."
        if retry
        else "Return JSON."
    )

    if domain == "seating":
        vocab = (
            "Allowed constraint types and args:\n"
            "- adjacent(A, B)\n"
            "- not_adjacent(A, B)\n"
            "- at_position(A, k)\n"
            "- left_of(A, B)\n"
            "- separated_by(A, B, n)\n"
            "- same_side(A, B)\n"
            "- opposite_side(A, B)\n"
        )
    elif domain == "scheduling":
        vocab = (
            "Allowed constraint types and args:\n"
            "- before(A, B)\n"
            "- at_time(A, t)\n"
            "- not_simultaneous(A, B)\n"
            "- within(A, t1, t2)\n"
            "- duration(A, d)\n"
            "- gap(A, B, n)\n"
        )
    elif domain == "logic_grid":
        cats = ", ".join(context.get("categories", {}).keys()) or "color, pet, profession"
        vocab = (
            "Allowed constraint types and args:\n"
            "- assign(entity, category, value)\n"
            "- not_assign(entity, category, value)\n"
            "- same_as(entity1, entity2, category)\n"
            "- different(entity1, entity2, category)\n"
            "- ordered(entity1, entity2, category)\n"
            f"Allowed categories: {cats}\n"
        )
    else:
        vocab = "Allowed constraint types: use only known domain constraints."

    user_msg_block = f"Latest user message:\n{user_message}\n\n" if user_message else ""
    entities_block = f"Entities: {', '.join(entities)}\n" if entities else ""

    return [
        {
            "role": "system",
            "content": (
                "You extract formal constraints from an assistant answer. "
                "Extract only constraints introduced in the latest user turn. "
                "Do not restate full solution assignments unless they directly encode one allowed constraint."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Domain: {domain}\n"
                f"Source turn: {turn_number}\n"
                f"{entities_block}"
                f"{user_msg_block}"
                f"Assistant response:\n{assistant_response}\n\n"
                f"{vocab}\n"
                "Rules:\n"
                "- Use only the allowed type names exactly as written.\n"
                "- Keep args order exactly as specified.\n"
                "- Return only constraints for this turn (typically 1-3 constraints).\n"
                "- If no constraints are recoverable, return {\"constraints\": []}.\n\n"
                "Output schema:\n"
                "{\n"
                "  \"constraints\": [\n"
                "    {\"id\": \"C001\", \"type\": \"...\", \"args\": [...], \"nl\": \"...\", \"source_turn\": 1}\n"
                "  ]\n"
                "}\n"
                f"{strict}"
            ),
        },
    ]


def format_mus_repair(mus_constraints: list[dict[str, Any]]) -> str:
    # Kept for backward compatibility with prior analysis scripts/artifacts.
    return format_repair_signal(trigger_reasons=[{"code": "unsat_ledger"}], mus_constraints=mus_constraints)


def format_repair_signal(
    trigger_reasons: list[dict[str, Any]] | None = None,
    mus_constraints: list[dict[str, Any]] | None = None,
) -> str:
    trigger_reasons = trigger_reasons or []
    mus_constraints = mus_constraints or []

    if not trigger_reasons and not mus_constraints:
        return "Repair required. Re-check consistency with prior commitments and provide revised JSON."

    lines = ["REPAIR REQUIRED:"]

    if trigger_reasons:
        lines.append("Detected issues:")
        for reason in trigger_reasons:
            code = str(reason.get("code", "unknown"))
            detail = str(reason.get("detail", "")).strip()
            if detail:
                lines.append(f"- {code}: {detail}")
            else:
                lines.append(f"- {code}")

    if not mus_constraints:
        lines.append("No MUS contradiction subset available for this step.")
    else:
        lines.append("MUS contradiction subset:")
        for constraint in mus_constraints:
            cid = constraint.get("id", "<no-id>")
            nl = constraint.get("nl", "<no-text>")
            src = constraint.get("source_turn", "?")
            lines.append(f"- {cid}: \"{nl}\" (from turn {src})")

    lines.append("Revise your answer to resolve all listed issues and return a new JSON solution.")
    return "\n".join(lines)
