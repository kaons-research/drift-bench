#!/usr/bin/env python3
"""Answer and constraint extraction utilities for CLR experiments."""

from __future__ import annotations

import json
import re
from typing import Any, Callable


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("+"):
            value = value[1:]
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
    return None


def _canonical_entity(name: Any, entities: list[str]) -> str | None:
    if name is None:
        return None
    name_str = str(name).strip()
    for entity in entities:
        if entity.lower() == name_str.lower():
            return entity
    return None


def _strip_think_blocks(text: str) -> str:
    """Remove explicit think blocks so JSON recovery sees cleaner text."""
    if not text:
        return text

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if cleaned != text:
        return cleaned.strip()

    # Handle unclosed <think> prefixes from truncated generations.
    lower = text.lower()
    think_idx = lower.find("<think>")
    if think_idx != -1:
        brace_idx = text.find("{")
        if brace_idx != -1 and brace_idx > think_idx:
            return text[brace_idx:].strip()
    return text


def _try_json(candidate: str) -> dict[str, Any] | list[Any] | None:
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, (dict, list)):
            return parsed
    except Exception:
        return None
    return None


def extract_json_block(text: str) -> dict[str, Any] | list[Any] | None:
    """Extract first valid JSON object/array from text."""
    if not text:
        return None

    text = _strip_think_blocks(text)
    stripped = text.strip()

    parsed = _try_json(stripped)
    if parsed is not None:
        return parsed

    fenced = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        parsed = _try_json(block)
        if parsed is not None:
            return parsed

    starts = [i for i, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        stack: list[str] = []
        for end in range(start, len(text)):
            ch = text[end]
            if ch in "[{":
                stack.append(ch)
            elif ch in "]}":
                if not stack:
                    break
                open_ch = stack.pop()
                if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                    break
                if not stack:
                    candidate = text[start : end + 1]
                    parsed = _try_json(candidate)
                    if parsed is not None:
                        return parsed

        # Truncation repair: close still-open braces/brackets and parse again.
        tail = text[start:].strip()
        if not tail:
            continue
        open_curly = tail.count("{")
        close_curly = tail.count("}")
        open_square = tail.count("[")
        close_square = tail.count("]")
        if open_curly >= close_curly and open_square >= close_square:
            repaired = tail + ("]" * (open_square - close_square)) + ("}" * (open_curly - close_curly))
            parsed = _try_json(repaired)
            if parsed is not None:
                return parsed
    return None


def _normalize_seating_answer(payload: Any, entities: list[str]) -> dict[str, Any] | None:
    if isinstance(payload, dict) and "solution" in payload:
        payload = payload["solution"]

    if not isinstance(payload, dict):
        return None

    answer: dict[str, int] = {}
    for key, value in payload.items():
        entity = _canonical_entity(key, entities)
        if entity is None:
            continue
        pos = _coerce_int(value)
        if pos is not None:
            answer[entity] = pos

    return answer or None


def _normalize_scheduling_answer(payload: Any, entities: list[str]) -> dict[str, Any] | None:
    if isinstance(payload, dict) and "solution" in payload:
        payload = payload["solution"]

    if not isinstance(payload, dict):
        return None

    answer: dict[str, Any] = {}
    for key, value in payload.items():
        entity = _canonical_entity(key, entities)
        if entity is None:
            continue

        if isinstance(value, dict):
            entry: dict[str, int] = {}
            start = _coerce_int(value.get("start"))
            dur = _coerce_int(value.get("duration"))
            if start is not None:
                entry["start"] = start
            if dur is not None:
                entry["duration"] = dur
            if entry:
                answer[entity] = entry
        else:
            start = _coerce_int(value)
            if start is not None:
                answer[entity] = start

    return answer or None


def _normalize_logic_answer(
    payload: Any,
    entities: list[str],
    context: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if isinstance(payload, dict) and "solution" in payload:
        payload = payload["solution"]

    categories = list((context or {}).get("categories", {}).keys()) or ["color", "pet", "profession"]
    category_set = set(categories)

    answer: dict[str, dict[str, str]] = {}

    # Shape A: {"Entity": {"category": "value"}}
    if isinstance(payload, dict):
        for key, value in payload.items():
            entity = _canonical_entity(key, entities)
            if entity is None or not isinstance(value, dict):
                continue
            row: dict[str, str] = {}
            for cat, cat_value in value.items():
                cat_name = str(cat)
                if cat_name not in category_set:
                    continue
                row[cat_name] = str(cat_value)
            if row:
                answer[entity] = row
        if answer:
            return answer

    # Shape B: {"category": {"Entity": "value"}}
    if isinstance(payload, dict):
        by_category: dict[str, dict[str, str]] = {}
        for cat, mapping in payload.items():
            cat_name = str(cat)
            if cat_name not in category_set or not isinstance(mapping, dict):
                continue
            for ent, value in mapping.items():
                entity = _canonical_entity(ent, entities)
                if entity is None:
                    continue
                by_category.setdefault(entity, {})[cat_name] = str(value)
        if by_category:
            return by_category

    # Shape C: [{"entity": "...", "color": "...", ...}, ...]
    if isinstance(payload, list):
        list_answer: dict[str, dict[str, str]] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            entity = _canonical_entity(row.get("entity") or row.get("name"), entities)
            if entity is None:
                continue
            parsed_row: dict[str, str] = {}
            for cat in categories:
                if cat in row:
                    parsed_row[cat] = str(row[cat])
            if parsed_row:
                list_answer[entity] = parsed_row
        if list_answer:
            return list_answer

    return None


def _regex_fallback_table(text: str, entities: list[str]) -> dict[str, int] | None:
    """Simple line-based fallback: Entity: number."""
    answer: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"\s*([A-Za-z0-9_\- ]+)\s*[:=\-]\s*(\d+)\s*$", line)
        if not m:
            continue
        entity = _canonical_entity(m.group(1), entities)
        if entity:
            answer[entity] = int(m.group(2))

    if answer:
        return answer

    # Additional seat wording fallback: "Alice ... position 3".
    for entity in entities:
        m = re.search(
            rf"\b{re.escape(entity)}\b[^\n]{{0,60}}(?:position|seat)\s*(?:is|=|:)?\s*(\d+)",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            answer[entity] = int(m.group(1))
    return answer or None


def _regex_fallback_scheduling(text: str, entities: list[str]) -> dict[str, Any] | None:
    answer: dict[str, Any] = {}

    for entity in entities:
        # JSON-ish: "Entity": {"start": 3, "duration": 2}
        block = re.search(
            rf"\"?{re.escape(entity)}\"?\s*[:=]\s*\{{([^{{}}]{{0,220}})\}}",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block:
            inner = block.group(1)
            start_m = re.search(r"\"?start\"?\s*[:=]\s*(\d+)", inner, flags=re.IGNORECASE)
            dur_m = re.search(r"\"?duration\"?\s*[:=]\s*(\d+)", inner, flags=re.IGNORECASE)
            entry: dict[str, int] = {}
            if start_m:
                entry["start"] = int(start_m.group(1))
            if dur_m:
                entry["duration"] = int(dur_m.group(1))
            if entry:
                answer[entity] = entry
                continue

        # Text-ish: "Entity start 3 duration 2"
        m = re.search(
            rf"\b{re.escape(entity)}\b[^\n]{{0,120}}start\s*(?:is|=|:)?\s*(\d+)(?:[^\n]{{0,80}}duration\s*(?:is|=|:)?\s*(\d+))?",
            text,
            flags=re.IGNORECASE,
        )
        if m:
            entry = {"start": int(m.group(1))}
            if m.group(2) is not None:
                entry["duration"] = int(m.group(2))
            answer[entity] = entry

    return answer or None


def _clean_value(value: str) -> str:
    return value.strip().strip(",;{}[]\"' ")


def _regex_fallback_logic(text: str, entities: list[str], context: dict[str, Any] | None) -> dict[str, Any] | None:
    categories = list((context or {}).get("categories", {}).keys()) or ["color", "pet", "profession"]
    answer: dict[str, dict[str, str]] = {}

    for entity in entities:
        row: dict[str, str] = {}

        block_match = re.search(
            rf"\"?{re.escape(entity)}\"?\s*[:=]\s*\{{([^{{}}]{{0,480}})\}}",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if block_match:
            block = block_match.group(1)
            for cat in categories:
                m = re.search(
                    rf"\"?{re.escape(cat)}\"?\s*[:=]\s*\"?([A-Za-z][A-Za-z0-9_\- ]{{0,40}})\"?",
                    block,
                    flags=re.IGNORECASE,
                )
                if m:
                    row[cat] = _clean_value(m.group(1))

        if not row:
            # Line-style fallback: "Entity: color Red, pet Cat, profession Doctor"
            line_match = re.search(
                rf"\b{re.escape(entity)}\b\s*[:\-]\s*([^\n]{{0,260}})",
                text,
                flags=re.IGNORECASE,
            )
            if line_match:
                line = line_match.group(1)
                for cat in categories:
                    m = re.search(
                        rf"\b{re.escape(cat)}\b\s*(?:is|=|:)?\s*([A-Za-z][A-Za-z0-9_\- ]{{0,40}})",
                        line,
                        flags=re.IGNORECASE,
                    )
                    if m:
                        row[cat] = _clean_value(m.group(1))

        if not row:
            # Loose global pattern with proximity constraints.
            for cat in categories:
                m = re.search(
                    rf"\b{re.escape(entity)}\b[^\n]{{0,140}}\b{re.escape(cat)}\b\s*(?:is|=|:)?\s*\"?([A-Za-z][A-Za-z0-9_\- ]{{0,40}})\"?",
                    text,
                    flags=re.IGNORECASE,
                )
                if m:
                    row[cat] = _clean_value(m.group(1))

        if row:
            answer[entity] = row

    return answer or None


def extract_answer(
    response_text: str,
    domain: str,
    entities: list[str],
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not isinstance(response_text, str):
        response_text = "" if response_text is None else str(response_text)

    response_text = _strip_think_blocks(response_text)
    payload = extract_json_block(response_text)
    if payload is not None:
        if domain == "seating":
            normalized = _normalize_seating_answer(payload, entities)
            if normalized is not None:
                return normalized
        if domain == "scheduling":
            normalized = _normalize_scheduling_answer(payload, entities)
            if normalized is not None:
                return normalized
        if domain == "logic_grid":
            normalized = _normalize_logic_answer(payload, entities, context)
            if normalized is not None:
                return normalized

    if domain == "seating":
        return _regex_fallback_table(response_text, entities)

    if domain == "scheduling":
        table = _regex_fallback_table(response_text, entities)
        if table is not None:
            return table
        return _regex_fallback_scheduling(response_text, entities)

    if domain == "logic_grid":
        return _regex_fallback_logic(response_text, entities, context)

    return None


def extract_answer_with_retry(
    response_text: str,
    domain: str,
    entities: list[str],
    context: dict[str, Any] | None,
    call_fn: Callable[..., dict[str, Any]],
) -> tuple[dict[str, Any] | None, int, str | None]:
    """Evaluation-only extraction with one retry prompt.

    Returns: (answer, extraction_null, retry_text)
    extraction_null polarity: 0 = success, 1 = failure.
    """
    if not isinstance(response_text, str):
        response_text = "" if response_text is None else str(response_text)

    answer = extract_answer(response_text, domain, entities, context)
    if answer is not None:
        # extraction_null=0 means answer extraction succeeded on first attempt.
        return answer, 0, None

    from prompts import build_answer_retry_messages

    retry_messages = build_answer_retry_messages(response_text, domain, entities, context)
    retry_raw = call_fn(messages=retry_messages, temperature=0.0, max_tokens=1024)
    retry_content = retry_raw.get("content", "")
    if str(retry_raw.get("finish_reason") or "").strip() == "length":
        # Treat clipped retry output as extraction failure.
        return None, 1, retry_content
    answer = extract_answer(retry_content, domain, entities, context)
    # extraction_null=0 on successful retry, extraction_null=1 on failure after retry.
    return answer, 0 if answer is not None else 1, retry_content


def _normalize_type_name(raw_type: Any) -> str:
    text = str(raw_type or "").strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _normalize_category_name(raw: Any, categories: list[str]) -> str | None:
    if raw is None:
        return None
    categories = categories or ["color", "pet", "profession"]
    lowered = {c.lower(): c for c in categories}
    alias = {
        "job": "profession",
        "occupation": "profession",
        "animal": "pet",
    }
    key = str(raw).strip().lower()
    key = alias.get(key, key)
    return lowered.get(key)


def _normalize_constraint_shape(
    domain: str,
    ctype: str,
    args: list[Any],
    entities: list[str] | None,
    context: dict[str, Any] | None,
) -> tuple[str, list[Any]] | None:
    entities = entities or []
    context = context or {}

    def ent(value: Any) -> str | None:
        if not entities:
            return str(value)
        return _canonical_entity(value, entities)

    # Seating domain.
    if domain == "seating":
        alias = {
            "next_to": "adjacent",
            "nextto": "adjacent",
            "adjacency": "adjacent",
            "non_adjacent": "not_adjacent",
            "not_next_to": "not_adjacent",
            "notnextto": "not_adjacent",
            "seat": "at_position",
            "position": "at_position",
            "seating_position": "at_position",
            "position_assignment": "at_position",
            "positioning": "at_position",
        }
        t = alias.get(ctype, ctype)
        allowed = {
            "adjacent",
            "not_adjacent",
            "at_position",
            "left_of",
            "separated_by",
            "same_side",
            "opposite_side",
        }
        if t not in allowed:
            return None

        if t in {"adjacent", "not_adjacent", "left_of", "same_side", "opposite_side"}:
            if len(args) < 2:
                return None
            a, b = ent(args[0]), ent(args[1])
            if a is None or b is None:
                return None
            return t, [a, b]

        if t == "at_position":
            if len(args) < 2:
                return None
            a = ent(args[0])
            pos = _coerce_int(args[1])
            if a is None or pos is None:
                return None
            return t, [a, pos]

        if t == "separated_by":
            if len(args) < 3:
                return None
            a, b = ent(args[0]), ent(args[1])
            n = _coerce_int(args[2])
            if a is None or b is None or n is None:
                return None
            return t, [a, b, n]

        return None

    # Scheduling domain.
    if domain == "scheduling":
        alias = {
            "start_time": "at_time",
            "task_start_time": "at_time",
            "event_start_time": "at_time",
            "time_slot": "at_time",
            "task_duration": "duration",
            "event_duration": "duration",
            "non_simultaneous": "not_simultaneous",
            "no_overlap": "not_simultaneous",
            "different_time": "not_simultaneous",
            "between": "within",
        }
        reverse_before = False
        if ctype in {"start_after", "after"}:
            t = "before"
            reverse_before = True
        else:
            t = alias.get(ctype, ctype)

        allowed = {"before", "at_time", "not_simultaneous", "within", "duration", "gap"}
        if t not in allowed:
            return None

        if t in {"before", "not_simultaneous"}:
            if len(args) < 2:
                return None
            a, b = ent(args[0]), ent(args[1])
            if a is None or b is None:
                return None
            if t == "before" and reverse_before:
                return t, [b, a]
            return t, [a, b]

        if t in {"at_time", "duration"}:
            if len(args) < 2:
                return None
            a = ent(args[0])
            n = _coerce_int(args[1])
            if a is None or n is None:
                return None
            return t, [a, n]

        if t == "within":
            if len(args) < 3:
                return None
            a = ent(args[0])
            t1 = _coerce_int(args[1])
            t2 = _coerce_int(args[2])
            if a is None or t1 is None or t2 is None:
                return None
            return t, [a, t1, t2]

        if t == "gap":
            if len(args) < 3:
                return None
            a, b = ent(args[0]), ent(args[1])
            n = _coerce_int(args[2])
            if a is None or b is None or n is None:
                return None
            return t, [a, b, n]

        return None

    # Logic-grid domain.
    if domain == "logic_grid":
        categories = list((context or {}).get("categories", {}).keys()) or ["color", "pet", "profession"]
        assign_alias = {
            "assignment": "assign",
            "attribute_assignment": "assign",
            "value_assignment": "assign",
            "not_assignment": "not_assign",
            "negative_assignment": "not_assign",
        }
        specialized_assign = {
            "pet_assignment": "pet",
            "color_assignment": "color",
            "profession_assignment": "profession",
        }
        relation_alias = {
            "same": "same_as",
            "same_value": "same_as",
            "same_category": "same_as",
            "diff": "different",
            "not_same": "different",
            "order": "ordered",
        }

        if ctype in specialized_assign:
            cat = _normalize_category_name(specialized_assign[ctype], categories)
            if cat is None:
                return None
            if len(args) < 2:
                return None
            a = ent(args[0])
            value = str(args[-1]).strip() if len(args) >= 2 else ""
            if a is None or not value:
                return None
            return "assign", [a, cat, value]

        t = assign_alias.get(ctype, relation_alias.get(ctype, ctype))
        allowed = {"assign", "not_assign", "same_as", "different", "ordered"}
        if t not in allowed:
            return None

        if t in {"assign", "not_assign"}:
            if len(args) < 3:
                return None
            a = ent(args[0])
            cat = _normalize_category_name(args[1], categories)
            value = str(args[2]).strip()
            if a is None or cat is None or not value:
                return None
            return t, [a, cat, value]

        if len(args) < 3:
            return None
        a, b = ent(args[0]), ent(args[1])
        cat = _normalize_category_name(args[2], categories)
        if a is None or b is None or cat is None:
            return None
        return t, [a, b, cat]

    return None


def parse_extracted_constraints(
    payload: Any,
    turn_number: int,
    domain: str,
    entities: list[str] | None = None,
    context: dict[str, Any] | None = None,
    max_constraints: int = 3,
) -> list[dict[str, Any]] | None:
    if not isinstance(payload, dict):
        return None
    raw_constraints = payload.get("constraints")
    if not isinstance(raw_constraints, list):
        return None

    constraints: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    auto_id = 1
    for raw in raw_constraints:
        if not isinstance(raw, dict):
            continue
        ctype = _normalize_type_name(raw.get("type"))
        args = raw.get("args")
        if not ctype or not isinstance(args, list):
            continue

        normalized = _normalize_constraint_shape(domain, ctype, args, entities, context)
        if normalized is None:
            continue
        ctype_norm, args_norm = normalized
        sig = (ctype_norm, tuple(str(a) for a in args_norm))
        if sig in seen:
            continue
        seen.add(sig)

        cid = raw.get("id")
        if not isinstance(cid, str) or not cid.strip():
            cid = f"C{auto_id:03d}"
            auto_id += 1

        nl = raw.get("nl")
        if not isinstance(nl, str) or not nl.strip():
            nl = f"{ctype}({', '.join(str(a) for a in args)})"

        source_turn = raw.get("source_turn", turn_number)
        try:
            source_turn = int(source_turn)
        except Exception:
            source_turn = turn_number

        constraints.append(
            {
                "id": cid,
                "type": ctype_norm,
                "args": args_norm,
                "nl": nl,
                "source_turn": source_turn,
            }
        )
        if max_constraints > 0 and len(constraints) >= max_constraints:
            break

    return constraints if constraints else None


def extract_constraints_with_retry(
    assistant_response: str,
    domain: str,
    turn_number: int,
    user_message: str | None,
    entities: list[str] | None,
    context: dict[str, Any] | None,
    call_fn: Callable[..., dict[str, Any]],
) -> tuple[list[dict[str, Any]] | None, int, str | None]:
    """Runtime extraction used by ledger methods with one retry.

    Returns: (constraints, success, retry_text)
    success polarity: 1 = success, 0 = failure.
    """
    from prompts import build_constraint_extraction_messages

    messages = build_constraint_extraction_messages(
        assistant_response=assistant_response,
        domain=domain,
        turn_number=turn_number,
        user_message=user_message,
        entities=entities,
        context=context,
        retry=False,
    )
    raw = call_fn(messages=messages, temperature=0.0, max_tokens=900)
    content = raw.get("content", "")
    is_truncated = str(raw.get("finish_reason") or "").strip() == "length"
    payload = extract_json_block(content) if not is_truncated else None
    parsed = (
        parse_extracted_constraints(payload, turn_number, domain, entities, context)
        if payload is not None
        else None
    )
    if parsed is not None:
        # success=1 means runtime constraint extraction succeeded on first attempt.
        return parsed, 1, None

    retry_messages = build_constraint_extraction_messages(
        assistant_response=assistant_response,
        domain=domain,
        turn_number=turn_number,
        user_message=user_message,
        entities=entities,
        context=context,
        retry=True,
    )
    retry_raw = call_fn(messages=retry_messages, temperature=0.0, max_tokens=900)
    retry_content = retry_raw.get("content", "")
    retry_truncated = str(retry_raw.get("finish_reason") or "").strip() == "length"
    retry_payload = extract_json_block(retry_content) if not retry_truncated else None
    parsed = (
        parse_extracted_constraints(retry_payload, turn_number, domain, entities, context)
        if retry_payload is not None
        else None
    )
    # success=1 on successful retry, success=0 on failure after retry.
    return parsed, 1 if parsed is not None else 0, retry_content
