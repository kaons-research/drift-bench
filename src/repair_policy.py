#!/usr/bin/env python3
"""Repair trigger policy helpers for MUS/state-consistency experiments."""

from __future__ import annotations

from typing import Any

UNSAT_LEDGER = "unsat_ledger"
CONSTRAINT_EXTRACTION_FAILED = "constraint_extraction_failed"
NO_NEW_CONSTRAINTS = "no_new_constraints"
LOW_COMMITMENT = "low_commitment"
ANSWER_PARSE_FAILED = "answer_parse_failed"
INCOMPLETE_ASSIGNMENT = "incomplete_assignment"
ANSWER_LEDGER_CONFLICT = "answer_ledger_conflict"

ALL_TRIGGER_CODES = {
    UNSAT_LEDGER,
    CONSTRAINT_EXTRACTION_FAILED,
    NO_NEW_CONSTRAINTS,
    LOW_COMMITMENT,
    ANSWER_PARSE_FAILED,
    INCOMPLETE_ASSIGNMENT,
    ANSWER_LEDGER_CONFLICT,
}


def policy_uses_sat_triggers(policy: str) -> bool:
    return str(policy) == "expanded"


def ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def unique_trigger_codes(reasons: list[dict[str, Any]]) -> list[str]:
    return ordered_unique([str(r.get("code", "")).strip() for r in reasons])


def _missing_assignments(
    answer: dict[str, Any] | None,
    domain: str,
    entities: list[str],
    context: dict[str, Any] | None = None,
) -> list[str]:
    context = context or {}
    if not isinstance(answer, dict):
        return list(entities)

    missing: list[str] = []
    if domain == "seating":
        for entity in entities:
            if entity not in answer:
                missing.append(entity)
        return missing

    if domain == "scheduling":
        for entity in entities:
            if entity not in answer:
                missing.append(entity)
                continue
            payload = answer.get(entity)
            if isinstance(payload, dict):
                if "start" not in payload:
                    missing.append(f"{entity}.start")
            elif isinstance(payload, int):
                # int payload implies start time only, which is acceptable.
                continue
            else:
                missing.append(f"{entity}.start")
        return missing

    if domain == "logic_grid":
        categories = list(context.get("categories", {}).keys())
        for entity in entities:
            payload = answer.get(entity)
            if not isinstance(payload, dict):
                missing.append(entity)
                continue
            for cat in categories:
                if cat not in payload:
                    missing.append(f"{entity}.{cat}")
        return missing

    for entity in entities:
        if entity not in answer:
            missing.append(entity)
    return missing


def build_repair_reasons(
    *,
    extracted_constraints: list[dict[str, Any]] | None,
    domain: str,
    entities: list[str],
    context: dict[str, Any] | None,
    is_sat: bool | None,
    quick_answer: dict[str, Any] | None,
    answer_ledger_conflict: bool | None,
    min_new_constraints_for_commitment: int,
    use_sat_triggers: bool,
) -> list[dict[str, Any]]:
    reasons: list[dict[str, Any]] = []

    if extracted_constraints is None:
        reasons.append(
            {
                "code": CONSTRAINT_EXTRACTION_FAILED,
                "detail": "Could not extract new-turn constraints from the response.",
            }
        )
    else:
        extracted_count = len(extracted_constraints)
        if extracted_count == 0:
            reasons.append(
                {
                    "code": NO_NEW_CONSTRAINTS,
                    "detail": "No new constraints were extracted for this turn.",
                }
            )
        elif extracted_count < max(1, int(min_new_constraints_for_commitment)):
            reasons.append(
                {
                    "code": LOW_COMMITMENT,
                    "detail": (
                        "Extracted constraint count below commitment threshold "
                        f"({extracted_count} < {max(1, int(min_new_constraints_for_commitment))})."
                    ),
                }
            )

    if is_sat is False:
        reasons.append(
            {
                "code": UNSAT_LEDGER,
                "detail": "Candidate ledger is unsatisfiable.",
            }
        )

    if use_sat_triggers and is_sat is True:
        if quick_answer is None:
            reasons.append(
                {
                    "code": ANSWER_PARSE_FAILED,
                    "detail": "Could not parse an answer assignment from the response.",
                }
            )
        else:
            missing = _missing_assignments(
                answer=quick_answer,
                domain=domain,
                entities=entities,
                context=context,
            )
            if missing:
                reasons.append(
                    {
                        "code": INCOMPLETE_ASSIGNMENT,
                        "detail": (
                            "Answer assignment is incomplete; missing components: "
                            + ", ".join(missing[:8])
                            + (" ..." if len(missing) > 8 else "")
                        ),
                        "missing_count": len(missing),
                    }
                )

        if answer_ledger_conflict is True:
            reasons.append(
                {
                    "code": ANSWER_LEDGER_CONFLICT,
                    "detail": "Parsed answer conflicts with the current candidate ledger.",
                }
            )

    return reasons
