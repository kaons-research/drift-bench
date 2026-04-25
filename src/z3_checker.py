#!/usr/bin/env python3
"""Z3 helpers for CLR experiment domains and MUS computation."""

from __future__ import annotations

from typing import Any

from z3 import (
    Abs,
    And,
    Distinct,
    If,
    Int,
    Not,
    Or,
    Solver,
    sat,
    unsat,
)

SYMMETRIC_CONSTRAINT_TYPES = {
    "adjacent",
    "not_adjacent",
    "same_side",
    "opposite_side",
    "not_simultaneous",
    "different",
}


def _to_int(value: Any, default: int | None = None) -> int | None:
    """Best-effort integer conversion."""
    if value is None:
        return default
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
    return default


def canonicalize_constraint(constraint: dict[str, Any]) -> tuple[Any, ...]:
    """Canonical form for matching extracted constraints to gold constraints."""
    ctype = constraint.get("type")
    args = list(constraint.get("args", []))

    if ctype in SYMMETRIC_CONSTRAINT_TYPES and len(args) >= 2:
        pair = sorted([str(args[0]), str(args[1])])
        args = pair + args[2:]

    if ctype == "same_as" and len(args) == 3:
        pair = sorted([str(args[0]), str(args[1])])
        args = [pair[0], pair[1], str(args[2])]

    if ctype == "assign" and len(args) == 3:
        return (ctype, str(args[0]), str(args[1]), str(args[2]))

    if ctype == "not_assign" and len(args) == 3:
        return (ctype, str(args[0]), str(args[1]), str(args[2]))

    return (ctype, *[str(a) for a in args])


def _circular_distance(a, b, n: int):
    diff = Abs(a - b)
    return If(diff <= (n - diff), diff, n - diff)


def _seating_context(entities: list[str], context: dict[str, Any] | None) -> dict[str, Any]:
    context = context or {}
    n = _to_int(context.get("num_entities"), len(entities)) or len(entities)
    table_shape = context.get("table_shape", "round")
    return {
        "num_entities": n,
        "table_shape": table_shape,
    }


def _scheduling_context(entities: list[str], context: dict[str, Any] | None) -> dict[str, Any]:
    context = context or {}
    num_slots = _to_int(context.get("num_slots"), max(6, len(entities) + 1)) or max(6, len(entities) + 1)
    max_duration = _to_int(context.get("max_duration"), 3) or 3
    return {
        "num_slots": num_slots,
        "max_duration": max_duration,
    }


def _logic_context(entities: list[str], context: dict[str, Any] | None) -> dict[str, Any]:
    context = context or {}
    categories = context.get("categories")
    if not categories:
        categories = {
            "color": ["Red", "Blue", "Green", "Yellow"],
            "pet": ["Cat", "Dog", "Bird", "Fish"],
            "profession": ["Doctor", "Artist", "Teacher", "Chef"],
        }
    for _, values in categories.items():
        if len(values) != len(entities):
            raise ValueError("Logic-grid category values must match number of entities.")
    return {"categories": categories}


def _add_seating_constraints(
    solver: Solver,
    vars_pos: dict[str, Any],
    constraints: list[dict[str, Any]],
    context: dict[str, Any],
):
    n = context["num_entities"]
    table_shape = context["table_shape"]

    def get_pos(name: Any):
        key = str(name)
        if key not in vars_pos:
            solver.add(False)
            return None
        return vars_pos[key]

    for c in constraints:
        ctype = c.get("type")
        args = c.get("args", [])
        if not isinstance(args, list):
            solver.add(False)
            continue

        if ctype in {"adjacent", "not_adjacent", "left_of", "same_side", "opposite_side"} and len(args) < 2:
            solver.add(False)
            continue

        if ctype == "adjacent":
            a = get_pos(args[0])
            b = get_pos(args[1])
            if a is None or b is None:
                continue
            d = a - b
            solver.add(Or(d == 1, d == -1, d == n - 1, d == -(n - 1)))

        elif ctype == "not_adjacent":
            a = get_pos(args[0])
            b = get_pos(args[1])
            if a is None or b is None:
                continue
            d = a - b
            solver.add(Not(Or(d == 1, d == -1, d == n - 1, d == -(n - 1))))

        elif ctype == "at_position":
            if len(args) < 2:
                solver.add(False)
                continue
            a = get_pos(args[0])
            pos = _to_int(args[1])
            if a is None or pos is None:
                solver.add(False)
                continue
            if 1 <= pos <= n:
                solver.add(a == pos - 1)
            elif 0 <= pos < n:
                solver.add(a == pos)
            else:
                solver.add(False)

        elif ctype == "left_of":
            a = get_pos(args[0])
            b = get_pos(args[1])
            if a is None or b is None:
                continue
            d = a - b
            solver.add(Or(d == 1, d == -(n - 1)))

        elif ctype == "separated_by":
            if len(args) < 3:
                solver.add(False)
                continue
            a = get_pos(args[0])
            b = get_pos(args[1])
            n_sep = _to_int(args[2])
            if a is None or b is None or n_sep is None:
                solver.add(False)
                continue
            solver.add(_circular_distance(a, b, n) >= n_sep + 1)

        elif ctype == "same_side":
            if table_shape != "rectangular":
                solver.add(False)
                continue
            if n % 2 != 0:
                solver.add(False)
                continue
            side_len = n // 2
            a = get_pos(args[0])
            b = get_pos(args[1])
            if a is None or b is None:
                continue
            solver.add(
                Or(
                    And(a < side_len, b < side_len),
                    And(a >= side_len, b >= side_len),
                )
            )

        elif ctype == "opposite_side":
            if table_shape != "rectangular":
                solver.add(False)
                continue
            if n % 2 != 0:
                solver.add(False)
                continue
            side_len = n // 2
            a = get_pos(args[0])
            b = get_pos(args[1])
            if a is None or b is None:
                continue
            solver.add(
                Or(
                    And(a < side_len, b >= side_len),
                    And(a >= side_len, b < side_len),
                )
            )

        else:
            solver.add(False)


def _add_scheduling_constraints(
    solver: Solver,
    vars_start: dict[str, Any],
    vars_dur: dict[str, Any],
    constraints: list[dict[str, Any]],
):
    def get_var(name: Any, mapping: dict[str, Any]):
        key = str(name)
        if key not in mapping:
            solver.add(False)
            return None
        return mapping[key]

    for c in constraints:
        ctype = c.get("type")
        args = c.get("args", [])
        if not isinstance(args, list):
            solver.add(False)
            continue

        if ctype == "before":
            if len(args) < 2:
                solver.add(False)
                continue
            a = get_var(args[0], vars_start)
            b = get_var(args[1], vars_start)
            da = get_var(args[0], vars_dur)
            if a is None or b is None or da is None:
                continue
            solver.add(a + da <= b)

        elif ctype == "at_time":
            if len(args) < 2:
                solver.add(False)
                continue
            a = get_var(args[0], vars_start)
            t = _to_int(args[1])
            if a is None or t is None:
                solver.add(False)
                continue
            solver.add(a == t)

        elif ctype == "not_simultaneous":
            if len(args) < 2:
                solver.add(False)
                continue
            a = get_var(args[0], vars_start)
            b = get_var(args[1], vars_start)
            if a is None or b is None:
                continue
            solver.add(a != b)

        elif ctype == "within":
            if len(args) < 3:
                solver.add(False)
                continue
            a = get_var(args[0], vars_start)
            t1 = _to_int(args[1])
            t2 = _to_int(args[2])
            if a is None or t1 is None or t2 is None:
                solver.add(False)
                continue
            lo, hi = sorted([t1, t2])
            solver.add(a >= lo, a <= hi)

        elif ctype == "duration":
            if len(args) < 2:
                solver.add(False)
                continue
            a = get_var(args[0], vars_dur)
            d = _to_int(args[1])
            if a is None or d is None:
                solver.add(False)
                continue
            solver.add(a == d)

        elif ctype == "gap":
            if len(args) < 3:
                solver.add(False)
                continue
            a_start = get_var(args[0], vars_start)
            b_start = get_var(args[1], vars_start)
            a_dur = get_var(args[0], vars_dur)
            b_dur = get_var(args[1], vars_dur)
            n_gap = _to_int(args[2])
            if None in (a_start, b_start, a_dur, b_dur) or n_gap is None:
                solver.add(False)
                continue
            solver.add(
                Or(
                    a_start + a_dur + n_gap <= b_start,
                    b_start + b_dur + n_gap <= a_start,
                )
            )

        else:
            solver.add(False)


def _add_logic_constraints(
    solver: Solver,
    vars_logic: dict[tuple[str, str], Any],
    context: dict[str, Any],
    constraints: list[dict[str, Any]],
):
    categories: dict[str, list[str]] = context["categories"]
    value_to_idx: dict[str, dict[str, int]] = {
        cat: {str(v): i for i, v in enumerate(values)} for cat, values in categories.items()
    }

    def get_var(entity: Any, category: Any):
        e_key = str(entity)
        c_key = str(category)
        key = (e_key, c_key)
        if key not in vars_logic:
            solver.add(False)
            return None
        return vars_logic[key]

    for c in constraints:
        ctype = c.get("type")
        args = c.get("args", [])
        if not isinstance(args, list):
            solver.add(False)
            continue

        if ctype == "assign":
            if len(args) < 3:
                solver.add(False)
                continue
            var = get_var(args[0], args[1])
            cat = str(args[1])
            val = str(args[2])
            if var is None or cat not in value_to_idx or val not in value_to_idx[cat]:
                solver.add(False)
                continue
            solver.add(var == value_to_idx[cat][val])

        elif ctype == "not_assign":
            if len(args) < 3:
                solver.add(False)
                continue
            var = get_var(args[0], args[1])
            cat = str(args[1])
            val = str(args[2])
            if var is None or cat not in value_to_idx or val not in value_to_idx[cat]:
                solver.add(False)
                continue
            solver.add(var != value_to_idx[cat][val])

        elif ctype == "same_as":
            if len(args) < 3:
                solver.add(False)
                continue
            a = get_var(args[0], args[2])
            b = get_var(args[1], args[2])
            if a is None or b is None:
                continue
            solver.add(a == b)

        elif ctype == "different":
            if len(args) < 3:
                solver.add(False)
                continue
            a = get_var(args[0], args[2])
            b = get_var(args[1], args[2])
            if a is None or b is None:
                continue
            solver.add(a != b)

        elif ctype == "ordered":
            if len(args) < 3:
                solver.add(False)
                continue
            a = get_var(args[0], args[2])
            b = get_var(args[1], args[2])
            if a is None or b is None:
                continue
            solver.add(a < b)

        else:
            solver.add(False)


def _extract_solution(model, domain: str, aux: dict[str, Any]):
    if domain == "seating":
        result = {}
        for entity, var in aux["vars_pos"].items():
            result[entity] = model.eval(var, model_completion=True).as_long() + 1
        return result

    if domain == "scheduling":
        result = {}
        for entity, var in aux["vars_start"].items():
            start = model.eval(var, model_completion=True).as_long()
            dur = model.eval(aux["vars_dur"][entity], model_completion=True).as_long()
            result[entity] = {"start": start, "duration": dur}
        return result

    if domain == "logic_grid":
        categories = aux["categories"]
        result: dict[str, dict[str, str]] = {}
        for (entity, cat), var in aux["vars_logic"].items():
            if entity not in result:
                result[entity] = {}
            idx = model.eval(var, model_completion=True).as_long()
            values = categories[cat]
            result[entity][cat] = str(values[idx])
        return result

    raise ValueError(f"Unsupported domain: {domain}")


def _checked_solver_result(solver: Solver):
    """Return sat/unsat or raise when solver returns unknown (e.g., timeout)."""
    result = solver.check()
    if result == sat or result == unsat:
        return result
    reason = ""
    try:
        reason = solver.reason_unknown()
    except Exception:
        reason = "unknown"
    raise TimeoutError(f"z3_unknown:{reason or 'unknown'}")


def build_domain_solver(
    domain: str,
    entities: list[str],
    constraints: list[dict[str, Any]],
    context: dict[str, Any] | None = None,
    extra_assignment: dict[str, Any] | None = None,
):
    """Build a solver for one domain with constraints and optional answer assignment."""
    solver = Solver()
    solver.set("timeout", 30000)
    domain = str(domain)

    if domain == "seating":
        ctx = _seating_context(entities, context)
        n = ctx["num_entities"]
        vars_pos = {e: Int(f"seat_{e}") for e in entities}
        for e in entities:
            solver.add(vars_pos[e] >= 0, vars_pos[e] < n)
        solver.add(Distinct(*[vars_pos[e] for e in entities]))
        _add_seating_constraints(solver, vars_pos, constraints, ctx)

        if extra_assignment:
            for entity, raw_pos in extra_assignment.items():
                if entity not in vars_pos:
                    solver.add(False)
                    continue
                pos = _to_int(raw_pos)
                if pos is None:
                    solver.add(False)
                    continue
                if 1 <= pos <= n:
                    solver.add(vars_pos[entity] == pos - 1)
                elif 0 <= pos < n:
                    solver.add(vars_pos[entity] == pos)
                else:
                    solver.add(False)

        aux = {
            "vars_pos": vars_pos,
            "context": ctx,
        }
        return solver, aux

    if domain == "scheduling":
        ctx = _scheduling_context(entities, context)
        num_slots = ctx["num_slots"]
        max_dur = ctx["max_duration"]

        vars_start = {e: Int(f"start_{e}") for e in entities}
        vars_dur = {e: Int(f"dur_{e}") for e in entities}

        for e in entities:
            solver.add(vars_start[e] >= 1, vars_start[e] <= num_slots)
            solver.add(vars_dur[e] >= 1, vars_dur[e] <= max_dur)
            solver.add(vars_start[e] + vars_dur[e] - 1 <= num_slots)

        _add_scheduling_constraints(solver, vars_start, vars_dur, constraints)

        if extra_assignment:
            for entity, payload in extra_assignment.items():
                if entity not in vars_start:
                    solver.add(False)
                    continue

                if isinstance(payload, dict):
                    start = _to_int(payload.get("start"))
                    dur = _to_int(payload.get("duration"))
                    if start is not None:
                        solver.add(vars_start[entity] == start)
                    if dur is not None:
                        solver.add(vars_dur[entity] == dur)
                    if start is None and dur is None:
                        solver.add(False)
                else:
                    start = _to_int(payload)
                    if start is None:
                        solver.add(False)
                    else:
                        solver.add(vars_start[entity] == start)

        aux = {
            "vars_start": vars_start,
            "vars_dur": vars_dur,
            "context": ctx,
        }
        return solver, aux

    if domain == "logic_grid":
        ctx = _logic_context(entities, context)
        categories: dict[str, list[str]] = ctx["categories"]

        vars_logic: dict[tuple[str, str], Any] = {}
        for entity in entities:
            for cat, values in categories.items():
                var = Int(f"logic_{entity}_{cat}")
                solver.add(var >= 0, var < len(values))
                vars_logic[(entity, cat)] = var

        for cat, _ in categories.items():
            solver.add(Distinct(*[vars_logic[(entity, cat)] for entity in entities]))

        _add_logic_constraints(solver, vars_logic, ctx, constraints)

        if extra_assignment:
            value_to_idx: dict[str, dict[str, int]] = {
                cat: {str(v): i for i, v in enumerate(values)} for cat, values in categories.items()
            }
            for entity, cat_map in extra_assignment.items():
                if entity not in entities or not isinstance(cat_map, dict):
                    solver.add(False)
                    continue
                for cat, value in cat_map.items():
                    key = (entity, str(cat))
                    if key not in vars_logic:
                        solver.add(False)
                        continue
                    str_val = str(value)
                    if str(cat) not in value_to_idx or str_val not in value_to_idx[str(cat)]:
                        solver.add(False)
                        continue
                    solver.add(vars_logic[key] == value_to_idx[str(cat)][str_val])

        aux = {
            "vars_logic": vars_logic,
            "categories": categories,
            "context": ctx,
        }
        return solver, aux

    raise ValueError(f"Unsupported domain: {domain}")


def check_satisfiability(
    constraints: list[dict[str, Any]],
    domain: str,
    entities: list[str],
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return SAT result and one model solution if SAT."""
    solver, aux = build_domain_solver(domain, entities, constraints, context=context)
    result = _checked_solver_result(solver)
    if result == sat:
        model = solver.model()
        return {
            "is_sat": True,
            "solution": _extract_solution(model, domain, aux),
        }
    return {"is_sat": False, "solution": None}


def verify_with_z3(
    answer: dict[str, Any],
    cumulative_constraints: list[dict[str, Any]],
    domain: str,
    entities: list[str],
    context: dict[str, Any] | None = None,
) -> int:
    """Verify proposed answer by SAT check of constraints + answer assignment."""
    if not isinstance(answer, dict) or not answer:
        return 0
    solver, _ = build_domain_solver(
        domain,
        entities,
        cumulative_constraints,
        context=context,
        extra_assignment=answer,
    )
    return 1 if _checked_solver_result(solver) == sat else 0


def compute_mus(
    constraints: list[dict[str, Any]],
    domain: str,
    entities: list[str],
    context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Compute deletion-based exact MUS for an UNSAT constraint list."""
    if not constraints:
        return []

    def is_unsat(subset: list[dict[str, Any]]) -> bool:
        solver, _ = build_domain_solver(domain, entities, subset, context=context)
        return _checked_solver_result(solver) == unsat

    if not is_unsat(constraints):
        return []

    mus = list(constraints)
    idx = 0
    while idx < len(mus):
        trial = mus[:idx] + mus[idx + 1 :]
        if trial and is_unsat(trial):
            mus = trial
        else:
            idx += 1
    return mus
