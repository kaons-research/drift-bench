#!/usr/bin/env python3
"""Generate synthetic multi-turn constraint problems with Z3 SAT guarantees."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from z3_checker import canonicalize_constraint, check_satisfiability

SEATING_NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "Diana",
    "Eve",
    "Frank",
    "Grace",
    "Henry",
    "Ivy",
    "Jack",
    "Karen",
    "Liam",
    "Mia",
    "Noah",
    "Olivia",
    "Peter",
    "Quinn",
    "Ruby",
    "Sam",
    "Tina",
]

SCHEDULING_EVENTS = [
    "Design",
    "Review",
    "Deploy",
    "Testing",
    "Planning",
    "Meeting",
    "Docs",
    "Refactor",
    "QA",
    "Sync",
]

LOGIC_ENTITIES_POOL = [
    "Avery",
    "Blake",
    "Casey",
    "Drew",
    "Emery",
    "Finley",
    "Hayden",
    "Jordan",
]

LOGIC_CATEGORIES = {
    "color": ["Red", "Blue", "Green", "Yellow"],
    "pet": ["Cat", "Dog", "Bird", "Fish"],
    "profession": ["Doctor", "Artist", "Teacher", "Chef"],
}


def build_user_message(constraints: list[dict[str, Any]]) -> str:
    parts = []
    for idx, constraint in enumerate(constraints):
        sentence = str(constraint["nl"]).strip().rstrip(".")
        if idx == 0:
            parts.append(f"{sentence}.")
        else:
            parts.append(f"Also, {sentence}.")
    return " ".join(parts)


def pick_distinct_pair(rng: random.Random, items: list[str]) -> tuple[str, str]:
    a, b = rng.sample(items, 2)
    return a, b


def sample_seating_constraint(
    rng: random.Random,
    entities: list[str],
    num_entities: int,
    table_shape: str,
) -> dict[str, Any]:
    kinds = ["adjacent", "not_adjacent", "at_position", "left_of", "separated_by"]
    if table_shape == "rectangular":
        kinds.extend(["same_side", "opposite_side"])
    ctype = rng.choice(kinds)

    if ctype == "adjacent":
        a, b = pick_distinct_pair(rng, entities)
        return {
            "type": ctype,
            "args": [a, b],
            "nl": f"{a} must sit next to {b}",
        }

    if ctype == "not_adjacent":
        a, b = pick_distinct_pair(rng, entities)
        return {
            "type": ctype,
            "args": [a, b],
            "nl": f"{a} cannot sit next to {b}",
        }

    if ctype == "at_position":
        a = rng.choice(entities)
        pos = rng.randint(1, num_entities)
        return {
            "type": ctype,
            "args": [a, pos],
            "nl": f"{a} must sit at position {pos}",
        }

    if ctype == "left_of":
        a, b = pick_distinct_pair(rng, entities)
        return {
            "type": ctype,
            "args": [a, b],
            "nl": f"{a} sits immediately left of {b}",
        }

    if ctype == "separated_by":
        a, b = pick_distinct_pair(rng, entities)
        n_sep = rng.randint(1, max(1, (num_entities // 2) - 1))
        return {
            "type": ctype,
            "args": [a, b, n_sep],
            "nl": f"{a} and {b} must have at least {n_sep} seats between them",
        }

    if ctype == "same_side":
        a, b = pick_distinct_pair(rng, entities)
        return {
            "type": ctype,
            "args": [a, b],
            "nl": f"{a} and {b} must sit on the same side of the table",
        }

    if ctype == "opposite_side":
        a, b = pick_distinct_pair(rng, entities)
        return {
            "type": ctype,
            "args": [a, b],
            "nl": f"{a} and {b} must sit on opposite sides of the table",
        }

    raise ValueError(f"Unsupported seating constraint type: {ctype}")


def sample_scheduling_constraint(
    rng: random.Random,
    entities: list[str],
    num_slots: int,
    max_duration: int,
) -> dict[str, Any]:
    ctype = rng.choice([
        "before",
        "at_time",
        "not_simultaneous",
        "within",
        "duration",
        "gap",
    ])

    if ctype == "before":
        a, b = pick_distinct_pair(rng, entities)
        return {
            "type": ctype,
            "args": [a, b],
            "nl": f"{a} must happen before {b}",
        }

    if ctype == "at_time":
        a = rng.choice(entities)
        t = rng.randint(1, num_slots)
        return {
            "type": ctype,
            "args": [a, t],
            "nl": f"{a} must start at time slot {t}",
        }

    if ctype == "not_simultaneous":
        a, b = pick_distinct_pair(rng, entities)
        return {
            "type": ctype,
            "args": [a, b],
            "nl": f"{a} and {b} cannot be simultaneous",
        }

    if ctype == "within":
        a = rng.choice(entities)
        t1 = rng.randint(1, num_slots)
        t2 = rng.randint(1, num_slots)
        lo, hi = sorted([t1, t2])
        return {
            "type": ctype,
            "args": [a, lo, hi],
            "nl": f"{a} must start between time slots {lo} and {hi}",
        }

    if ctype == "duration":
        a = rng.choice(entities)
        d = rng.randint(1, max_duration)
        return {
            "type": ctype,
            "args": [a, d],
            "nl": f"{a} has duration {d}",
        }

    if ctype == "gap":
        a, b = pick_distinct_pair(rng, entities)
        g = rng.randint(0, 2)
        return {
            "type": ctype,
            "args": [a, b, g],
            "nl": f"There must be at least {g} slots between {a} and {b}",
        }

    raise ValueError(f"Unsupported scheduling constraint type: {ctype}")


def sample_logic_constraint(
    rng: random.Random,
    entities: list[str],
    categories: dict[str, list[str]],
) -> dict[str, Any]:
    ctype = rng.choice(["assign", "not_assign", "same_as", "different", "ordered"])

    cat = rng.choice(list(categories.keys()))
    values = categories[cat]

    if ctype == "assign":
        e = rng.choice(entities)
        v = rng.choice(values)
        return {
            "type": ctype,
            "args": [e, cat, v],
            "nl": f"{e} is assigned {v} in category {cat}",
        }

    if ctype == "not_assign":
        e = rng.choice(entities)
        v = rng.choice(values)
        return {
            "type": ctype,
            "args": [e, cat, v],
            "nl": f"{e} is not assigned {v} in category {cat}",
        }

    if ctype == "same_as":
        a, b = pick_distinct_pair(rng, entities)
        return {
            "type": ctype,
            "args": [a, b, cat],
            "nl": f"{a} has the same {cat} as {b}",
        }

    if ctype == "different":
        a, b = pick_distinct_pair(rng, entities)
        return {
            "type": ctype,
            "args": [a, b, cat],
            "nl": f"{a} has a different {cat} than {b}",
        }

    if ctype == "ordered":
        a, b = pick_distinct_pair(rng, entities)
        return {
            "type": ctype,
            "args": [a, b, cat],
            "nl": f"{a}'s {cat} value comes before {b}'s {cat} value",
        }

    raise ValueError(f"Unsupported logic constraint type: {ctype}")


def sample_constraints_for_domain(
    rng: random.Random,
    domain: str,
    entities: list[str],
    context: dict[str, Any],
    count: int,
) -> list[dict[str, Any]]:
    sampled = []
    for _ in range(count):
        if domain == "seating":
            sampled.append(
                sample_seating_constraint(
                    rng,
                    entities,
                    num_entities=context["num_entities"],
                    table_shape=context.get("table_shape", "round"),
                )
            )
        elif domain == "scheduling":
            sampled.append(
                sample_scheduling_constraint(
                    rng,
                    entities,
                    num_slots=context["num_slots"],
                    max_duration=context["max_duration"],
                )
            )
        elif domain == "logic_grid":
            sampled.append(sample_logic_constraint(rng, entities, context["categories"]))
        else:
            raise ValueError(f"Unsupported domain: {domain}")
    return sampled


def generate_problem(
    rng: random.Random,
    domain: str,
    problem_idx: int,
    max_turns: int = 10,
    min_turns: int = 4,
) -> dict[str, Any]:
    """Generate one SAT-valid multi-turn problem; retries internally if needed."""
    for _ in range(200):
        if domain == "seating":
            num_entities = rng.choice([6, 7, 8])
            entities = rng.sample(SEATING_NAMES, num_entities)
            table_shape = "rectangular" if (num_entities % 2 == 0 and rng.random() < 0.25) else "round"
            context = {
                "num_entities": num_entities,
                "table_shape": table_shape,
            }
        elif domain == "scheduling":
            num_entities = rng.choice([5, 6, 7])
            entities = rng.sample(SCHEDULING_EVENTS, num_entities)
            num_slots = num_entities + rng.randint(2, 4)
            context = {
                "num_slots": num_slots,
                "max_duration": 3,
            }
        elif domain == "logic_grid":
            num_entities = 4
            entities = rng.sample(LOGIC_ENTITIES_POOL, num_entities)
            context = {
                "categories": copy.deepcopy(LOGIC_CATEGORIES),
            }
        else:
            raise ValueError(f"Unsupported domain: {domain}")

        turn_count = rng.randint(min_turns, max_turns)
        cumulative: list[dict[str, Any]] = []
        turns: list[dict[str, Any]] = []
        used_canon = set()
        failed = False

        for turn_number in range(1, turn_count + 1):
            accepted = False
            for _ in range(300):
                num_new = rng.randint(1, 3)
                proposals = sample_constraints_for_domain(rng, domain, entities, context, num_new)

                deduped: list[dict[str, Any]] = []
                local_seen = set()
                for candidate in proposals:
                    ckey = canonicalize_constraint(candidate)
                    if ckey in used_canon or ckey in local_seen:
                        continue
                    local_seen.add(ckey)
                    deduped.append(candidate)

                if not deduped:
                    continue

                trial = cumulative + deduped
                sat_result = check_satisfiability(trial, domain, entities, context=context)
                if sat_result["is_sat"]:
                    cumulative = trial
                    used_canon |= {canonicalize_constraint(c) for c in deduped}
                    turns.append(
                        {
                            "turn_number": turn_number,
                            "user_message": build_user_message(deduped),
                            "new_constraints": copy.deepcopy(deduped),
                            "cumulative_constraints": copy.deepcopy(cumulative),
                            "gold_solution": sat_result["solution"],
                            "is_satisfiable": True,
                        }
                    )
                    accepted = True
                    break

            if not accepted:
                failed = True
                break

        if failed:
            continue

        problem = {
            "problem_id": f"{domain}_{problem_idx:03d}",
            "domain": domain,
            "num_entities": len(entities),
            "entities": entities,
            "turns": turns,
        }
        problem.update(context)
        return problem

    raise RuntimeError(f"Failed to generate SAT problem after retries: {domain}_{problem_idx:03d}")


def assign_splits(
    rng: random.Random,
    problems_by_domain: dict[str, list[dict[str, Any]]],
    dev_ratio: float = 0.2,
):
    for domain, items in problems_by_domain.items():
        indices = list(range(len(items)))
        rng.shuffle(indices)
        dev_count = int(round(len(items) * dev_ratio))
        dev_indices = set(indices[:dev_count])
        for idx, problem in enumerate(items):
            problem["split"] = "dev" if idx in dev_indices else "test"


def write_problems(output_dir: Path, problems_by_domain: dict[str, list[dict[str, Any]]]):
    output_dir.mkdir(parents=True, exist_ok=True)
    for domain, items in problems_by_domain.items():
        for problem in items:
            path = output_dir / f"{problem['problem_id']}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(problem, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLR synthetic SAT problems.")
    parser.add_argument("--output-dir", type=Path, default=Path("problems"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-domain", type=int, default=340)
    parser.add_argument("--min-turns", type=int, default=4)
    parser.add_argument("--max-turns", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    domains = ["seating", "scheduling", "logic_grid"]
    problems_by_domain: dict[str, list[dict[str, Any]]] = {d: [] for d in domains}

    for domain in domains:
        for idx in range(1, args.per_domain + 1):
            problem = generate_problem(
                rng=rng,
                domain=domain,
                problem_idx=idx,
                max_turns=args.max_turns,
                min_turns=args.min_turns,
            )
            problems_by_domain[domain].append(problem)

    assign_splits(rng, problems_by_domain, dev_ratio=0.2)
    write_problems(args.output_dir, problems_by_domain)

    total = sum(len(v) for v in problems_by_domain.values())
    dev_count = sum(1 for arr in problems_by_domain.values() for p in arr if p["split"] == "dev")
    test_count = total - dev_count

    print(f"Generated {total} problems in {args.output_dir}")
    print(f"Dev: {dev_count}, Test: {test_count}")
    for domain in domains:
        arr = problems_by_domain[domain]
        ddev = sum(1 for p in arr if p["split"] == "dev")
        dtest = len(arr) - ddev
        print(f"  {domain}: total={len(arr)} dev={ddev} test={dtest}")


if __name__ == "__main__":
    main()
