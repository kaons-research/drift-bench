# DRIFT problem set

1,020 multi-turn constraint satisfaction problems, Z3-validated satisfiable at every gold turn.

## Splits

| Split | Total | logic_grid | scheduling | seating |
|---|---:|---:|---:|---:|
| `dev`  | 204 | 68  | 68  | 68  |
| `test` | 816 | 272 | 272 | 272 |

Files live as `data/problems/{split}/{domain}_NNN.json`. The `"split"` field inside each JSON is redundant with the directory but preserved for downstream tooling that scans flat.

## Schema

Every problem is a single JSON object with the following fields:

```json
{
  "problem_id": "scheduling_042",
  "domain": "scheduling",
  "split": "test",
  "num_entities": 5,
  "entities": ["Meeting1", "Meeting2", "Meeting3", "Meeting4", "Meeting5"],
  "turns": [
    {
      "turn_number": 1,
      "user_message": "Free-form natural-language turn text introducing new constraints.",
      "new_constraints": [
        {"type": "at_time", "args": ["Meeting1", 9], "nl": "Meeting1 starts at 9:00"}
      ],
      "cumulative_constraints": [
        {"type": "at_time", "args": ["Meeting1", 9], "nl": "Meeting1 starts at 9:00"}
      ],
      "gold_solution": {"Meeting1": {"start": 9, "duration": 60}, ...},
      "is_satisfiable": true
    },
    ...
  ]
}
```

Logic-grid problems add a top-level `"categories"` field (e.g., `{"color": ["red", "blue", ...], "size": ["small", "medium"]}`).

## Constraint shape

Constraints are `{"type": <str>, "args": <list>, "nl": <str>}`. Types are domain-specific:

### Seating (positions are 1-indexed seat numbers)

| Type | Args | Meaning |
|---|---|---|
| `adjacent` | `[A, B]` | Seats A and B are next to each other. |
| `not_adjacent` | `[A, B]` | Seats A and B are not next to each other. |
| `at_position` | `[A, n]` | A is at position n. |
| `left_of` | `[A, B]` | A's position is strictly less than B's. |
| `separated_by` | `[A, B, k]` | &#124;pos(A) − pos(B)&#124; = k. |
| `same_side` | `[A, B]` | Same half of a rectangular table. |
| `opposite_side` | `[A, B]` | Opposite halves. |

### Scheduling (times are integer slots)

| Type | Args | Meaning |
|---|---|---|
| `before` | `[A, B]` | A ends before B starts. |
| `at_time` | `[A, t]` | A starts at time t. |
| `not_simultaneous` | `[A, B]` | A and B do not overlap. |
| `within` | `[A, lo, hi]` | A starts in `[lo, hi]`. |
| `duration` | `[A, d]` | A has duration d. |
| `gap` | `[A, B, g]` | At least g units between A's end and B's start. |

### Logic grid (bitvector assignments over categories)

| Type | Args | Meaning |
|---|---|---|
| `assign` | `[entity, category, value]` | `entity.category == value`. |
| `not_assign` | `[entity, category, value]` | `entity.category != value`. |
| `ordered` | `[A, B, category]` | A's value in that category precedes B's (category defines an ordering). |
| `same_as` | `[A, B, category]` | A and B share the same value for that category. |
| `different` | `[A, B, category]` | A and B differ on that category. |
| `same_category` / `different_category` | `[A, B]` | All categories equal / all differ. |

## Turn depth distribution

Turns per problem are sampled uniformly in `[4, 10]`. Mean turns across all 1,020 problems is ≈ 6.97 (logic_grid 6.89, scheduling 7.06, seating 6.97).

## Regenerating / extending

See [`../../src/generate_problems.py`](../../src/generate_problems.py). It samples candidate constraints, rejects any that make the cumulative set unsatisfiable under Z3, and writes the resulting problem JSON. Run `python -m src.generate_problems --help` for CLI flags.
