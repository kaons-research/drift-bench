# Prompt templates

This reference dumps the DRIFT prompt templates so reviewers can read them without running the code. The canonical source is [`src/prompts.py`](../src/prompts.py); when the two disagree, the source wins.

All four methods (Direct, CoT, Ledger, MUS-Repair) share the same problem setup, domain schema hint, and constraint-extraction pass. They differ only in (a) their system prompt, (b) whether a ledger is shown to the generator, and (c) whether Z3 repair signals are injected between turns.

## System prompts

```python
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
```

## Domain schema hints (`_domain_schema_hint`)

Tells the generator exactly what JSON shape to return.

* **Seating:** `{"solution": {"Entity": position_int, ...}}` with 1-indexed seat numbers.
* **Scheduling:** `{"solution": {"Event": {"start": int, "duration": int}, ...}}`. Duration is optional.
* **Logic grid:** `{"solution": {"Entity": {"category": "value", ...}, ...}}` with categories enumerated from the problem.

## Builders

All builders return a `list[dict]` of OpenAI chat-messages ready to hand to any OpenAI-compatible endpoint.

### `build_turn_messages(method, domain, entities, context, turn, ledger)`

Primary per-turn prompt. Constructs: the system prompt for the chosen method, a problem-setup block (entity list plus domain schema hint), the ledger (if the method uses one), and the new user constraints for this turn.

### `build_answer_retry_messages(...)`

Sent when the generator's answer fails to parse or is incomplete. Re-prompts with the original context plus a short nudge describing the parse or completeness failure.

### `build_truncation_retry_messages(...)`

Sent when the response was truncated by max-tokens. Re-prompts with a reduced token budget and a hint to omit scratchpad reasoning, then re-extracts.

### `build_constraint_extraction_messages(...)`

Two-stage extraction. After the generator answers, a separate call extracts the explicit constraints it acknowledged, normalized to the canonical `{type, args, nl}` schema. These extracted constraints feed the ledger and the Z3 check.

### `format_mus_repair(mus_constraints)`

Renders a minimal unsatisfiable subset as a human-readable string for injection into the next turn: "The constraints Ci and Cj are jointly unsatisfiable. Revise your answer so it no longer commits to both."

### `format_repair_signal(trigger_codes, ...)`

Composes a structured repair signal from the seven trigger codes defined in [`src/repair_policy.py`](../src/repair_policy.py):

| Code | Meaning |
|:-----|:--------|
| `answer_ledger_conflict` | SAT ledger, but parsed answer violates active constraints (drift). |
| `unsat_ledger` | Ledger itself is unsatisfiable. MUS is computed. |
| `incomplete_assignment` | Parsed answer is missing entities. |
| `answer_parse_failed` | Response did not yield valid schema JSON. |
| `constraint_extraction_failed` | Extractor returned an empty constraint set. |
| `no_new_constraints` | Extractor returned only a subset of prior commitments. |
| `low_commitment` | Extraction below a per-turn constraint cap. |

The signal is injected as a user message on the next turn so the generator sees what went wrong and what to fix.

## Reading order

If you are new to the prompts, open them in this order:

1. `SYSTEM_PROMPTS` above. The four method-specific instructions.
2. `_problem_setup` in [`src/prompts.py`](../src/prompts.py). What every turn starts with.
3. `build_turn_messages`. How the pieces get assembled.
4. `build_constraint_extraction_messages`. The second, extraction-only pass.
5. `format_repair_signal` and `format_mus_repair`. The MUS-Repair feedback.

That covers the full surface the benchmark exposes to the generator.
