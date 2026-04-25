# Example transcripts

Real multi-model traces from the DRIFT benchmark evaluation pipeline. Each file covers one problem across all four evaluated models (`qwen3-8b`, `qwen3-32b`, `gpt-oss-20b`, `gpt-oss-120b`) running the `mus_repair` method. Captured deterministically (temperature 0, seed 42) via the OpenRouter API, so the traces are reproducible with any OpenRouter-compatible endpoint.

## Files

* `transcripts/scheduling_example.jsonl`. Scheduling problem `scheduling_249`, 4 turns, 6 activities.
* `transcripts/logic_grid_example.jsonl`. Logic-grid problem `logic_grid_021`, 5 turns, 4 entities × 3 categories.
* `transcripts/seating_example.jsonl`. Seating problem `seating_062`, 4 turns, 7 people around a round table.

## Line format

Each line is one pipeline event as a JSON object. Events appear in the order they occur.

| `type`           | Key fields                                                                 |
|:-----------------|:---------------------------------------------------------------------------|
| `problem_intro`  | `problem_id`, `domain`, `split`, `entities`, `n_turns`, `models`           |
| `turn_start`     | `turn_number`, `new_constraints`                                           |
| `model_trace`    | `model`, `turn_number`, `response_snippet`, `ledger`, `z3_sat`, `mus`, `answer_correct`, `repair_attempted`, `repair_triggers`, `repair_events` |
| `turn_summary`   | `turn_number`, plus a nested object per model with `answer_correct` / `z3_sat` / `repair_attempted` |

Each turn produces one `turn_start`, four `model_trace` events (one per model), and one `turn_summary`. `response_snippet` is truncated to 600 characters with a trailing " ... [truncated]" marker for readability; the full raw response lives in the source SQLite DB during actual runs.

## Provenance

| Field             | Value                                                                  |
|:------------------|:-----------------------------------------------------------------------|
| Capture date      | 2026-04-15                                                             |
| Endpoint          | `https://openrouter.ai/api/v1`                                         |
| Model slugs       | `qwen/qwen3-8b`, `qwen/qwen3-32b`, `openai/gpt-oss-20b`, `openai/gpt-oss-120b` |
| Decoding          | `temperature=0`, `seed=42`, `top_p=1` (`--deterministic`)              |
| Reasoning (gpt-oss) | `--gpt-oss-reasoning-effort medium`                                  |
| Max tokens        | 4096                                                                    |
| Repair policy     | `expanded` (SAT-aware triggers on)                                      |
| Max repair attempts | 2                                                                     |
| Commit (src/)     | See `git log -1 -- src/` at time of capture                             |

To reproduce a transcript from scratch, install `requirements.txt`, set `OPENROUTER_API_KEY`, then run `python -m src.run_experiment --model <m> --method mus_repair --split test --problems-dir <path with only the target problem JSON> --api-base https://openrouter.ai/api/v1 --deterministic --max-tokens 4096` (add `--gpt-oss-reasoning-effort medium` for the gpt-oss models).

## Matching to code paths

* `model_trace.ledger` corresponds to the output of `extraction.extract_constraints` at that turn.
* `model_trace.z3_sat` is the return of `z3_checker.check_satisfiability` on the cumulative ledger.
* `model_trace.mus` is populated only when the ledger is unsatisfiable and `z3_checker.compute_mus` runs.
* `model_trace.repair_triggers` are the codes from `src/repair_policy.py`; see [`../docs/prompts.md`](../docs/prompts.md) for the seven-code glossary.
