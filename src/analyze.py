#!/usr/bin/env python3
"""Analyze CLR experiment outputs from SQLite and emit paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from z3_checker import canonicalize_constraint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CLR results.db")
    parser.add_argument("--db-path", type=Path, default=Path("results.db"))
    parser.add_argument("--problems-dir", type=Path, default=Path("problems"))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--out-dir", type=Path, default=Path("analysis_outputs"))
    return parser.parse_args()


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {str(row[1]) for row in cur.fetchall()}


def load_rows(conn: sqlite3.Connection, split: str, problems_split_map: dict[str, str]) -> list[dict[str, Any]]:
    available_cols = table_columns(conn, "results")
    select_cols = [
        "model",
        "method",
        "domain",
        "problem_id",
        "turn_number",
        "answer_correct",
        "extraction_null",
        "ledger_json",
        "constraint_extraction_success",
        "z3_sat",
        "mus_size",
        "mus_json",
        "repair_attempted",
    ]
    optional_cols = ["repair_trigger", "repair_trigger_count", "repair_reason_json"]
    for col in optional_cols:
        if col in available_cols:
            select_cols.append(col)
        else:
            select_cols.append(f"NULL AS {col}")

    cur = conn.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM results
        ORDER BY model, method, problem_id, turn_number
        """
    )

    rows = []
    columns = [d[0] for d in cur.description]
    for raw in cur.fetchall():
        row = dict(zip(columns, raw))
        p_split = problems_split_map.get(row["problem_id"])
        if split != "all" and p_split != split:
            continue
        rows.append(row)
    return rows


def load_problem_gold(problems_dir: Path) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], dict[str, str]]:
    gold_new_constraints: dict[tuple[str, int], list[dict[str, Any]]] = {}
    split_map: dict[str, str] = {}

    for path in sorted(problems_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            problem = json.load(f)
        pid = problem["problem_id"]
        split_map[pid] = problem.get("split", "test")
        for turn in problem["turns"]:
            key = (pid, int(turn["turn_number"]))
            gold_new_constraints[key] = turn.get("new_constraints", [])
    return gold_new_constraints, split_map


def safe_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def depth_bin(turn_number: int) -> str:
    if 1 <= turn_number <= 3:
        return "1-3"
    if 4 <= turn_number <= 6:
        return "4-6"
    return "7-10"


def parse_ledger_constraints_for_turn(ledger_json: str | None, turn_number: int) -> list[dict[str, Any]]:
    if not ledger_json:
        return []
    try:
        payload = json.loads(ledger_json)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    parsed: list[dict[str, Any]] = []
    for c in payload:
        if not isinstance(c, dict):
            continue
        try:
            src_turn = int(c.get("source_turn", -1))
        except Exception:
            continue
        if src_turn == int(turn_number):
            parsed.append(c)
    return parsed


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()

    if not args.db_path.exists():
        raise FileNotFoundError(f"Missing DB file: {args.db_path}")

    gold_new_constraints, split_map = load_problem_gold(args.problems_dir)

    conn = sqlite3.connect(str(args.db_path))
    try:
        rows = load_rows(conn, args.split, split_map)
    finally:
        conn.close()

    if not rows:
        raise RuntimeError("No matching rows found for selected split.")

    # Table 1: per-problem means, then mean/std over problems.
    per_problem_metrics: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    # key=(model,method,problem_id), values are answer_corrects
    problem_turn_values: dict[tuple[str, str, str], list[int]] = defaultdict(list)

    for row in rows:
        key = (row["model"], row["method"], row["problem_id"])
        problem_turn_values[key].append(int(row["answer_correct"] or 0))

    grouped_problem_stats: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: {"contradiction": [], "accuracy": []})
    for (model, method, _pid), vals in problem_turn_values.items():
        acc = safe_mean([float(v) for v in vals])
        contra = safe_mean([1.0 - float(v) for v in vals])
        grouped_problem_stats[(model, method)]["accuracy"].append(acc)
        grouped_problem_stats[(model, method)]["contradiction"].append(contra)

    table1_rows: list[dict[str, Any]] = []
    for (model, method), stats in sorted(grouped_problem_stats.items()):
        table1_rows.append(
            {
                "model": model,
                "method": method,
                "n_problems": len(stats["accuracy"]),
                "contradiction_rate_mean": round(safe_mean(stats["contradiction"]), 6),
                "contradiction_rate_std": round(safe_std(stats["contradiction"]), 6),
                "solution_accuracy_mean": round(safe_mean(stats["accuracy"]), 6),
                "solution_accuracy_std": round(safe_std(stats["accuracy"]), 6),
            }
        )

    # Table 2 diagnostics.
    grouped_diag: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "extraction_null": 0,
            "rows": 0,
            "constraint_success_sum": 0,
            "constraint_rows": 0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "repair_attempts": 0,
            "repair_success": 0,
            "triggered_rows": 0,
            "trigger_event_count": 0,
            "trigger_code_counts": defaultdict(int),
        }
    )

    for row in rows:
        key = (row["model"], row["method"])
        bucket = grouped_diag[key]
        bucket["rows"] += 1
        bucket["extraction_null"] += int(row["extraction_null"] or 0)

        method = row["method"]
        if method in {"ledger_only", "mus_repair"}:
            bucket["constraint_rows"] += 1
            bucket["constraint_success_sum"] += int(row["constraint_extraction_success"] or 0)

            gold = gold_new_constraints.get((row["problem_id"], int(row["turn_number"])), [])
            gold_set = {canonicalize_constraint(c) for c in gold}

            extracted_constraints = parse_ledger_constraints_for_turn(row.get("ledger_json"), int(row["turn_number"]))
            pred_set = {canonicalize_constraint(c) for c in extracted_constraints}

            tp = len(gold_set & pred_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)

            bucket["tp"] += tp
            bucket["fp"] += fp
            bucket["fn"] += fn

        if method == "mus_repair":
            attempted = int(row["repair_attempted"] or 0)
            if attempted:
                bucket["repair_attempts"] += 1
                if int(row["z3_sat"] or 0) == 1:
                    bucket["repair_success"] += 1

            trigger_count = int(row.get("repair_trigger_count") or 0)
            if trigger_count > 0:
                bucket["triggered_rows"] += 1
                bucket["trigger_event_count"] += trigger_count

            trigger_json = row.get("repair_reason_json")
            if isinstance(trigger_json, str) and trigger_json.strip():
                try:
                    payload = json.loads(trigger_json)
                    if isinstance(payload, list):
                        for event in payload:
                            if not isinstance(event, dict):
                                continue
                            for code in event.get("codes", []):
                                code_text = str(code).strip()
                                if code_text:
                                    bucket["trigger_code_counts"][code_text] += 1
                except Exception:
                    pass
            elif isinstance(row.get("repair_trigger"), str) and row.get("repair_trigger"):
                for code in str(row.get("repair_trigger")).split(","):
                    code_text = code.strip()
                    if code_text:
                        bucket["trigger_code_counts"][code_text] += 1

    table2_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    for (model, method), bucket in sorted(grouped_diag.items()):
        precision = (
            bucket["tp"] / (bucket["tp"] + bucket["fp"])
            if (bucket["tp"] + bucket["fp"]) > 0
            else 0.0
        )
        recall = (
            bucket["tp"] / (bucket["tp"] + bucket["fn"])
            if (bucket["tp"] + bucket["fn"]) > 0
            else 0.0
        )
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        table2_rows.append(
            {
                "model": model,
                "method": method,
                "parse_fail_rate": round(bucket["extraction_null"] / max(1, bucket["rows"]), 6),
                "constraint_parse_fail_rate": (
                    round(1.0 - (bucket["constraint_success_sum"] / max(1, bucket["constraint_rows"])), 6)
                    if method in {"ledger_only", "mus_repair"}
                    else None
                ),
                "extraction_precision": round(precision, 6) if method in {"ledger_only", "mus_repair"} else None,
                "extraction_recall": round(recall, 6) if method in {"ledger_only", "mus_repair"} else None,
                "extraction_f1": round(f1, 6) if method in {"ledger_only", "mus_repair"} else None,
                "repair_success_rate": (
                    round(bucket["repair_success"] / max(1, bucket["repair_attempts"]), 6)
                    if method == "mus_repair"
                    else None
                ),
                "repair_attempts": bucket["repair_attempts"] if method == "mus_repair" else None,
                "repair_trigger_rate": (
                    round(bucket["triggered_rows"] / max(1, bucket["rows"]), 6)
                    if method == "mus_repair"
                    else None
                ),
                "avg_trigger_events_when_triggered": (
                    round(bucket["trigger_event_count"] / max(1, bucket["triggered_rows"]), 6)
                    if method == "mus_repair"
                    else None
                ),
            }
        )
        if method == "mus_repair":
            for code, count in sorted(bucket["trigger_code_counts"].items()):
                trigger_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "trigger_code": code,
                        "count": count,
                    }
                )

    # Turn-depth stats.
    depth_bucket: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["method"], depth_bin(int(row["turn_number"])))
        depth_bucket[key].append(int(row["answer_correct"] or 0))

    depth_rows: list[dict[str, Any]] = []
    for (model, method, d_bin), vals in sorted(depth_bucket.items()):
        depth_rows.append(
            {
                "model": model,
                "method": method,
                "turn_depth_bin": d_bin,
                "n_turns": len(vals),
                "contradiction_rate": round(safe_mean([1.0 - float(v) for v in vals]), 6),
                "solution_accuracy": round(safe_mean([float(v) for v in vals]), 6),
            }
        )

    # Per-domain breakdown.
    domain_bucket: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in rows:
        key = (row["model"], row["method"], row["domain"])
        domain_bucket[key].append(int(row["answer_correct"] or 0))

    domain_rows: list[dict[str, Any]] = []
    for (model, method, domain), vals in sorted(domain_bucket.items()):
        domain_rows.append(
            {
                "model": model,
                "method": method,
                "domain": domain,
                "n_turns": len(vals),
                "contradiction_rate": round(safe_mean([1.0 - float(v) for v in vals]), 6),
                "solution_accuracy": round(safe_mean([float(v) for v in vals]), 6),
            }
        )

    out_dir = args.out_dir
    write_csv(
        out_dir / "table1_main_results.csv",
        table1_rows,
        [
            "model",
            "method",
            "n_problems",
            "contradiction_rate_mean",
            "contradiction_rate_std",
            "solution_accuracy_mean",
            "solution_accuracy_std",
        ],
    )
    write_csv(
        out_dir / "table2_diagnostics.csv",
        table2_rows,
        [
            "model",
            "method",
            "parse_fail_rate",
            "constraint_parse_fail_rate",
            "extraction_precision",
            "extraction_recall",
            "extraction_f1",
            "repair_success_rate",
            "repair_attempts",
            "repair_trigger_rate",
            "avg_trigger_events_when_triggered",
        ],
    )
    write_csv(
        out_dir / "trigger_breakdown.csv",
        trigger_rows,
        [
            "model",
            "method",
            "trigger_code",
            "count",
        ],
    )
    write_csv(
        out_dir / "turn_depth_stats.csv",
        depth_rows,
        [
            "model",
            "method",
            "turn_depth_bin",
            "n_turns",
            "contradiction_rate",
            "solution_accuracy",
        ],
    )
    write_csv(
        out_dir / "domain_breakdown.csv",
        domain_rows,
        [
            "model",
            "method",
            "domain",
            "n_turns",
            "contradiction_rate",
            "solution_accuracy",
        ],
    )

    print(f"Rows analyzed: {len(rows)}")
    print(f"Wrote: {out_dir / 'table1_main_results.csv'}")
    print(f"Wrote: {out_dir / 'table2_diagnostics.csv'}")
    print(f"Wrote: {out_dir / 'trigger_breakdown.csv'}")
    print(f"Wrote: {out_dir / 'turn_depth_stats.csv'}")
    print(f"Wrote: {out_dir / 'domain_breakdown.csv'}")


if __name__ == "__main__":
    main()
