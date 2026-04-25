#!/usr/bin/env python3
"""Run CLR multi-turn experiments against a running vLLM OpenAI-compatible server."""

from __future__ import annotations

import argparse
import email.utils
import json
import multiprocessing as mp
import os
import queue
import random
import re
import sqlite3
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from extraction import extract_answer, extract_answer_with_retry, extract_constraints_with_retry
from prompts import (
    build_truncation_retry_messages,
    build_turn_messages,
    format_repair_signal,
)
from repair_policy import (
    UNSAT_LEDGER,
    build_repair_reasons,
    ordered_unique,
    policy_uses_sat_triggers,
    unique_trigger_codes,
)

METHODS = ["direct", "cot", "ledger_only", "mus_repair"]

MODELS = {
    "qwen3-8b": {
        "hf_repo": "Qwen/Qwen3-8B",
        "serve_flags": ["--dtype", "auto", "--quantization", "fp8"],
    },
    "qwen3-32b": {
        "hf_repo": "Qwen/Qwen3-32B",
        "serve_flags": ["--dtype", "auto", "--quantization", "fp8"],
    },
    "glm-4.7-flash": {
        "hf_repo": "zai-org/GLM-4.7-Flash",
        "serve_flags": ["--dtype", "auto", "--quantization", "fp8", "--trust-remote-code"],
    },
    "llama-3.1-8b-instruct": {
        "hf_repo": "meta-llama/Llama-3.1-8B-Instruct",
        "serve_flags": ["--dtype", "auto"],
    },
    "mistral-7b-instruct-v0.3": {
        "hf_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "serve_flags": ["--dtype", "auto"],
    },
    "gpt-oss-20b": {
        "hf_repo": "openai/gpt-oss-20b",
        "serve_flags": ["--async-scheduling"],
    },
    "gpt-oss-120b": {
        "hf_repo": "openai/gpt-oss-120b",
        "serve_flags": ["--async-scheduling"],
    },
}

LEDGER_METHODS = {"ledger_only", "mus_repair"}
HTTP_RETRY_ATTEMPTS = 3
HTTP_RETRY_BASE_BACKOFF_SEC = 2.0
HTTP_RETRY_MAX_BACKOFF_SEC = 30.0
HTTP_RETRY_JITTER_SEC = 0.75
PROBLEM_RETRY_ATTEMPTS = 2
PROBLEM_RETRY_BACKOFF_SEC = 2
Z3_CALL_TIMEOUT_SEC = 45
DEFAULT_MAX_WORKERS = 16
DEFAULT_MAX_Z3_WORKERS = 8
HISTORY_ASSISTANT_CHAR_BUDGET = 1200
DEFAULT_MAX_TRUNCATION_RETRIES = 2
TRUNCATION_RETRY_MAX_TOKENS = 1024

# Runtime request controls configured from CLI in main().
REQUEST_DETERMINISTIC = False
REQUEST_SEED: int | None = None
REQUEST_OPENROUTER_PROVIDER: str | None = None
OPENROUTER_PROVIDER_WARNED = False


class ProgressState:
    """Shared heartbeat state."""

    def __init__(self, start_time: float):
        self.start_time = start_time
        self.rows_written = 0
        self.current_method = ""
        self.current_problem_id = ""
        self.current_turn = 0
        self.lock = threading.Lock()

    def set_current(self, method: str, problem_id: str, turn_number: int):
        with self.lock:
            self.current_method = method
            self.current_problem_id = problem_id
            self.current_turn = turn_number

    def mark_row(self):
        with self.lock:
            self.rows_written += 1

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "start_time": self.start_time,
                "rows_written": self.rows_written,
                "current_method": self.current_method,
                "current_problem_id": self.current_problem_id,
                "current_turn": self.current_turn,
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CLR experiments and log SQLite results.")
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--method", choices=METHODS, default=None, help="Run only one method if set")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--problems-dir", type=Path, default=Path("data/problems"))
    parser.add_argument("--db-path", type=Path, default=Path("results.db"))
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Force deterministic request settings: seed usage, temperature=0, top_p=1, "
            "and OpenRouter provider pinning (when --openrouter-provider is set)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed sent to the API for reproducibility.",
    )
    parser.add_argument(
        "--openrouter-provider",
        default=os.getenv("OPENROUTER_PROVIDER", ""),
        help=(
            "OpenRouter provider slug to pin when deterministic mode is enabled "
            "(e.g., 'AtlasCloud')."
        ),
    )
    parser.add_argument(
        "--gpt-oss-reasoning-effort",
        choices=["minimal", "low", "medium", "high", "none", "off"],
        default="off",
        help="Reasoning effort passed to GPT-OSS via extra_body.reasoning_effort.",
    )
    parser.add_argument("--ledger-token-budget", type=int, default=3000)
    parser.add_argument(
        "--max-truncation-retries",
        type=int,
        default=DEFAULT_MAX_TRUNCATION_RETRIES,
        help="Retries when a turn output is clipped by max_tokens.",
    )
    parser.add_argument("--max-repair-attempts", type=int, default=2)
    parser.add_argument(
        "--repair-policy",
        choices=["unsat_only", "expanded"],
        default="expanded",
        help="Repair trigger policy for mus_repair.",
    )
    parser.add_argument(
        "--min-new-constraints-for-commitment",
        type=int,
        default=1,
        help="Minimum extracted constraints expected per turn before low-commitment trigger fires.",
    )
    parser.add_argument("--max-problems", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help="Parallel problem workers per method",
    )
    parser.add_argument(
        "--max-z3-workers",
        type=int,
        default=DEFAULT_MAX_Z3_WORKERS,
        help="Max concurrent isolated Z3 subprocess calls",
    )
    parser.add_argument(
        "--z3-timeout-sec",
        type=int,
        default=Z3_CALL_TIMEOUT_SEC,
        help="Timeout per isolated Z3 call",
    )
    parser.add_argument(
        "--io-log-path",
        type=Path,
        default=None,
        help="Optional JSONL path to log per-turn main model input/output payloads.",
    )
    return parser.parse_args()


def _extract_context_budget(body: str) -> tuple[int, int] | None:
    """Return (total_context_tokens, input_tokens) from known vLLM error formats."""
    # Example format:
    # "request has 3420 input tokens (3420 > 16384 - 2048)"
    m = re.search(
        r"request has\s+(\d+)\s+input tokens\s+\((\d+)\s*>\s*(\d+)\s*-\s*(\d+)\)",
        body,
        flags=re.IGNORECASE,
    )
    if m:
        total_ctx = int(m.group(3))
        # The first two captures are the input-token count in this format.
        input_tokens = int(m.group(1))
        return total_ctx, input_tokens

    # Example format:
    # "maximum context length is 16384 tokens ... requested 18100 tokens
    # (16000 in the messages, 2100 in the completion)"
    m = re.search(
        (
            r"maximum context length is\s*(\d+)\s*tokens.*?"
            r"requested\s*(\d+)\s*tokens.*?"
            r"\((\d+)\s*in the messages,\s*(\d+)\s*in the completion\)"
        ),
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        total_ctx = int(m.group(1))
        input_tokens = int(m.group(3))
        return total_ctx, input_tokens

    return None


def ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_suffix: str):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_suffix}")


def create_results_table(conn: sqlite3.Connection):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            method TEXT NOT NULL,
            domain TEXT NOT NULL,
            problem_id TEXT NOT NULL,
            turn_number INTEGER NOT NULL,
            raw_response TEXT,
            tokens_in INTEGER,
            tokens_out INTEGER,
            latency_ms REAL,
            finish_reason TEXT,
            response_truncated INTEGER DEFAULT 0,
            answer_correct INTEGER,
            extraction_null INTEGER,
            ledger_size INTEGER,
            ledger_json TEXT,
            constraint_extraction_success INTEGER,
            z3_sat INTEGER,
            mus_size INTEGER,
            mus_json TEXT,
            repair_attempted INTEGER,
            repair_trigger TEXT,
            repair_trigger_count INTEGER,
            repair_reason_json TEXT,
            z3_error INTEGER DEFAULT 0,
            timestamp TEXT DEFAULT (datetime('now'))
        );
        """
    )
    ensure_column(conn, "results", "z3_error", "INTEGER DEFAULT 0")
    ensure_column(conn, "results", "finish_reason", "TEXT")
    ensure_column(conn, "results", "response_truncated", "INTEGER DEFAULT 0")
    ensure_column(conn, "results", "repair_trigger", "TEXT")
    ensure_column(conn, "results", "repair_trigger_count", "INTEGER")
    ensure_column(conn, "results", "repair_reason_json", "TEXT")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_results_model_method ON results(model, method);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_results_problem ON results(problem_id, turn_number);")
    conn.commit()


def load_problems(problems_dir: Path, split: str) -> list[dict[str, Any]]:
    # Problems may live either as data/problems/<split>/*.json (drift-bench
    # release layout, physically partitioned) or as a flat data/problems/*.json
    # with each JSON carrying a "split" field (original research layout).
    # Support both so existing runbooks keep working.
    problems: list[dict[str, Any]] = []
    split_dir = problems_dir / split
    search_dir = split_dir if split_dir.is_dir() else problems_dir
    for path in sorted(search_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            item = json.load(f)
        if item.get("split") == split:
            problems.append(item)
    return problems


def estimate_tokens_from_text(text: str) -> int:
    return max(1, len(text) // 4)


def ledger_token_count(ledger: list[dict[str, Any]]) -> int:
    if not ledger:
        return 0
    payload = json.dumps({"constraints": ledger}, ensure_ascii=True)
    return estimate_tokens_from_text(payload)


def compact_history_assistant_response(
    response_text: str,
    domain: str,
    entities: list[str],
    context: dict[str, Any],
    max_chars: int = HISTORY_ASSISTANT_CHAR_BUDGET,
) -> tuple[str, str]:
    """Keep conversation history bounded to avoid truncation cascades."""
    text = response_text if isinstance(response_text, str) else ("" if response_text is None else str(response_text))
    if len(text) <= max_chars:
        return text, "full"

    parsed_answer = extract_answer(text, domain, entities, context)
    if parsed_answer is not None:
        compact_json = json.dumps({"solution": parsed_answer}, ensure_ascii=True)
        return compact_json, "parsed_json"

    return text[:max_chars], "char_clip"


def enforce_budget(ledger: list[dict[str, Any]], max_tokens: int) -> list[dict[str, Any]]:
    trimmed = list(ledger)
    while trimmed and ledger_token_count(trimmed) > max_tokens:
        trimmed.pop(0)
    return trimmed


def _is_openrouter_api_base(api_base: str) -> bool:
    return "openrouter.ai" in str(api_base).lower()


def _chat_completions_url(api_base: str) -> str:
    base = str(api_base).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _parse_retry_after_seconds(retry_after_header: str | None) -> float | None:
    if not retry_after_header:
        return None
    value = retry_after_header.strip()
    if not value:
        return None
    try:
        seconds = float(value)
        if seconds >= 0:
            return min(seconds, HTTP_RETRY_MAX_BACKOFF_SEC)
    except ValueError:
        pass

    try:
        dt = email.utils.parsedate_to_datetime(value)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta_sec = (dt - datetime.now(timezone.utc)).total_seconds()
        if delta_sec > 0:
            return min(delta_sec, HTTP_RETRY_MAX_BACKOFF_SEC)
    except Exception:
        return None
    return None


def _compute_http_backoff_sec(
    attempt: int,
    status_code: int | None,
    retry_after_header: str | None,
) -> float:
    retry_after_sec = _parse_retry_after_seconds(retry_after_header)
    if retry_after_sec is not None:
        base = retry_after_sec
    else:
        exponent = max(0, attempt - 1)
        base = min(HTTP_RETRY_MAX_BACKOFF_SEC, HTTP_RETRY_BASE_BACKOFF_SEC * (2**exponent))
        if status_code in {429, 503, 504, 520, 522, 524}:
            base = min(HTTP_RETRY_MAX_BACKOFF_SEC, base * 1.5)
    jitter = random.uniform(0.0, HTTP_RETRY_JITTER_SEC)
    return base + jitter


def call_vllm(
    api_base: str,
    model_repo: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    gpt_oss_reasoning_effort: str = "off",
    force_json_object: bool = False,
    timeout_sec: int = 300,
) -> dict[str, Any]:
    global OPENROUTER_PROVIDER_WARNED

    def _trim_oldest_history_turn(msgs: list[dict[str, Any]]) -> bool:
        """Drop oldest history items while keeping system prompt and latest user prompt."""
        if len(msgs) <= 2:
            return False
        start_idx = 1 if msgs and msgs[0].get("role") == "system" else 0
        # Preserve the latest prompt (typically current user turn).
        keep_tail = 1
        removable = len(msgs) - start_idx - keep_tail
        if removable <= 0:
            return False
        drop_count = 2 if removable >= 2 else 1
        del msgs[start_idx : start_idx + drop_count]
        return True

    def _coerce_message_content(message: dict[str, Any]) -> str:
        content = message.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    for key in ("text", "content", "value", "output_text"):
                        value = item.get(key)
                        if isinstance(value, str):
                            parts.append(value)
                            break
            if parts:
                return "\n".join(parts)

        # GPT-OSS can emit reasoning text while content is null.
        for key in ("reasoning_content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str):
                return value
        return ""

    requested_max_tokens = int(max_tokens)
    effective_max_tokens = max(1, requested_max_tokens)
    if effective_max_tokens != requested_max_tokens:
        print(
            f"[http_adjust] model={model_repo} clamping invalid max_tokens "
            f"{requested_max_tokens}->{effective_max_tokens}"
        )
    is_gpt_oss = model_repo.startswith("openai/gpt-oss")
    is_openrouter = _is_openrouter_api_base(api_base)
    request_url = _chat_completions_url(api_base)
    thinking_payload_mode = "disabled" if is_gpt_oss else "chat_template_kwargs"
    effective_messages: list[dict[str, Any]] = list(messages)
    headers: dict[str, str] = {}

    if is_openrouter:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required when api_base points to OpenRouter."
            )
        headers["Authorization"] = f"Bearer {openrouter_api_key}"
        # Optional but recommended by OpenRouter for analytics/routing context.
        openrouter_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        openrouter_title = os.getenv("OPENROUTER_APP_NAME", "").strip()
        if openrouter_referer:
            headers["HTTP-Referer"] = openrouter_referer
        if openrouter_title:
            headers["X-Title"] = openrouter_title

    last_err: Exception | None = None
    attempt = 1
    while attempt <= HTTP_RETRY_ATTEMPTS:
        if int(effective_max_tokens) < 1:
            print(
                f"[http_adjust] model={model_repo} correcting retry max_tokens "
                f"{effective_max_tokens}->1"
            )
            effective_max_tokens = 1
        effective_temperature = 0.0 if REQUEST_DETERMINISTIC else temperature
        payload = {
            "model": model_repo,
            "messages": effective_messages,
            "temperature": effective_temperature,
            "max_tokens": effective_max_tokens,
        }
        if REQUEST_SEED is not None:
            payload["seed"] = int(REQUEST_SEED)
        if REQUEST_DETERMINISTIC:
            payload["top_p"] = 1.0
            if is_openrouter:
                provider_slug = REQUEST_OPENROUTER_PROVIDER
                if provider_slug:
                    payload["provider"] = {
                        "only": [provider_slug],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                elif not OPENROUTER_PROVIDER_WARNED:
                    OPENROUTER_PROVIDER_WARNED = True
                    print(
                        "[determinism_warn] deterministic mode enabled for OpenRouter "
                        "without --openrouter-provider; provider routing may vary."
                    )
        if force_json_object:
            payload["response_format"] = {"type": "json_object"}
        if is_gpt_oss:
            # Set GPT-OSS reasoning explicitly across backends.
            # OpenRouter supports `reasoning_effort` (paid routes) and `reasoning` controls.
            # vLLM uses extra_body.reasoning_effort.
            if is_openrouter:
                if gpt_oss_reasoning_effort in {"minimal", "low", "medium", "high"}:
                    payload["reasoning_effort"] = gpt_oss_reasoning_effort
                else:
                    payload["reasoning"] = {"effort": "none"}
            else:
                payload["extra_body"] = {"reasoning_effort": gpt_oss_reasoning_effort}
        elif not is_openrouter:
            if thinking_payload_mode == "chat_template_kwargs":
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                payload["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        t0 = time.perf_counter()
        try:
            response = requests.post(
                request_url,
                json=payload,
                headers=headers,
                timeout=timeout_sec,
            )
            latency_ms = (time.perf_counter() - t0) * 1000.0
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(
                    f"vLLM/OpenAI response must be JSON object, got {type(data).__name__}"
                )
            choices = data.get("choices")
            if not isinstance(choices, list) or not choices:
                compact = json.dumps(data, ensure_ascii=True)
                compact = " ".join(compact.split())
                error_message = ""
                if isinstance(data.get("error"), dict):
                    error_message = str(data["error"].get("message") or "")
                print(
                    f"[http_malformed] model={model_repo} missing choices body={compact[:320]}"
                )
                raise RuntimeError(
                    f"vLLM/OpenAI response missing choices error={error_message[:220]}"
                )
            choice = choices[0]
            if not isinstance(choice, dict):
                raise RuntimeError(
                    f"vLLM/OpenAI choice must be JSON object, got {type(choice).__name__}"
                )
            message = choice.get("message", {})
            usage = data.get("usage", {})

            return {
                "content": _coerce_message_content(message),
                "tokens_in": usage.get("prompt_tokens"),
                "tokens_out": usage.get("completion_tokens"),
                "latency_ms": latency_ms,
                "finish_reason": choice.get("finish_reason"),
                "raw": data,
            }
        except Exception as exc:
            last_err = exc
            handled_without_backoff = False
            non_retryable_http = False
            status_code: int | None = None
            retry_after_header: str | None = None
            if isinstance(exc, RuntimeError):
                exc_text = str(exc)
                if (
                    "vLLM/OpenAI response missing choices" in exc_text
                    and "unexpected tokens remaining in message header" in exc_text.lower()
                ):
                    trimmed = _trim_oldest_history_turn(effective_messages)
                    if trimmed:
                        handled_without_backoff = True
                        print(
                            f"[http_adjust] model={model_repo} trimmed oldest history "
                            "after malformed message header"
                        )
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                status = exc.response.status_code
                body = exc.response.text or ""
                status_code = status
                retry_after_header = exc.response.headers.get("Retry-After")
                if status == 400:
                    compact = " ".join(body.split())
                    print(
                        f"[http_400] model={model_repo} body={compact[:280]}"
                    )
                    if re.search(r"max_tokens must be at least 1, got -\d+", body):
                        trimmed = _trim_oldest_history_turn(effective_messages)
                        if trimmed:
                            handled_without_backoff = True
                            print(
                                f"[http_adjust] model={model_repo} trimmed oldest history "
                                "after negative remaining max_tokens"
                            )
                if (
                    status == 400
                    and not is_gpt_oss
                    and thinking_payload_mode == "chat_template_kwargs"
                    and "chat_template_kwargs" in body
                ):
                    thinking_payload_mode = "extra_body"
                    handled_without_backoff = True
                    print(
                        f"[http_adjust] model={model_repo} switching thinking control "
                        "to payload.extra_body.chat_template_kwargs"
                    )
                # vLLM validation error: max_tokens too large for remaining context budget.
                if status == 400 and (
                    "input tokens" in body.lower()
                    or "maximum context length" in body.lower()
                ):
                    budget = _extract_context_budget(body)
                    if budget is not None:
                        total_ctx, input_tokens = budget
                        available = total_ctx - input_tokens
                        available_with_margin = available - 16
                        # Keep a small safety margin for tokenizer/templating jitter.
                        adjusted = max(1, min(int(effective_max_tokens), available_with_margin))
                        if adjusted < int(effective_max_tokens):
                            old_max = int(effective_max_tokens)
                            effective_max_tokens = adjusted
                            handled_without_backoff = True
                            print(
                                f"[http_adjust] model={model_repo} reducing max_tokens "
                                f"{old_max}->{adjusted} due to context budget "
                                f"(input={input_tokens}, total={total_ctx})"
                            )
                        if available_with_margin <= 0:
                            trimmed = _trim_oldest_history_turn(effective_messages)
                            if trimmed:
                                handled_without_backoff = True
                                print(
                                    f"[http_adjust] model={model_repo} trimmed oldest history "
                                    "due to exceeded prompt context"
                                )
                if status == 400 and not handled_without_backoff:
                    non_retryable_http = True
                    print(
                        f"[http_fail_fast] model={model_repo} status=400 "
                        "retry_suppressed=1"
                    )
            if handled_without_backoff:
                continue

            if non_retryable_http:
                break

            if attempt < HTTP_RETRY_ATTEMPTS:
                backoff_sec = _compute_http_backoff_sec(
                    attempt=attempt,
                    status_code=status_code,
                    retry_after_header=retry_after_header,
                )
                retry_after_info = (
                    f" retry_after={retry_after_header}" if retry_after_header else ""
                )
                print(
                    f"[http_retry] attempt={attempt}/{HTTP_RETRY_ATTEMPTS} "
                    f"model={model_repo} status={status_code} error={repr(exc)} "
                    f"backoff={backoff_sec:.2f}s{retry_after_info}"
                )
                time.sleep(backoff_sec)
                attempt += 1
            else:
                break

    if last_err is not None:
        raise last_err
    raise RuntimeError("call_vllm failed with unknown error")


def _z3_worker(function_name: str, kwargs: dict[str, Any], output_queue):
    """Child worker for isolated Z3 execution."""
    try:
        from z3_checker import check_satisfiability, compute_mus, verify_with_z3

        fn_map = {
            "check_satisfiability": check_satisfiability,
            "compute_mus": compute_mus,
            "verify_with_z3": verify_with_z3,
        }
        if function_name not in fn_map:
            output_queue.put({"ok": 0, "error": f"unsupported_function:{function_name}"})
            return
        result = fn_map[function_name](**kwargs)
        output_queue.put({"ok": 1, "result": result})
    except Exception as exc:
        output_queue.put(
            {
                "ok": 0,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def call_z3_isolated(function_name: str, kwargs: dict[str, Any], timeout_sec: int = Z3_CALL_TIMEOUT_SEC) -> dict[str, Any]:
    """Run Z3 logic in a subprocess so SIGABRT/SIGSEGV cannot kill the main experiment process."""
    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(target=_z3_worker, args=(function_name, kwargs, output_queue))
    try:
        proc.start()
        deadline = time.monotonic() + float(timeout_sec)
        # Poll with a deadline to avoid occasional early returns from one-shot join() under thread contention.
        while proc.is_alive():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            proc.join(min(0.1, remaining))

        if proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join(1)
            return {"ok": False, "error": f"timeout_{timeout_sec}s"}

        proc.join(0)
        if proc.exitcode != 0:
            return {"ok": False, "error": f"child_exit_{proc.exitcode}"}

        msg_deadline = time.monotonic() + 1.0
        msg: dict[str, Any] | None = None
        while msg is None:
            remaining = msg_deadline - time.monotonic()
            if remaining <= 0:
                return {"ok": False, "error": "no_result_from_child"}
            try:
                msg = output_queue.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue

        if msg.get("ok") == 1:
            return {"ok": True, "result": msg.get("result")}

        return {
            "ok": False,
            "error": msg.get("error", "unknown_child_error"),
            "traceback": msg.get("traceback"),
        }
    finally:
        output_queue.close()
        output_queue.join_thread()


def log_z3_failure(
    method: str,
    problem_id: str,
    turn_number: int,
    function_name: str,
    domain: str,
    constraints: list[dict[str, Any]] | None,
    error: str,
):
    constraint_preview = "null"
    if constraints is not None:
        constraint_preview = json.dumps(constraints, ensure_ascii=True)
        if len(constraint_preview) > 2500:
            constraint_preview = constraint_preview[:2500] + "..."
    print(
        f"[z3_error] method={method} problem_id={problem_id} turn={turn_number} "
        f"fn={function_name} domain={domain} error={error} constraints={constraint_preview}"
    )


def get_problem_context(problem: dict[str, Any]) -> dict[str, Any]:
    domain = problem["domain"]
    if domain == "seating":
        return {
            "num_entities": problem.get("num_entities", len(problem["entities"])),
            "table_shape": problem.get("table_shape", "round"),
        }
    if domain == "scheduling":
        return {
            "num_slots": problem.get("num_slots", max(6, len(problem["entities"]) + 1)),
            "max_duration": problem.get("max_duration", 3),
        }
    if domain == "logic_grid":
        return {
            "categories": problem.get("categories", {}),
        }
    return {}


def prepare_problem_resume_state(
    conn: sqlite3.Connection,
    db_lock: threading.Lock,
    model_key: str,
    method: str,
    problem_id: str,
    expected_turns: int,
) -> bool:
    """Return True if the problem should run now, False if it should be skipped."""
    with db_lock:
        row = conn.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT turn_number)
            FROM results
            WHERE model = ? AND method = ? AND problem_id = ?
            """,
            (model_key, method, problem_id),
        ).fetchone()
        existing_rows = int(row[0] or 0)
        existing_turns = int(row[1] or 0)

        if existing_rows == 0:
            return True

        is_complete = existing_rows == expected_turns and existing_turns == expected_turns
        if is_complete:
            print(
                f"[resume_skip] warning=existing_complete_rows model={model_key} "
                f"method={method} problem_id={problem_id} rows={existing_rows} "
                f"expected_turns={expected_turns}"
            )
            return False

        conn.execute(
            """
            DELETE FROM results
            WHERE model = ? AND method = ? AND problem_id = ?
            """,
            (model_key, method, problem_id),
        )
        conn.commit()

    print(
        f"[resume_reset] warning=partial_rows_deleted model={model_key} "
        f"method={method} problem_id={problem_id} rows={existing_rows} "
        f"distinct_turns={existing_turns} expected_turns={expected_turns}"
    )
    return True


def insert_row(
    conn: sqlite3.Connection,
    row: dict[str, Any],
    db_lock: threading.Lock,
    progress_state: ProgressState,
):
    with db_lock:
        conn.execute(
            """
            INSERT INTO results (
                model, method, domain, problem_id, turn_number,
                raw_response, tokens_in, tokens_out, latency_ms, finish_reason, response_truncated,
                answer_correct, extraction_null,
                ledger_size, ledger_json, constraint_extraction_success,
                z3_sat, mus_size, mus_json, repair_attempted,
                repair_trigger, repair_trigger_count, repair_reason_json, z3_error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("model"),
                row.get("method"),
                row.get("domain"),
                row.get("problem_id"),
                row.get("turn_number"),
                row.get("raw_response"),
                row.get("tokens_in"),
                row.get("tokens_out"),
                row.get("latency_ms"),
                row.get("finish_reason"),
                row.get("response_truncated", 0),
                row.get("answer_correct"),
                row.get("extraction_null"),
                row.get("ledger_size"),
                row.get("ledger_json"),
                row.get("constraint_extraction_success"),
                row.get("z3_sat"),
                row.get("mus_size"),
                row.get("mus_json"),
                row.get("repair_attempted"),
                row.get("repair_trigger"),
                row.get("repair_trigger_count"),
                row.get("repair_reason_json"),
                row.get("z3_error", 0),
            ),
        )
        conn.commit()
    progress_state.mark_row()


def heartbeat_loop(stop_event: threading.Event, progress_state: ProgressState):
    while not stop_event.wait(60):
        snap = progress_state.snapshot()
        elapsed = time.time() - snap["start_time"]
        print(
            f"[heartbeat] elapsed={elapsed/60.0:.1f}min rows={snap['rows_written']} "
            f"method={snap['current_method']} problem_id={snap['current_problem_id']} "
            f"turn={snap['current_turn']}"
        )


def append_io_log(io_log_path: Path | None, io_lock: threading.Lock | None, payload: dict[str, Any]) -> None:
    """Append one JSONL entry for optional model I/O tracing."""
    if io_log_path is None or io_lock is None:
        return
    line = json.dumps(payload, ensure_ascii=True)
    with io_lock:
        with io_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_problem(
    problem: dict[str, Any],
    method: str,
    model_key: str,
    model_repo: str,
    api_base: str,
    temperature: float,
    gpt_oss_reasoning_effort: str,
    max_tokens: int,
    max_truncation_retries: int,
    ledger_token_budget: int,
    max_repair_attempts: int,
    repair_policy: str,
    min_new_constraints_for_commitment: int,
    conn: sqlite3.Connection,
    db_lock: threading.Lock,
    progress_state: ProgressState,
    z3_semaphore: threading.Semaphore,
    z3_timeout_sec: int,
    io_log_path: Path | None,
    io_lock: threading.Lock | None,
):
    domain = problem["domain"]
    entities = problem["entities"]
    problem_id = problem["problem_id"]
    context = get_problem_context(problem)
    expected_turns = len(problem.get("turns", []))
    should_run = prepare_problem_resume_state(
        conn=conn,
        db_lock=db_lock,
        model_key=model_key,
        method=method,
        problem_id=problem_id,
        expected_turns=expected_turns,
    )
    if not should_run:
        return

    conversation_history: list[dict[str, str]] = []
    ledger: list[dict[str, Any]] = []
    next_constraint_id = 1

    for turn in problem["turns"]:
        turn_no = int(turn["turn_number"])
        progress_state.set_current(method, problem_id, turn_no)

        repair_attempted = 0
        last_mus: list[dict[str, Any]] | None = None
        z3_sat: int | None = None
        constraint_extraction_success: int | None = None
        z3_error = 0
        repair_events: list[dict[str, Any]] = []
        repair_trigger_codes: list[str] = []

        final_response_text = ""
        final_finish_reason: str | None = None
        response_truncated = 0
        total_tokens_in = 0
        total_tokens_out = 0
        total_latency_ms = 0.0

        # For repair loops, we keep prior-turn history fixed and only commit the final response.
        repair_signal: str | None = None
        final_ledger_for_turn = list(ledger)
        use_sat_triggers = method == "mus_repair" and policy_uses_sat_triggers(repair_policy)

        main_attempts = 1
        if method == "mus_repair":
            main_attempts = max_repair_attempts + 1

        for attempt_idx in range(main_attempts):
            if attempt_idx > 0:
                repair_attempted = 1

            messages = build_turn_messages(
                method=method,
                conversation_history=conversation_history,
                new_user_message=turn["user_message"],
                domain=domain,
                entities=entities,
                context=context,
                ledger=ledger if method in LEDGER_METHODS else None,
                repair_signal=repair_signal,
            )

            main_raw = call_vllm(
                api_base=api_base,
                model_repo=model_repo,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                gpt_oss_reasoning_effort=gpt_oss_reasoning_effort,
            )
            final_response_text = main_raw["content"]
            final_finish_reason_raw = str(main_raw.get("finish_reason") or "").strip()
            final_finish_reason = final_finish_reason_raw or None
            response_truncated = int(final_finish_reason_raw == "length")
            total_tokens_in += int(main_raw.get("tokens_in") or 0)
            main_tokens_out = int(main_raw.get("tokens_out") or 0)
            total_tokens_out += main_tokens_out
            total_latency_ms += float(main_raw.get("latency_ms") or 0.0)
            if main_tokens_out >= int(max_tokens):
                print(
                    f"[cap_hit] model={model_key} method={method} "
                    f"problem_id={problem_id} turn={turn_no} attempt={attempt_idx + 1} "
                    f"tokens_out={main_tokens_out} max_tokens={int(max_tokens)}"
                )
            if response_truncated:
                print(
                    f"[truncation] model={model_key} method={method} "
                    f"problem_id={problem_id} turn={turn_no} attempt={attempt_idx + 1} "
                    f"finish_reason=length"
                )
            append_io_log(
                io_log_path=io_log_path,
                io_lock=io_lock,
                payload={
                    "event": "main_generation",
                    "timestamp": time.time(),
                    "model_key": model_key,
                    "model_repo": model_repo,
                    "method": method,
                    "domain": domain,
                    "problem_id": problem_id,
                    "turn_number": turn_no,
                    "attempt": attempt_idx + 1,
                    "messages": messages,
                    "response": {
                        "content": final_response_text,
                        "tokens_in": main_raw.get("tokens_in"),
                        "tokens_out": main_raw.get("tokens_out"),
                        "latency_ms": main_raw.get("latency_ms"),
                        "finish_reason": main_raw.get("finish_reason"),
                    },
                },
            )

            if response_truncated and max_truncation_retries > 0:
                retry_messages = build_truncation_retry_messages(
                    base_messages=messages,
                    domain=domain,
                    entities=entities,
                    context=context,
                )
                retry_max_tokens = max(256, min(TRUNCATION_RETRY_MAX_TOKENS, int(max_tokens)))
                for trunc_retry_idx in range(max_truncation_retries):
                    retry_raw = call_vllm(
                        api_base=api_base,
                        model_repo=model_repo,
                        messages=retry_messages,
                        temperature=0.0,
                        max_tokens=retry_max_tokens,
                        gpt_oss_reasoning_effort=gpt_oss_reasoning_effort,
                        force_json_object=True,
                    )
                    retry_finish_reason_raw = str(retry_raw.get("finish_reason") or "").strip()
                    retry_truncated = int(retry_finish_reason_raw == "length")
                    retry_content = retry_raw.get("content", "")
                    retry_answer = None
                    if isinstance(retry_content, str) and retry_content:
                        retry_answer = extract_answer(retry_content, domain, entities, context)
                    retry_tokens_out = int(retry_raw.get("tokens_out") or 0)
                    total_tokens_in += int(retry_raw.get("tokens_in") or 0)
                    total_tokens_out += retry_tokens_out
                    total_latency_ms += float(retry_raw.get("latency_ms") or 0.0)
                    append_io_log(
                        io_log_path=io_log_path,
                        io_lock=io_lock,
                        payload={
                            "event": "truncation_retry_generation",
                            "timestamp": time.time(),
                            "model_key": model_key,
                            "model_repo": model_repo,
                            "method": method,
                            "domain": domain,
                            "problem_id": problem_id,
                            "turn_number": turn_no,
                            "retry_index": trunc_retry_idx + 1,
                            "retry_max_tokens": retry_max_tokens,
                            "messages": retry_messages,
                            "response": {
                                "content": retry_content,
                                "tokens_in": retry_raw.get("tokens_in"),
                                "tokens_out": retry_raw.get("tokens_out"),
                                "latency_ms": retry_raw.get("latency_ms"),
                                "finish_reason": retry_raw.get("finish_reason"),
                            },
                        },
                    )
                    if retry_truncated == 0 and retry_answer is not None:
                        final_response_text = json.dumps({"solution": retry_answer}, ensure_ascii=True)
                        final_finish_reason = "stop"
                        response_truncated = 0
                        print(
                            f"[truncation_recovered] model={model_key} method={method} "
                            f"problem_id={problem_id} turn={turn_no} "
                            f"retry={trunc_retry_idx + 1} max_tokens={retry_max_tokens}"
                        )
                        break
                    if retry_truncated == 0 and retry_answer is None:
                        print(
                            f"[truncation_retry_invalid] model={model_key} method={method} "
                            f"problem_id={problem_id} turn={turn_no} retry={trunc_retry_idx + 1} "
                            "finish_reason=stop parsed_answer=0"
                        )
                        retry_max_tokens = max(256, retry_max_tokens // 2)
                        continue
                    print(
                        f"[truncation_retry] model={model_key} method={method} "
                        f"problem_id={problem_id} turn={turn_no} retry={trunc_retry_idx + 1} "
                        f"finish_reason=length tokens_out={retry_tokens_out} max_tokens={retry_max_tokens}"
                    )
                    retry_max_tokens = max(256, retry_max_tokens // 2)
                if response_truncated:
                    parsed_from_clipped = extract_answer(final_response_text, domain, entities, context)
                    if parsed_from_clipped is not None:
                        final_response_text = json.dumps({"solution": parsed_from_clipped}, ensure_ascii=True)
                        final_finish_reason = "postprocess"
                        response_truncated = 0
                        print(
                            f"[truncation_canonicalized] model={model_key} method={method} "
                            f"problem_id={problem_id} turn={turn_no} source=clipped_response"
                        )
                    else:
                        fallback_messages = build_truncation_retry_messages(
                            base_messages=messages,
                            domain=domain,
                            entities=entities,
                            context=context,
                        )
                        fallback_raw = call_vllm(
                            api_base=api_base,
                            model_repo=model_repo,
                            messages=fallback_messages,
                            temperature=0.0,
                            max_tokens=512,
                            gpt_oss_reasoning_effort=gpt_oss_reasoning_effort,
                            force_json_object=False,
                        )
                        total_tokens_in += int(fallback_raw.get("tokens_in") or 0)
                        total_tokens_out += int(fallback_raw.get("tokens_out") or 0)
                        total_latency_ms += float(fallback_raw.get("latency_ms") or 0.0)
                        fallback_finish_reason_raw = str(fallback_raw.get("finish_reason") or "").strip()
                        fallback_content = fallback_raw.get("content", "")
                        fallback_answer = None
                        if isinstance(fallback_content, str) and fallback_content:
                            fallback_answer = extract_answer(fallback_content, domain, entities, context)
                        append_io_log(
                            io_log_path=io_log_path,
                            io_lock=io_lock,
                            payload={
                                "event": "truncation_fallback_generation",
                                "timestamp": time.time(),
                                "model_key": model_key,
                                "model_repo": model_repo,
                                "method": method,
                                "domain": domain,
                                "problem_id": problem_id,
                                "turn_number": turn_no,
                                "messages": fallback_messages,
                                "response": {
                                    "content": fallback_content,
                                    "tokens_in": fallback_raw.get("tokens_in"),
                                    "tokens_out": fallback_raw.get("tokens_out"),
                                    "latency_ms": fallback_raw.get("latency_ms"),
                                    "finish_reason": fallback_raw.get("finish_reason"),
                                },
                            },
                        )
                        if fallback_finish_reason_raw != "length" and fallback_answer is not None:
                            final_response_text = json.dumps({"solution": fallback_answer}, ensure_ascii=True)
                            final_finish_reason = "fallback"
                            response_truncated = 0
                            print(
                                f"[truncation_fallback_recovered] model={model_key} method={method} "
                                f"problem_id={problem_id} turn={turn_no}"
                            )
                        else:
                            retry_extract_answer, retry_extract_null, _ = extract_answer_with_retry(
                                response_text=final_response_text,
                                domain=domain,
                                entities=entities,
                                context=context,
                                call_fn=lambda **kwargs: call_vllm(
                                    api_base=api_base,
                                    model_repo=model_repo,
                                    messages=kwargs["messages"],
                                    temperature=0.0,
                                    max_tokens=min(512, kwargs.get("max_tokens", 512)),
                                    gpt_oss_reasoning_effort=gpt_oss_reasoning_effort,
                                    force_json_object=True,
                                ),
                            )
                            if retry_extract_answer is not None and retry_extract_null == 0:
                                final_response_text = json.dumps(
                                    {"solution": retry_extract_answer},
                                    ensure_ascii=True,
                                )
                                final_finish_reason = "truncation_retry_extract"
                                response_truncated = 0
                                print(
                                    f"[truncation_extract_recovered] model={model_key} method={method} "
                                    f"problem_id={problem_id} turn={turn_no}"
                                )
                            else:
                                cumulative_constraints = turn.get("cumulative_constraints", [])
                                schema_hint = {"solution": {entity: {} for entity in entities}}
                                regen_user_message = (
                                    "The prior response kept getting clipped.\n"
                                    "Re-solve ONLY from cumulative constraints and return exactly one JSON object.\n"
                                    "Do not include any prose or explanation.\n"
                                    f"Required JSON shape:\n{json.dumps(schema_hint, ensure_ascii=True)}\n\n"
                                    f"Cumulative constraints:\n{json.dumps(cumulative_constraints, ensure_ascii=True)}"
                                )
                                regen_messages = build_turn_messages(
                                    method="direct",
                                    conversation_history=[],
                                    new_user_message=regen_user_message,
                                    domain=domain,
                                    entities=entities,
                                    context=context,
                                    ledger=None,
                                    repair_signal=None,
                                )
                                regen_raw = call_vllm(
                                    api_base=api_base,
                                    model_repo=model_repo,
                                    messages=regen_messages,
                                    temperature=0.0,
                                    max_tokens=384,
                                    gpt_oss_reasoning_effort=gpt_oss_reasoning_effort,
                                    force_json_object=True,
                                )
                                total_tokens_in += int(regen_raw.get("tokens_in") or 0)
                                total_tokens_out += int(regen_raw.get("tokens_out") or 0)
                                total_latency_ms += float(regen_raw.get("latency_ms") or 0.0)
                                regen_finish_reason_raw = str(regen_raw.get("finish_reason") or "").strip()
                                regen_content = regen_raw.get("content", "")
                                regen_answer = None
                                if isinstance(regen_content, str) and regen_content:
                                    regen_answer = extract_answer(regen_content, domain, entities, context)
                                append_io_log(
                                    io_log_path=io_log_path,
                                    io_lock=io_lock,
                                    payload={
                                        "event": "truncation_last_resort_generation",
                                        "timestamp": time.time(),
                                        "model_key": model_key,
                                        "model_repo": model_repo,
                                        "method": method,
                                        "domain": domain,
                                        "problem_id": problem_id,
                                        "turn_number": turn_no,
                                        "messages": regen_messages,
                                        "response": {
                                            "content": regen_content,
                                            "tokens_in": regen_raw.get("tokens_in"),
                                            "tokens_out": regen_raw.get("tokens_out"),
                                            "latency_ms": regen_raw.get("latency_ms"),
                                            "finish_reason": regen_raw.get("finish_reason"),
                                        },
                                    },
                                )
                                if regen_finish_reason_raw != "length" and regen_answer is not None:
                                    final_response_text = json.dumps({"solution": regen_answer}, ensure_ascii=True)
                                    final_finish_reason = "truncation_last_resort"
                                    response_truncated = 0
                                    print(
                                        f"[truncation_last_resort_recovered] model={model_key} method={method} "
                                        f"problem_id={problem_id} turn={turn_no}"
                                    )
                                else:
                                    print(
                                        f"[truncation_unresolved] model={model_key} method={method} "
                                        f"problem_id={problem_id} turn={turn_no} retries={max_truncation_retries}"
                                    )

            # Methods without runtime ledger operations are done after one main answer.
            if method not in LEDGER_METHODS:
                break

            if response_truncated:
                extracted = None
                constraint_success = 0
            else:
                extracted, constraint_success, _ = extract_constraints_with_retry(
                    assistant_response=final_response_text,
                    domain=domain,
                    turn_number=turn_no,
                    user_message=turn["user_message"],
                    entities=entities,
                    context=context,
                    call_fn=lambda **kwargs: call_vllm(
                        api_base=api_base,
                        model_repo=model_repo,
                        messages=kwargs["messages"],
                        temperature=kwargs.get("temperature", 0.0),
                        max_tokens=kwargs.get("max_tokens", 900),
                        gpt_oss_reasoning_effort=gpt_oss_reasoning_effort,
                    ),
                )
            # success polarity: 1=success, 0=failure -> maps to constraint_extraction_success.
            constraint_extraction_success = constraint_success

            local_counter = next_constraint_id
            turn_constraints: list[dict[str, Any]] | None = None
            if extracted is not None:
                # Runtime extraction succeeded: assign stable IDs for this attempt.
                turn_constraints = []
                for c in extracted:
                    c_copy = dict(c)
                    c_copy["id"] = f"C{local_counter:03d}"
                    c_copy["source_turn"] = turn_no
                    turn_constraints.append(c_copy)
                    local_counter += 1

            candidate_ledger = list(ledger)
            if turn_constraints is not None:
                candidate_ledger = ledger + turn_constraints
                candidate_ledger = enforce_budget(candidate_ledger, ledger_token_budget)

            if method == "ledger_only":
                if turn_constraints is not None:
                    final_ledger_for_turn = candidate_ledger
                    next_constraint_id = local_counter
                else:
                    final_ledger_for_turn = list(ledger)
                break

            with z3_semaphore:
                sat_call = call_z3_isolated(
                    "check_satisfiability",
                    {
                        "constraints": candidate_ledger,
                        "domain": domain,
                        "entities": entities,
                        "context": context,
                    },
                    timeout_sec=z3_timeout_sec,
                )
            if not sat_call["ok"]:
                z3_error = 1
                z3_sat = None
                log_z3_failure(
                    method,
                    problem_id,
                    turn_no,
                    "check_satisfiability",
                    domain,
                    candidate_ledger,
                    sat_call["error"],
                )
                final_ledger_for_turn = list(ledger)
                break

            sat_result = sat_call["result"]
            z3_sat = 1 if sat_result["is_sat"] else 0

            mus: list[dict[str, Any]] | None = None
            if not sat_result["is_sat"]:
                with z3_semaphore:
                    mus_call = call_z3_isolated(
                        "compute_mus",
                        {
                            "constraints": candidate_ledger,
                            "domain": domain,
                            "entities": entities,
                            "context": context,
                        },
                        timeout_sec=z3_timeout_sec,
                    )
                if not mus_call["ok"]:
                    z3_error = 1
                    last_mus = []
                    log_z3_failure(
                        method,
                        problem_id,
                        turn_no,
                        "compute_mus",
                        domain,
                        candidate_ledger,
                        mus_call["error"],
                    )
                    final_ledger_for_turn = list(ledger)
                    break
                mus = mus_call["result"]
                last_mus = mus

            quick_answer = None
            answer_ledger_conflict: bool | None = None
            if use_sat_triggers and sat_result["is_sat"]:
                quick_answer = extract_answer(
                    response_text=final_response_text,
                    domain=domain,
                    entities=entities,
                    context=context,
                )
                if quick_answer is not None:
                    with z3_semaphore:
                        verify_ledger_call = call_z3_isolated(
                            "verify_with_z3",
                            {
                                "answer": quick_answer,
                                "cumulative_constraints": candidate_ledger,
                                "domain": domain,
                                "entities": entities,
                                "context": context,
                            },
                            timeout_sec=z3_timeout_sec,
                        )
                    if verify_ledger_call["ok"]:
                        answer_ledger_conflict = int(verify_ledger_call["result"]) == 0
                    else:
                        z3_error = 1
                        answer_ledger_conflict = None
                        log_z3_failure(
                            method,
                            problem_id,
                            turn_no,
                            "verify_with_z3",
                            domain,
                            candidate_ledger,
                            verify_ledger_call["error"],
                        )
                        final_ledger_for_turn = list(ledger)
                        break

            attempt_reasons = build_repair_reasons(
                extracted_constraints=turn_constraints,
                domain=domain,
                entities=entities,
                context=context,
                is_sat=sat_result["is_sat"],
                quick_answer=quick_answer,
                answer_ledger_conflict=answer_ledger_conflict,
                min_new_constraints_for_commitment=min_new_constraints_for_commitment,
                use_sat_triggers=use_sat_triggers,
            )
            if repair_policy == "unsat_only":
                attempt_reasons = [r for r in attempt_reasons if r.get("code") == UNSAT_LEDGER]

            if attempt_reasons:
                codes = unique_trigger_codes(attempt_reasons)
                repair_trigger_codes = ordered_unique(repair_trigger_codes + codes)

                event = {
                    "attempt": attempt_idx + 1,
                    "codes": codes,
                    "reason_count": len(attempt_reasons),
                    "reasons": attempt_reasons,
                    "candidate_ledger_size": len(candidate_ledger),
                    "candidate_is_sat": int(sat_result["is_sat"]),
                    "extracted_constraints_count": len(turn_constraints or []),
                }
                if mus is not None:
                    event["mus_size"] = len(mus)

                # Retry only while budget remains; otherwise keep final diagnostics.
                if attempt_idx < main_attempts - 1:
                    repair_attempted = 1
                    repair_signal = format_repair_signal(
                        trigger_reasons=attempt_reasons,
                        mus_constraints=mus,
                    )
                    repair_events.append(event)
                    final_ledger_for_turn = list(ledger)
                    continue

                event["attempt_budget_exhausted"] = 1
                repair_events.append(event)

            final_ledger_for_turn = candidate_ledger
            if turn_constraints is not None:
                next_constraint_id = local_counter
            break

        history_assistant_text, history_mode = compact_history_assistant_response(
            response_text=final_response_text,
            domain=domain,
            entities=entities,
            context=context,
        )
        if history_mode != "full":
            print(
                f"[history_compact] model={model_key} method={method} "
                f"problem_id={problem_id} turn={turn_no} mode={history_mode} "
                f"original_chars={len(final_response_text)} compact_chars={len(history_assistant_text)}"
            )

        # Commit final turn output into history.
        conversation_history.append({"role": "user", "content": turn["user_message"]})
        conversation_history.append({"role": "assistant", "content": history_assistant_text})

        if method in LEDGER_METHODS:
            ledger = final_ledger_for_turn
            ledger = enforce_budget(ledger, ledger_token_budget)

        if method == "mus_repair" and z3_sat is None and z3_error == 0:
            with z3_semaphore:
                sat_call = call_z3_isolated(
                    "check_satisfiability",
                    {
                        "constraints": ledger,
                        "domain": domain,
                        "entities": entities,
                        "context": context,
                    },
                    timeout_sec=z3_timeout_sec,
                )
            if sat_call["ok"]:
                z3_sat = 1 if sat_call["result"]["is_sat"] else 0
            else:
                z3_error = 1
                z3_sat = None
                log_z3_failure(
                    method,
                    problem_id,
                    turn_no,
                    "check_satisfiability",
                    domain,
                    ledger,
                    sat_call["error"],
                )

        # Evaluation-only extraction (all methods): one retry then move on.
        if response_truncated:
            answer, extraction_null = None, 1
        else:
            answer, extraction_null, _ = extract_answer_with_retry(
                response_text=final_response_text,
                domain=domain,
                entities=entities,
                context=context,
                call_fn=lambda **kwargs: call_vllm(
                    api_base=api_base,
                    model_repo=model_repo,
                    messages=kwargs["messages"],
                    temperature=kwargs.get("temperature", 0.0),
                    max_tokens=kwargs.get("max_tokens", 1024),
                    gpt_oss_reasoning_effort=gpt_oss_reasoning_effort,
                ),
            )
        # extraction_null polarity: 0=success, 1=failure -> maps to extraction_null.

        if z3_error == 1 or answer is None:
            answer_correct = 0
        else:
            with z3_semaphore:
                verify_call = call_z3_isolated(
                    "verify_with_z3",
                    {
                        "answer": answer,
                        "cumulative_constraints": turn["cumulative_constraints"],
                        "domain": domain,
                        "entities": entities,
                        "context": context,
                    },
                    timeout_sec=z3_timeout_sec,
                )
            if verify_call["ok"]:
                answer_correct = int(verify_call["result"])
            else:
                z3_error = 1
                answer_correct = 0
                log_z3_failure(
                    method,
                    problem_id,
                    turn_no,
                    "verify_with_z3",
                    domain,
                    turn["cumulative_constraints"],
                    verify_call["error"],
                )

        row = {
            "model": model_key,
            "method": method,
            "domain": domain,
            "problem_id": problem_id,
            "turn_number": turn_no,
            "raw_response": final_response_text,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "latency_ms": total_latency_ms,
            "finish_reason": final_finish_reason,
            "response_truncated": int(response_truncated),
            "answer_correct": int(answer_correct),
            "extraction_null": int(extraction_null),
            "ledger_size": len(ledger) if method in LEDGER_METHODS else None,
            "ledger_json": json.dumps(ledger, ensure_ascii=True) if method in LEDGER_METHODS else None,
            "constraint_extraction_success": (
                int(constraint_extraction_success)
                if method in LEDGER_METHODS and constraint_extraction_success is not None
                else (0 if method in LEDGER_METHODS else None)
            ),
            "z3_sat": int(z3_sat) if method == "mus_repair" and z3_sat is not None else None,
            "mus_size": len(last_mus) if (method == "mus_repair" and last_mus) else None,
            "mus_json": (
                json.dumps(last_mus, ensure_ascii=True)
                if method == "mus_repair" and last_mus is not None
                else None
            ),
            "repair_attempted": int(repair_attempted) if method == "mus_repair" else None,
            "repair_trigger": (
                ",".join(repair_trigger_codes)
                if method == "mus_repair" and repair_trigger_codes
                else None
            ),
            "repair_trigger_count": (
                len(repair_events)
                if method == "mus_repair" and repair_events
                else 0 if method == "mus_repair" else None
            ),
            "repair_reason_json": (
                json.dumps(repair_events, ensure_ascii=True)
                if method == "mus_repair" and repair_events
                else None
            ),
            "z3_error": int(z3_error),
        }
        insert_row(conn, row, db_lock, progress_state)


def run_problem_safe(**kwargs) -> bool:
    problem = kwargs["problem"]
    method = kwargs["method"]
    for attempt_idx in range(1, PROBLEM_RETRY_ATTEMPTS + 1):
        try:
            run_problem(**kwargs)
            return True
        except Exception as exc:
            print(
                f"[problem_error] method={method} problem_id={problem.get('problem_id')} "
                f"attempt={attempt_idx}/{PROBLEM_RETRY_ATTEMPTS} error={repr(exc)}"
            )
            print(traceback.format_exc())
            if attempt_idx >= PROBLEM_RETRY_ATTEMPTS:
                return False
            print(
                f"[problem_retry] method={method} problem_id={problem.get('problem_id')} "
                f"next_attempt={attempt_idx + 1}/{PROBLEM_RETRY_ATTEMPTS} "
                f"backoff={PROBLEM_RETRY_BACKOFF_SEC}s"
            )
            time.sleep(PROBLEM_RETRY_BACKOFF_SEC)
    return False


def main():
    args = parse_args()
    global REQUEST_DETERMINISTIC, REQUEST_SEED, REQUEST_OPENROUTER_PROVIDER

    REQUEST_DETERMINISTIC = bool(args.deterministic)
    REQUEST_SEED = int(args.seed) if args.seed is not None else None
    REQUEST_OPENROUTER_PROVIDER = args.openrouter_provider.strip() or None

    if args.model not in MODELS:
        raise ValueError(f"Unknown model: {args.model}")

    methods = [args.method] if args.method else list(METHODS)
    model_repo = MODELS[args.model]["hf_repo"]
    if REQUEST_DETERMINISTIC:
        print(
            f"[determinism] enabled=1 seed={REQUEST_SEED} "
            f"openrouter_provider={REQUEST_OPENROUTER_PROVIDER or 'unset'}"
        )

    problems = load_problems(args.problems_dir, args.split)
    if args.max_problems > 0:
        problems = problems[: args.max_problems]

    if not problems:
        raise RuntimeError(f"No problems found in {args.problems_dir} for split={args.split}")

    conn = sqlite3.connect(str(args.db_path), check_same_thread=False)
    db_lock = threading.Lock()
    io_lock: threading.Lock | None = None
    io_log_path: Path | None = None
    if args.io_log_path is not None:
        io_log_path = Path(args.io_log_path)
        io_log_path.parent.mkdir(parents=True, exist_ok=True)
        io_log_path.write_text("", encoding="utf-8")
        io_lock = threading.Lock()
    z3_workers = max(1, int(args.max_z3_workers))
    z3_semaphore = threading.Semaphore(z3_workers)
    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.z3_timeout_sec < 1:
        raise ValueError("--z3-timeout-sec must be >= 1")
    if args.max_repair_attempts < 0:
        raise ValueError("--max-repair-attempts must be >= 0")
    if args.max_truncation_retries < 0:
        raise ValueError("--max-truncation-retries must be >= 0")
    if args.min_new_constraints_for_commitment < 1:
        raise ValueError("--min-new-constraints-for-commitment must be >= 1")

    # Better write behavior under threaded inserts and frequent commits.
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
    except sqlite3.DatabaseError as exc:
        print(f"[sqlite_warn] pragma_setup_failed error={repr(exc)}")
    create_results_table(conn)

    start_time = time.time()
    progress_state = ProgressState(start_time)
    heartbeat_stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=heartbeat_loop,
        args=(heartbeat_stop, progress_state),
        daemon=True,
    )
    heartbeat_thread.start()

    total_cells = len(problems) * len(methods)
    done = 0
    failed_problems = 0

    try:
        for method in methods:
            print(
                f"[run] model={args.model} method={method} split={args.split} "
                f"problems={len(problems)} repair_policy={args.repair_policy} "
                f"max_repair_attempts={args.max_repair_attempts}"
            )

        max_workers_total = max(1, int(args.max_workers) * len(methods))
        with ThreadPoolExecutor(max_workers=max_workers_total) as executor:
            futures = [
                executor.submit(
                    run_problem_safe,
                    problem=problem,
                    method=method,
                    model_key=args.model,
                    model_repo=model_repo,
                    api_base=args.api_base,
                    temperature=args.temperature,
                    gpt_oss_reasoning_effort=args.gpt_oss_reasoning_effort,
                    max_tokens=args.max_tokens,
                    max_truncation_retries=args.max_truncation_retries,
                    ledger_token_budget=args.ledger_token_budget,
                    max_repair_attempts=args.max_repair_attempts,
                    repair_policy=args.repair_policy,
                    min_new_constraints_for_commitment=args.min_new_constraints_for_commitment,
                    conn=conn,
                    db_lock=db_lock,
                    progress_state=progress_state,
                    z3_semaphore=z3_semaphore,
                    z3_timeout_sec=int(args.z3_timeout_sec),
                    io_log_path=io_log_path,
                    io_lock=io_lock,
                )
                for method in methods
                for problem in problems
            ]
            for future in as_completed(futures):
                ok = future.result()
                done += 1
                if not ok:
                    failed_problems += 1
                if done % 25 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[progress] {done}/{total_cells} cells completed in {elapsed/60.0:.1f} min "
                        f"failed_problems={failed_problems}"
                    )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=2)
        with db_lock:
            conn.commit()
            conn.close()

    elapsed = time.time() - start_time
    print(
        f"[done] model={args.model} split={args.split} methods={methods} "
        f"elapsed={elapsed/60.0:.2f} min failed_problems={failed_problems}"
    )


if __name__ == "__main__":
    main()
