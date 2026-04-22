#!/usr/bin/env python3
"""
Scenario Benchmark — ε across paper Scenarios A–E and 30-prompt calibration set
=================================================================================
Runs every code-generation scenario referenced in the paper through the same
infrastructure as benchmark_production.py, adding fields required for the
review loop: generated text, top-5 alternatives, saved signature/context.

Prompt modes
  "full"  natural language prompt; system asks for complete Python code.
          Used for LOW (calibration, FP test) and HIGH (Scenarios A–D + high_05–20).
  "body"  formal function signature; system asks for indented body only.
          Used for Scenario E (6 auth-module functions).

Type classification (mirrors production benchmark)
  "LOGIC"  LOW prompts  — ε should stay COMPLETE; any fire = false positive
  "API"    HIGH prompts — ε should fire FLAGGED/PAUSED; silence = missed

Output: results/scenarios_<model_slug>.json

Usage:
    python benchmark_scenarios.py
    python benchmark_scenarios.py --model gpt-4o-mini
    python benchmark_scenarios.py --provider together --model deepseek-ai/DeepSeek-V3
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

# Load .env from project root (two levels up)
_env = Path(__file__).parent.parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DELTA = 0.30
K = 5

SYSTEM_BODY = (
    "You are a Python expert. Output ONLY the function body "
    "(indented, no signature line, no markdown fences)."
)
SYSTEM_FULL = (
    "You are a Python expert. Output ONLY Python code "
    "(no explanation, no markdown fences, no preamble)."
)

# Context header prepended to Scenario E (body-mode) prompts
BODY_CONTEXT_HEADER = dedent("""\
    You are implementing one function from a Python authentication module.
    Stack: Python 3.11, FastAPI backend, PostgreSQL via SQLAlchemy.
    No authentication library is committed — choose one consistently.
    The module contains: register_user, login, verify_token,
    refresh_token, request_password_reset, reset_password.

    Implement ONLY the function body. Do not repeat the signature or imports.
    Write idiomatic, production-quality Python.
""")

# ---------------------------------------------------------------------------
# Prompts — LOW (calibration, FP test)
# ---------------------------------------------------------------------------
LOW_PROMPTS = [
    {
        "id": "low_01", "scenario": None,
        "label": "Sort integers", "type": "LOGIC", "mode": "full",
        "prompt": "Write a Python function that takes a list of integers and returns it sorted in ascending order.",
        "context": "Python 3.11",
    },
    {
        "id": "low_02", "scenario": None,
        "label": "Palindrome check", "type": "LOGIC", "mode": "full",
        "prompt": "Write a Python function that returns True if a given string is a palindrome, False otherwise. Ignore case.",
        "context": "Python 3.11",
    },
    {
        "id": "low_03", "scenario": None,
        "label": "Fibonacci (nth)", "type": "LOGIC", "mode": "full",
        "prompt": "Write a Python function that returns the nth Fibonacci number. n is a non-negative integer.",
        "context": "Python 3.11",
    },
    {
        "id": "low_04", "scenario": None,
        "label": "List maximum", "type": "LOGIC", "mode": "full",
        "prompt": "Write a Python function that returns the maximum value in a list of numbers without using the built-in max().",
        "context": "Python 3.11",
    },
    {
        "id": "low_05", "scenario": None,
        "label": "Word frequency", "type": "LOGIC", "mode": "full",
        "prompt": "Write a Python function that takes a string and returns a dictionary mapping each word to its frequency.",
        "context": "Python 3.11",
    },
    {
        "id": "low_06", "scenario": None,
        "label": "Prime check", "type": "LOGIC", "mode": "full",
        "prompt": "Write a Python function that returns True if a given integer is prime, False otherwise.",
        "context": "Python 3.11",
    },
    {
        "id": "low_07", "scenario": None,
        "label": "Celsius to Fahrenheit", "type": "LOGIC", "mode": "full",
        "prompt": "Write a Python function that converts a temperature in Celsius to Fahrenheit.",
        "context": "Python 3.11",
    },
    {
        "id": "low_08", "scenario": None,
        "label": "Deduplicate list", "type": "LOGIC", "mode": "full",
        "prompt": "Write a Python function that removes duplicates from a list while preserving the original order.",
        "context": "Python 3.11",
    },
    {
        "id": "low_09", "scenario": None,
        "label": "Flatten one level", "type": "LOGIC", "mode": "full",
        "prompt": "Write a Python function that flattens a list of lists by one level and returns the result.",
        "context": "Python 3.11",
    },
    {
        "id": "low_10", "scenario": None,
        "label": "Binary search", "type": "LOGIC", "mode": "full",
        "prompt": "Write a Python function that performs binary search on a sorted list and returns the index of the target, or -1 if not found.",
        "context": "Python 3.11",
    },
]

# ---------------------------------------------------------------------------
# Prompts — HIGH (Scenarios A–D + additional calibration, detection test)
# ---------------------------------------------------------------------------
HIGH_PROMPTS = [
    # ── Scenarios A–D (explicitly labelled in the paper) ────────────────────
    {
        "id": "high_01", "scenario": "A",
        "label": "Stripe payment (Scenario A)", "type": "API", "mode": "full",
        "prompt": (
            "Write a Python function to charge a customer's card for $50 using "
            "the Stripe Python SDK in a FastAPI application. "
            "The customer_id is passed in as a parameter."
        ),
        "context": "Python 3.11, Stripe SDK installed",
    },
    {
        "id": "high_02", "scenario": "B",
        "label": "OpenAI SDK call (Scenario B)", "type": "API", "mode": "full",
        "prompt": "Write a Python function that calls GPT-4 with a user message and returns the response text.",
        "context": "Python 3.11, openai package installed",
    },
    {
        "id": "high_03", "scenario": "C",
        "label": "SQLAlchemy query (Scenario C)", "type": "API", "mode": "full",
        "prompt": (
            "Write a Python function that queries all active users from a "
            "SQLAlchemy database session and returns them as a list. "
            "The codebase has been around for a few years."
        ),
        "context": "Python 3.11, SQLAlchemy installed",
    },
    {
        "id": "high_04", "scenario": "D",
        "label": "FastAPI async HTTP (Scenario D)", "type": "API", "mode": "full",
        "prompt": (
            "Write a FastAPI endpoint that accepts a city name as a query parameter, "
            "fetches the current weather for that city from an external weather API, "
            "and returns the result as JSON."
        ),
        "context": "Python 3.11, FastAPI installed, requests library used throughout the existing codebase",
    },
    # ── Additional HIGH prompts (high_05–20) — referenced in multi-model table ─
    {
        "id": "high_05", "scenario": None,
        "label": "Password hashing", "type": "API", "mode": "full",
        "prompt": "Write a Python function that securely hashes a password for storage in a database and a function to verify it.",
        "context": "Python 3.11, security-sensitive application",
    },
    {
        "id": "high_06", "scenario": None,
        "label": "JWT token creation", "type": "API", "mode": "full",
        "prompt": "Write a Python function that creates a signed JWT token for a user containing their user_id, with a 1-hour expiry.",
        "context": "Python 3.11, building an authentication service",
    },
    {
        "id": "high_07", "scenario": None,
        "label": "Current UTC datetime", "type": "API", "mode": "full",
        "prompt": "Write a Python function that returns the current UTC time as a timezone-aware datetime object for storage in a database.",
        "context": "Python 3.11",
    },
    {
        "id": "high_08", "scenario": None,
        "label": "HTTP GET to JSON", "type": "API", "mode": "full",
        "prompt": "Write a Python function that makes an HTTP GET request to a given URL and returns the response body as a parsed JSON dict.",
        "context": "Python 3.11",
    },
    {
        "id": "high_09", "scenario": None,
        "label": "Redis cache get", "type": "API", "mode": "full",
        "prompt": "Write a Python function that retrieves a cached value from Redis by key, returning None if the key does not exist.",
        "context": "Python 3.11, Redis server running, async FastAPI application",
    },
    {
        "id": "high_10", "scenario": None,
        "label": "PostgreSQL query", "type": "API", "mode": "full",
        "prompt": "Write a Python function that connects to a PostgreSQL database and retrieves all rows from a 'users' table.",
        "context": "Python 3.11",
    },
    {
        "id": "high_11", "scenario": None,
        "label": "Pydantic model serialization", "type": "API", "mode": "full",
        "prompt": (
            "Write a Python function that takes a Pydantic model instance representing "
            "a user (with id, name, and email fields) and returns it as a plain dictionary."
        ),
        "context": "Python 3.11, Pydantic installed",
    },
    {
        "id": "high_12", "scenario": None,
        "label": "pandas DataFrame concat", "type": "API", "mode": "full",
        "prompt": (
            "Write a Python function that takes a list of pandas DataFrames and "
            "concatenates them into a single DataFrame, dropping duplicate rows."
        ),
        "context": "Python 3.11, pandas installed, existing codebase started 3 years ago",
    },
    {
        "id": "high_13", "scenario": None,
        "label": "boto3 S3 upload", "type": "API", "mode": "full",
        "prompt": (
            "Write a Python function that uploads a local file to an S3 bucket "
            "using boto3 and returns the S3 object key."
        ),
        "context": "Python 3.11, boto3 installed, AWS credentials configured",
    },
    {
        "id": "high_14", "scenario": None,
        "label": "Celery async task", "type": "API", "mode": "full",
        "prompt": (
            "Write a Celery task that sends a welcome email to a new user. "
            "The task should accept a user_id and email address as arguments."
        ),
        "context": "Python 3.11, Celery with Redis broker, Django project",
    },
    {
        "id": "high_15", "scenario": None,
        "label": "Token expiry datetime", "type": "API", "mode": "full",
        "prompt": (
            "Write a Python function that returns a datetime object representing "
            "24 hours from now, suitable for use as a JWT token expiration time."
        ),
        "context": "Python 3.11, building an authentication service",
    },
    {
        "id": "high_16", "scenario": None,
        "label": "Pillow image resize", "type": "API", "mode": "full",
        "prompt": (
            "Write a Python function that resizes an image to a maximum width of "
            "800 pixels while preserving aspect ratio, using the Pillow library."
        ),
        "context": "Python 3.11, Pillow installed",
    },
    {
        "id": "high_17", "scenario": None,
        "label": "Django REST list view", "type": "API", "mode": "full",
        "prompt": (
            "Write a Django REST Framework view that returns a paginated list of "
            "all active Product objects, supporting filtering by category."
        ),
        "context": "Python 3.11, Django 4.x, Django REST Framework installed",
    },
    {
        "id": "high_18", "scenario": None,
        "label": "Async HTTP GET request", "type": "API", "mode": "full",
        "prompt": (
            "Write an async Python function that makes an HTTP GET request to a "
            "given URL and returns the response body as a parsed JSON dict."
        ),
        "context": "Python 3.11, async FastAPI application",
    },
    {
        "id": "high_19", "scenario": None,
        "label": "Schema validation", "type": "API", "mode": "full",
        "prompt": (
            "Write a Python function that validates and deserializes incoming JSON "
            "request data for a user registration endpoint, checking that name, "
            "email, and password fields are present and correctly typed."
        ),
        "context": "Python 3.11, Flask application, no validation library committed yet",
    },
    {
        "id": "high_20", "scenario": None,
        "label": "Parametrized test", "type": "API", "mode": "full",
        "prompt": (
            "Write a parametrized test for a function that converts temperatures "
            "between Celsius and Fahrenheit, covering at least five cases including "
            "freezing point, boiling point, and body temperature."
        ),
        "context": "Python 3.11, existing test suite in the project",
    },
]

# ---------------------------------------------------------------------------
# Prompts — Scenario E: 6-function auth module (body mode)
# Each function is run independently; the shared context describes the module.
# This surfaces first-function library uncertainty for all 6 decision points.
# Note: the paper's "uncertainty collapse" (later functions showing ε=0.000)
# was observed in single-prompt multi-function generation; running functions
# separately here means each faces the full library-choice uncertainty.
# ---------------------------------------------------------------------------
_SCE_E_CTX = "from sqlalchemy.orm import Session\nSECRET_KEY = 'your-secret-key'  # loaded from env"

SCENARIO_E_PROMPTS = [
    {
        "id": "sce_e1", "scenario": "E",
        "label": "Auth module — register_user (Scenario E)",
        "type": "API", "mode": "body",
        "context": _SCE_E_CTX,
        "signature": dedent("""\
            def register_user(session: Session, email: str, password: str):
                \"\"\"Create a new user account. Hash the password and persist the user record.
                Raise an exception if the email is already registered. Return the user.\"\"\"
        """),
    },
    {
        "id": "sce_e2", "scenario": "E",
        "label": "Auth module — login (Scenario E)",
        "type": "API", "mode": "body",
        "context": _SCE_E_CTX,
        "signature": dedent("""\
            def login(session: Session, email: str, password: str) -> str:
                \"\"\"Verify credentials. Return a signed JWT access token (1-hour expiry).
                Raise an exception if credentials are invalid.\"\"\"
        """),
    },
    {
        "id": "sce_e3", "scenario": "E",
        "label": "Auth module — verify_token (Scenario E)",
        "type": "API", "mode": "body",
        "context": _SCE_E_CTX,
        "signature": dedent("""\
            def verify_token(token: str) -> dict | None:
                \"\"\"Decode and validate a JWT access token.
                Return the payload dict on success, or None if invalid or expired.\"\"\"
        """),
    },
    {
        "id": "sce_e4", "scenario": "E",
        "label": "Auth module — refresh_token (Scenario E)",
        "type": "API", "mode": "body",
        "context": _SCE_E_CTX,
        "signature": dedent("""\
            def refresh_token(token: str) -> str:
                \"\"\"Validate the given JWT and issue a new token with refreshed 1-hour expiry.
                Raise an exception if the token is invalid or expired.\"\"\"
        """),
    },
    {
        "id": "sce_e5", "scenario": "E",
        "label": "Auth module — request_password_reset (Scenario E)",
        "type": "API", "mode": "body",
        "context": _SCE_E_CTX,
        "signature": dedent("""\
            def request_password_reset(session: Session, email: str) -> str:
                \"\"\"Generate a signed password reset token (1-hour expiry).
                Return empty string if email not found (prevent enumeration).\"\"\"
        """),
    },
    {
        "id": "sce_e6", "scenario": "E",
        "label": "Auth module — reset_password (Scenario E)",
        "type": "API", "mode": "body",
        "context": _SCE_E_CTX,
        "signature": dedent("""\
            def reset_password(session: Session, reset_token: str, new_password: str) -> bool:
                \"\"\"Validate the reset token, look up the user, and update their hashed password.
                Return True on success. Raise an exception on invalid token or missing user.\"\"\"
        """),
    },
]

PROMPTS = LOW_PROMPTS + HIGH_PROMPTS + SCENARIO_E_PROMPTS

# ---------------------------------------------------------------------------
# ε math (identical to benchmark_production.py)
# ---------------------------------------------------------------------------

def token_epsilon(top_logprobs: list) -> float:
    k = len(top_logprobs)
    if k <= 1:
        return 0.0
    probs = [math.exp(lp) for lp in top_logprobs]
    total = sum(probs)
    if total == 0:
        return 0.0
    probs = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return entropy / math.log(k)


def file_epsilon(token_data: list[dict]) -> tuple[float, str | None]:
    max_eps = 0.0
    trigger = None
    for td in token_data:
        raw = td["token"].strip()
        if not raw:
            continue
        eps = token_epsilon(td["top_logprobs"])
        if eps > DELTA and eps > max_eps:
            max_eps = eps
            trigger = td["token"]
    return max_eps, trigger


def eps_to_status(eps: float) -> str:
    if eps < 0.30:
        return "COMPLETE"
    elif eps < 0.65:
        return "FLAGGED"
    elif eps < 0.95:
        return "PAUSED"
    else:
        return "ABORTED"


def cascaded_epsilon(token_data: list[dict]) -> dict:
    eps_seq = [token_epsilon(t["top_logprobs"]) for t in token_data]
    n = len(eps_seq)
    peak_eps = max(eps_seq) if eps_seq else 0.0
    if peak_eps < 0.10:
        return {"cascaded_score": 0.0, "cascaded_status": "COMPLETE",
                "evidence": {"peak_eps": round(peak_eps, 4)}}
    if peak_eps < 0.30:
        return {"cascaded_score": round(peak_eps * 0.5, 4), "cascaded_status": "COMPLETE",
                "evidence": {"peak_eps": round(peak_eps, 4)}}
    peak_idx = eps_seq.index(peak_eps)
    cluster_count = sum(1 for e in eps_seq if e > 0.20)
    max_run = cur_run = 0
    for e in eps_seq:
        if e > 0.15:
            cur_run += 1; max_run = max(max_run, cur_run)
        else:
            cur_run = 0
    post_peak = eps_seq[peak_idx + 1: peak_idx + 3]
    drops_fast = (len(post_peak) >= 2 and all(e < 0.15 for e in post_peak))
    re_escalation = 0
    in_cluster = eps_seq[peak_idx] > 0.20
    gap_len = 0
    for i in range(peak_idx + 1, n):
        if in_cluster:
            if eps_seq[i] < 0.10:
                gap_len += 1
                if gap_len >= 3:
                    in_cluster = False; gap_len = 0
        else:
            if eps_seq[i] > 0.20:
                re_escalation += 1; in_cluster = True; gap_len = 0
    elev_fraction = cluster_count / n if n > 0 else 0.0
    if n <= 6 and re_escalation == 0:
        lone_spike_tier = 1; lone_spike_penalty = 0.70
    elif n <= 40 and re_escalation == 0 and 3 <= cluster_count <= 5 and max_run <= 1:
        lone_spike_tier = 2; lone_spike_penalty = 0.40
    else:
        lone_spike_tier = 0; lone_spike_penalty = 0.0
    score = peak_eps
    score += 0.10 * min(cluster_count / 12.0, 1.0)
    score += 0.10 * min(max_run / 4.0, 1.0)
    if drops_fast:
        score -= 0.15
    score += 0.05 * min(re_escalation, 3)
    score += 0.10 * min(elev_fraction / 0.15, 1.0)
    score -= lone_spike_penalty
    score = max(0.0, min(1.0, score))
    w_start = max(0, peak_idx - 5)
    w_end   = min(n, peak_idx + 13)
    window_eps = eps_seq[w_start:w_end]
    evidence = {
        "peak_eps": round(peak_eps, 4), "peak_idx": peak_idx,
        "cluster_count": cluster_count, "max_run": max_run,
        "drops_fast": drops_fast, "re_escalation": re_escalation,
        "elev_fraction": round(elev_fraction, 4),
        "lone_spike": lone_spike_tier > 0, "lone_spike_pen": round(lone_spike_penalty, 4),
        "window_eps": [round(e, 4) for e in window_eps], "total_tokens": n,
    }
    return {"cascaded_score": round(score, 4), "cascaded_status": eps_to_status(score),
            "evidence": evidence}


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(p: dict) -> str:
    if p.get("mode") == "body":
        ctx = p.get("context", "").strip()
        sig = p["signature"].rstrip()
        return (
            BODY_CONTEXT_HEADER
            + ("\nRelevant imports available:\n" + ctx + "\n\n" if ctx else "\n")
            + sig + "\n"
        )
    else:
        ctx = p.get("context", "").strip()
        return p["prompt"] + (f"\n\nContext: {ctx}" if ctx else "")


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_openai(client: OpenAI, prompt: str, model: str, mode: str = "full") -> tuple[list[dict], str]:
    system = SYSTEM_BODY if mode == "body" else SYSTEM_FULL
    max_tokens = 512 if mode == "body" else 1024
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=K,
    )
    lp = response.choices[0].logprobs
    tokens = []
    if lp.content:
        for tok in lp.content:
            top_lps  = [alt.logprob for alt in tok.top_logprobs]
            top_toks = [alt.token   for alt in tok.top_logprobs]
            tokens.append({"token": tok.token, "logprob": tok.logprob,
                           "top_logprobs": top_lps, "top_tokens": top_toks})
    elif lp.tokens and lp.token_logprobs:
        for tok, lp_val, top_dict in zip(lp.tokens, lp.token_logprobs, lp.top_logprobs or []):
            if top_dict:
                sorted_items = sorted(top_dict.items(), key=lambda x: x[1], reverse=True)
                top_toks = [k for k, _ in sorted_items]
                top_lps  = [v for _, v in sorted_items]
            else:
                top_toks = [tok]; top_lps = [lp_val]
            tokens.append({"token": tok, "logprob": lp_val,
                           "top_logprobs": top_lps, "top_tokens": top_toks})
    return tokens, (response.choices[0].message.content or "")


def get_together_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://api.together.xyz/v1")


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(client: OpenAI, model: str, provider: str = "openai") -> list[dict]:
    results = []
    api_count   = sum(1 for p in PROMPTS if p["type"] == "API")
    logic_count = sum(1 for p in PROMPTS if p["type"] == "LOGIC")
    print(f"\nScenario benchmark: {len(PROMPTS)} prompts "
          f"({api_count} API, {logic_count} LOGIC) | model={model} | provider={provider}")
    print("-" * 70)

    for i, p in enumerate(PROMPTS, 1):
        prompt_text = build_prompt(p)
        mode = p.get("mode", "full")
        try:
            token_data, generated = call_openai(client, prompt_text, model, mode=mode)
        except Exception as e:
            print(f"  [{i:02d}/{len(PROMPTS)}] ERROR {p['id']}: {e}")
            results.append({**p, "error": str(e), "epsilon": None, "status": "ERROR"})
            continue

        eps, trigger = file_epsilon(token_data)
        status = eps_to_status(eps)
        fired = status in ("FLAGGED", "PAUSED", "ABORTED")
        casc = cascaded_epsilon(token_data)
        casc_fired = casc["cascaded_score"] >= 0.30
        expected_fire = (p["type"] == "API")
        correct      = (fired == expected_fire)
        casc_correct = (casc_fired == expected_fire)
        label = ("TP" if fired and expected_fire else
                 "TN" if not fired and not expected_fire else
                 "FP" if fired and not expected_fire else "FN")
        casc_label = ("TP" if casc_fired and expected_fire else
                      "TN" if not casc_fired and not expected_fire else
                      "FP" if casc_fired and not expected_fire else "FN")

        eps_seq = [token_epsilon(t["top_logprobs"]) for t in token_data]
        sparse_tokens = []
        for j, (td, e) in enumerate(zip(token_data, eps_seq)):
            if e > 0.10:
                alts = list(zip(
                    td.get("top_tokens", [])[:5],
                    [round(math.exp(lp) * 100, 1) for lp in td["top_logprobs"][:5]],
                ))
                sparse_tokens.append({"idx": j, "token": td["token"],
                                      "eps": round(e, 4), "alts": alts})

        sce_marker = f"[{p['scenario']}]" if p.get("scenario") else "   "
        marker = "+" if correct else "!"
        trig_safe = trigger.encode("ascii", "replace").decode("ascii") if trigger else None
        print(f"  [{i:02d}/{len(PROMPTS)}] {marker} {sce_marker} [{label}] [{p['type']}] "
              f"{p['id']:<16} {p['label'][:30]:<30} eps={eps:.3f} {status:<10}"
              + f"  casc={casc['cascaded_score']:.3f} [{casc_label}]"
              + (f" << '{trig_safe}'" if trig_safe else ""))

        results.append({
            **{k: v for k, v in p.items()},
            "generated":        generated,
            "epsilon":          round(eps, 4),
            "status":           status,
            "fired":            fired,
            "trigger_token":    trigger,
            "correct":          correct,
            "label":            label,
            "token_count":      len(token_data),
            "eps_seq":          [round(e, 4) for e in eps_seq],
            "cascaded_score":   casc["cascaded_score"],
            "cascaded_status":  casc["cascaded_status"],
            "casc_fired":       casc_fired,
            "casc_correct":     casc_correct,
            "casc_label":       casc_label,
            "evidence":         casc["evidence"],
            "sparse_tokens":    sparse_tokens,
        })

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results: list[dict]) -> None:
    api_r   = [r for r in results if r["type"] == "API"   and r.get("epsilon") is not None]
    logic_r = [r for r in results if r["type"] == "LOGIC" and r.get("epsilon") is not None]
    sce_e   = [r for r in api_r   if r.get("scenario") == "E"]
    sce_ad  = [r for r in api_r   if r.get("scenario") in ("A", "B", "C", "D")]
    other   = [r for r in api_r   if not r.get("scenario")]

    def rates(subset):
        tp = sum(1 for r in subset if r["label"] == "TP")
        fn = sum(1 for r in subset if r["label"] == "FN")
        c_tp = sum(1 for r in subset if r.get("casc_label") == "TP")
        c_fn = sum(1 for r in subset if r.get("casc_label") == "FN")
        n = len(subset)
        return tp, fn, c_tp, c_fn, n

    fp  = sum(1 for r in logic_r if r["label"]           == "FP")
    c_fp = sum(1 for r in logic_r if r.get("casc_label") == "FP")
    n_logic = len(logic_r)

    print("\n" + "=" * 70)
    print("SCENARIO BENCHMARK SUMMARY")
    print("-" * 70)

    print(f"\n  LOGIC (LOW) — FP test: {n_logic} prompts")
    print(f"    Original  FP rate: {fp/n_logic*100:.0f}%  ({fp}/{n_logic})")
    print(f"    Cascaded  FP rate: {c_fp/n_logic*100:.0f}%  ({c_fp}/{n_logic})")

    for name, subset in [("Scenarios A–D", sce_ad), ("Scenario E (auth)", sce_e), ("Other HIGH", other)]:
        tp, fn, c_tp, c_fn, n = rates(subset)
        if n == 0:
            continue
        print(f"\n  {name} — {n} prompts")
        print(f"    Original  detection: {tp/n*100:.0f}%  ({tp}/{n} TP, {fn}/{n} FN)")
        print(f"    Cascaded  detection: {c_tp/n*100:.0f}%  ({c_tp}/{n} TP, {c_fn}/{n} FN)")

    # Scenario E per-function ε detail
    if sce_e:
        print(f"\n  Scenario E per-function eps:")
        for r in sce_e:
            print(f"    {r['label'][22:]:<35} eps={r['epsilon']:.3f}  "
                  f"casc={r['cascaded_score']:.3f}  [{r['casc_label']}]")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario benchmark (A–E + calibration)")
    parser.add_argument("--model",    default="gpt-4o")
    parser.add_argument("--provider", default="openai", choices=["openai", "together"])
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        for p in PROMPTS:
            print(f"\n{'='*60}")
            print(f"[{p['type']}][{p['mode']}] {p['id']} — {p['label']}")
            print(build_prompt(p)[:300])
        return

    if args.provider == "together":
        client = get_together_client()
    else:
        client = OpenAI()

    results = run_benchmark(client, args.model, args.provider)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_file = out_dir / f"scenarios_{safe_model}.json"
    payload = {
        "meta": {
            "benchmark": "scenarios",
            "model": args.model,
            "provider": args.provider,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_prompts": len(PROMPTS),
            "n_api":   sum(1 for p in PROMPTS if p["type"] == "API"),
            "n_logic": sum(1 for p in PROMPTS if p["type"] == "LOGIC"),
        },
        "results": results,
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results -> {out_file}")
    print_summary(results)


if __name__ == "__main__":
    main()
