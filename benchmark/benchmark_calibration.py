#!/usr/bin/env python3
"""
Calibration Benchmark — ε false positive / detection rate
==========================================================
Runs two prompt sets against the ε library:

  LOW  (10 prompts) — simple, unambiguous algorithms
       ε should stay COMPLETE. Any fire = false positive.

  HIGH (20 prompts) — version-split APIs, context-dependent patterns
       ε should fire FLAGGED or PAUSED. Silence = missed detection.

Outputs per-prompt result live, then a summary table with:
  - False positive rate  (LOW set — fires that shouldn't have)
  - Detection rate       (HIGH set — fires that should have)
  - Baseline comparison  (min/mean logprob, top-2 gap)
  - Aggregation ablation (max vs mean ε)

Results saved to benchmark_<model>.json.

Usage:
    python benchmark_calibration.py
    python benchmark_calibration.py --model gpt-4o-mini
    python benchmark_calibration.py --model gpt-4-turbo
    python benchmark_calibration.py --no-color
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Load .env from project root (two levels up) — same pattern as benchmark_scenarios.py
_env = Path(__file__).parent.parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.  pip install openai")
    sys.exit(1)

from epsilon.core import EpsilonWrapper

# ------------------------------------------------------------------ #
# Prompt sets
# ------------------------------------------------------------------ #

LOW_PROMPTS = [
    {
        "id": "low_01",
        "label": "Sort integers",
        "prompt": "Write a Python function that takes a list of integers and returns it sorted in ascending order.",
        "context": "Python 3.11",
    },
    {
        "id": "low_02",
        "label": "Palindrome check",
        "prompt": "Write a Python function that returns True if a given string is a palindrome, False otherwise. Ignore case.",
        "context": "Python 3.11",
    },
    {
        "id": "low_03",
        "label": "Fibonacci (nth)",
        "prompt": "Write a Python function that returns the nth Fibonacci number. n is a non-negative integer.",
        "context": "Python 3.11",
    },
    {
        "id": "low_04",
        "label": "List maximum",
        "prompt": "Write a Python function that returns the maximum value in a list of numbers without using the built-in max().",
        "context": "Python 3.11",
    },
    {
        "id": "low_05",
        "label": "Word frequency",
        "prompt": "Write a Python function that takes a string and returns a dictionary mapping each word to its frequency.",
        "context": "Python 3.11",
    },
    {
        "id": "low_06",
        "label": "Prime check",
        "prompt": "Write a Python function that returns True if a given integer is prime, False otherwise.",
        "context": "Python 3.11",
    },
    {
        "id": "low_07",
        "label": "Celsius to Fahrenheit",
        "prompt": "Write a Python function that converts a temperature in Celsius to Fahrenheit.",
        "context": "Python 3.11",
    },
    {
        "id": "low_08",
        "label": "Deduplicate list",
        "prompt": "Write a Python function that removes duplicates from a list while preserving the original order.",
        "context": "Python 3.11",
    },
    {
        "id": "low_09",
        "label": "Flatten one level",
        "prompt": "Write a Python function that flattens a list of lists by one level and returns the result.",
        "context": "Python 3.11",
    },
    {
        "id": "low_10",
        "label": "Binary search",
        "prompt": "Write a Python function that performs binary search on a sorted list and returns the index of the target, or -1 if not found.",
        "context": "Python 3.11",
    },
]

HIGH_PROMPTS = [
    {
        "id": "high_01",
        "label": "Stripe payment (Scenario A)",
        "prompt": (
            "Write a Python function to charge a customer's card for $50 using "
            "the Stripe Python SDK in a FastAPI application. The customer_id is "
            "passed in as a parameter."
        ),
        "context": "Python 3.11, Stripe SDK installed",
    },
    {
        "id": "high_02",
        "label": "OpenAI SDK call (Scenario B)",
        "prompt": "Write a Python function that calls GPT-4 with a user message and returns the response text.",
        "context": "Python 3.11, openai package installed",
    },
    {
        "id": "high_03",
        "label": "SQLAlchemy query (Scenario C)",
        "prompt": (
            "Write a Python function that queries all active users from a "
            "SQLAlchemy database session and returns them as a list. "
            "The codebase has been around for a few years."
        ),
        "context": "Python 3.11, SQLAlchemy installed",
    },
    {
        "id": "high_04",
        "label": "FastAPI async HTTP (Scenario D)",
        "prompt": (
            "Write a FastAPI endpoint that accepts a city name as a query parameter, "
            "fetches the current weather for that city from an external weather API, "
            "and returns the result as JSON."
        ),
        "context": "Python 3.11, FastAPI installed, requests library used throughout the existing codebase",
    },
    {
        "id": "high_05",
        "label": "Password hashing",
        "prompt": "Write a Python function that securely hashes a password for storage in a database and a function to verify it.",
        "context": "Python 3.11, security-sensitive application",
    },
    {
        "id": "high_06",
        "label": "JWT token creation",
        "prompt": "Write a Python function that creates a signed JWT token for a user containing their user_id, with a 1-hour expiry.",
        "context": "Python 3.11, building an authentication service",
    },
    {
        "id": "high_07",
        "label": "Current UTC datetime",
        "prompt": "Write a Python function that returns the current UTC time as a timezone-aware datetime object for storage in a database.",
        "context": "Python 3.11",
    },
    {
        "id": "high_08",
        "label": "HTTP GET to JSON",
        "prompt": "Write a Python function that makes an HTTP GET request to a given URL and returns the response body as a parsed JSON dict.",
        "context": "Python 3.11",
    },
    {
        "id": "high_09",
        "label": "Redis cache get",
        "prompt": "Write a Python function that retrieves a cached value from Redis by key, returning None if the key does not exist.",
        "context": "Python 3.11, Redis server running, async FastAPI application",
    },
    {
        "id": "high_10",
        "label": "PostgreSQL query",
        "prompt": "Write a Python function that connects to a PostgreSQL database and retrieves all rows from a 'users' table.",
        "context": "Python 3.11",
    },
    {
        "id": "high_11",
        "label": "Pydantic model serialization",
        "prompt": (
            "Write a Python function that takes a Pydantic model instance representing "
            "a user (with id, name, and email fields) and returns it as a plain dictionary."
        ),
        "context": "Python 3.11, Pydantic installed",
    },
    {
        "id": "high_12",
        "label": "pandas DataFrame concat",
        "prompt": (
            "Write a Python function that takes a list of pandas DataFrames and "
            "concatenates them into a single DataFrame, dropping duplicate rows."
        ),
        "context": "Python 3.11, pandas installed, existing codebase started 3 years ago",
    },
    {
        "id": "high_13",
        "label": "boto3 S3 upload",
        "prompt": (
            "Write a Python function that uploads a local file to an S3 bucket "
            "using boto3 and returns the S3 object key."
        ),
        "context": "Python 3.11, boto3 installed, AWS credentials configured",
    },
    {
        "id": "high_14",
        "label": "Celery async task",
        "prompt": (
            "Write a Celery task that sends a welcome email to a new user. "
            "The task should accept a user_id and email address as arguments."
        ),
        "context": "Python 3.11, Celery with Redis broker, Django project",
    },
    {
        "id": "high_15",
        "label": "Token expiry datetime",
        "prompt": (
            "Write a Python function that returns a datetime object representing "
            "24 hours from now, suitable for use as a JWT token expiration time."
        ),
        "context": "Python 3.11, building an authentication service",
    },
    {
        "id": "high_16",
        "label": "Pillow image resize",
        "prompt": (
            "Write a Python function that resizes an image to a maximum width of "
            "800 pixels while preserving aspect ratio, using the Pillow library."
        ),
        "context": "Python 3.11, Pillow installed",
    },
    {
        "id": "high_17",
        "label": "Django REST list view",
        "prompt": (
            "Write a Django REST Framework view that returns a paginated list of "
            "all active Product objects, supporting filtering by category."
        ),
        "context": "Python 3.11, Django 4.x, Django REST Framework installed",
    },
    {
        "id": "high_18",
        "label": "Async HTTP GET request",
        "prompt": (
            "Write an async Python function that makes an HTTP GET request to a "
            "given URL and returns the response body as a parsed JSON dict."
        ),
        "context": "Python 3.11, async FastAPI application",
    },
    {
        "id": "high_19",
        "label": "Schema validation",
        "prompt": (
            "Write a Python function that validates and deserializes incoming JSON "
            "request data for a user registration endpoint, checking that name, "
            "email, and password fields are present and correctly typed."
        ),
        "context": "Python 3.11, Flask application, no validation library committed yet",
    },
    {
        "id": "high_20",
        "label": "Parametrized test",
        "prompt": (
            "Write a parametrized test for a function that converts temperatures "
            "between Celsius and Fahrenheit, covering at least five cases including "
            "freezing point, boiling point, and body temperature."
        ),
        "context": "Python 3.11, existing test suite in the project",
    },
]

THRESHOLDS = {
    "threshold_soft":    0.30,
    "threshold_hard":    0.65,
    "threshold_abort":   0.95,
    "accumulation_floor": 0.30,
    "peak_min_epsilon":  0.40,
    "peak_max_count":    3,    # keep output concise during benchmark
}

FIRES = {"FLAGGED", "PAUSED", "ABORTED"}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def print_separator(label: str = "", width: int = 62) -> None:
    if label:
        pad = max(0, (width - len(label) - 2) // 2)
        print(f"{'━' * pad} {label} {'━' * (width - pad - len(label) - 2)}")
    else:
        print("━" * width)


def status_marker(status: str) -> str:
    return {"COMPLETE": "✓", "FLAGGED": "⚠", "PAUSED": "⚠⚠", "ABORTED": "✖"}.get(status, "?")


def run_prompt(wrapper: EpsilonWrapper, entry: dict, model: str, idx: int, total: int) -> dict:
    label   = entry["label"]
    cat     = "LOW " if entry["id"].startswith("low") else "HIGH"
    print(f"  [{idx:02d}/{total}] {cat} — {label} ... ", end="", flush=True)

    result  = wrapper.generate_code(
        prompt=entry["prompt"],
        context=entry.get("context", ""),
        model=model,
    )

    marker  = status_marker(result.status)
    fired   = result.status in FIRES
    top_flag = result.flags[0][:55] if result.flags else "—"

    print(f"{marker} {result.status:<8}  ε={result.epsilon_file:.3f}  {top_flag}")

    # ---- Baseline metrics (for reviewer comparison) ----
    all_logprobs = [te.logprob for te in result.token_epsilons]
    min_logprob  = round(min(all_logprobs), 4)  if all_logprobs else None
    mean_logprob = round(sum(all_logprobs) / len(all_logprobs), 4) if all_logprobs else None

    top2_gap = None
    if result.peak_tokens:
        alts = result.peak_tokens[0].top_alternatives
        if len(alts) >= 2:
            top2_gap = round(alts[0][1] - alts[1][1], 4)

    # ---- Mean aggregation (ablation vs max) ----
    floor = THRESHOLDS["accumulation_floor"]
    mean_cands   = [te.epsilon for te in result.token_epsilons
                    if te.is_code_token and te.epsilon > floor]
    epsilon_mean = round(sum(mean_cands) / len(mean_cands), 4) if mean_cands else 0.0
    fired_mean   = epsilon_mean >= THRESHOLDS["threshold_soft"]

    return {
        "id":       entry["id"],
        "label":    label,
        "category": cat.strip(),
        "prompt":   entry["prompt"][:80],
        "context":  entry.get("context", ""),
        "model":    result.model,
        "epsilon":  round(result.epsilon_file, 4),
        "epsilon_mean": epsilon_mean,
        "status":   result.status,
        "fired":    fired,
        "fired_mean": fired_mean,
        "flags":    result.flags[:3],
        "min_logprob":  min_logprob,
        "mean_logprob": mean_logprob,
        "top2_gap":     top2_gap,
        "prompt_tokens":     result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def get_together_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://api.together.xyz/v1")


def main():
    parser = argparse.ArgumentParser(description="ε calibration benchmark")
    parser.add_argument("--model",    default="gpt-4o")
    parser.add_argument("--provider", default="openai", choices=["openai", "together"])
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--low-only", action="store_true", help="Run LOW set only")
    parser.add_argument("--high-only", action="store_true", help="Run HIGH set only")
    args = parser.parse_args()

    if args.provider == "together":
        client = get_together_client()
    else:
        client = OpenAI()
    wrapper = EpsilonWrapper(client, config=THRESHOLDS)

    if args.high_only:
        prompts = HIGH_PROMPTS
    elif args.low_only:
        prompts = LOW_PROMPTS
    else:
        prompts = LOW_PROMPTS + HIGH_PROMPTS

    total = len(prompts)

    print()
    print_separator("ε CALIBRATION BENCHMARK")
    print()
    print(f"  Model:      {args.model}")
    print(f"  Prompts:    {total}  ({sum(1 for p in prompts if p['id'].startswith('low'))} low / "
          f"{sum(1 for p in prompts if p['id'].startswith('high'))} high)")
    print(f"  Thresholds: soft={THRESHOLDS['threshold_soft']} | "
          f"hard={THRESHOLDS['threshold_hard']} | "
          f"abort={THRESHOLDS['threshold_abort']}")
    print()
    print_separator()
    print()

    results = []
    total_cost_in  = 0
    total_cost_out = 0

    for i, entry in enumerate(prompts, 1):
        rec = run_prompt(wrapper, entry, args.model, i, total)
        results.append(rec)
        total_cost_in  += rec["prompt_tokens"]
        total_cost_out += rec["completion_tokens"]

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print()
    print_separator("RESULTS SUMMARY")
    print()

    low_results  = [r for r in results if r["category"] == "LOW"]
    high_results = [r for r in results if r["category"] == "HIGH"]

    # False positive rate — LOW prompts that fired (max aggregation)
    false_positives = [r for r in low_results if r["fired"]]
    fp_rate = len(false_positives) / len(low_results) if low_results else 0.0

    # Detection rate — HIGH prompts that fired (max aggregation)
    detections = [r for r in high_results if r["fired"]]
    det_rate = len(detections) / len(high_results) if high_results else 0.0

    # Ablation: mean aggregation rates
    fp_mean  = sum(1 for r in low_results  if r["fired_mean"]) / len(low_results)  if low_results  else 0.0
    det_mean = sum(1 for r in high_results if r["fired_mean"]) / len(high_results) if high_results else 0.0

    # Per-prompt table
    col_w = 32
    print(f"  {'Label':<{col_w}}  {'Cat':4}  {'ε':>6}  {'Status':<8}  {'Note'}")
    print(f"  {'-'*col_w}  {'----':4}  {'------':>6}  {'--------':<8}  ------")
    for r in results:
        category = r["category"]
        note = ""
        if category == "LOW"  and r["fired"]:  note = "← FALSE POSITIVE"
        if category == "HIGH" and not r["fired"]: note = "← MISSED"
        print(f"  {r['label']:<{col_w}}  {category:<4}  {r['epsilon']:>6.3f}  {r['status']:<8}  {note}")

    print()
    print_separator("METRICS")
    print()

    if low_results:
        low_epsilons = [r["epsilon"] for r in low_results]
        fp_mean_count = sum(1 for r in low_results if r["fired_mean"])
        print(f"  LOW set   ({len(low_results)} prompts)")
        print(f"    FP rate (max agg)   : {fp_rate:.0%}  ({len(false_positives)}/{len(low_results)} fired)")
        print(f"    FP rate (mean agg)  : {fp_mean:.0%}  ({fp_mean_count}/{len(low_results)} fired)")
        print(f"    ε range             : {min(low_epsilons):.3f} – {max(low_epsilons):.3f}")
        print(f"    ε mean (file)       : {sum(low_epsilons)/len(low_epsilons):.3f}")
        if false_positives:
            print(f"    False positives     : {', '.join(r['label'] for r in false_positives)}")
        print()

    if high_results:
        high_epsilons = [r["epsilon"] for r in high_results]
        missed = [r for r in high_results if not r["fired"]]
        missed_mean = [r for r in high_results if not r["fired_mean"]]
        print(f"  HIGH set  ({len(high_results)} prompts)")
        print(f"    Detection (max agg) : {det_rate:.0%}  ({len(detections)}/{len(high_results)} fired)")
        print(f"    Detection (mean agg): {det_mean:.0%}  ({sum(1 for r in high_results if r['fired_mean'])}/{len(high_results)} fired)")
        print(f"    ε range             : {min(high_epsilons):.3f} – {max(high_epsilons):.3f}")
        print(f"    ε mean (file)       : {sum(high_epsilons)/len(high_epsilons):.3f}")
        if missed:
            print(f"    Missed (max)        : {', '.join(r['label'] for r in missed)}")
        if missed_mean:
            print(f"    Missed (mean)       : {', '.join(r['label'] for r in missed_mean)}")
        print()

    cost_usd = (total_cost_in * 2.50 + total_cost_out * 10.0) / 1_000_000
    print(f"  Total cost: ${cost_usd:.4f}  "
          f"({total_cost_in} in + {total_cost_out} out tokens)")
    print()

    # Save results
    output = {
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
        "model":            args.model,
        "thresholds":       THRESHOLDS,
        "n_low":            len(low_results),
        "n_high":           len(high_results),
        "false_positive_rate_max":  round(fp_rate,   4),
        "false_positive_rate_mean": round(fp_mean,   4),
        "detection_rate_max":       round(det_rate,  4),
        "detection_rate_mean":      round(det_mean,  4),
        "results":          results,
    }
    model_slug = args.model.replace("/", "_")
    out_path = f"benchmark_{model_slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {out_path}")
    print()


if __name__ == "__main__":
    main()
