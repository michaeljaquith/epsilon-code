#!/usr/bin/env python3
"""
UQLM comparison benchmark
=========================
Runs the same 20-prompt benchmark (10 LOW + 10 HIGH) through UQLM's
BlackBoxUQ scorer and compares detection rate / false positive rate
against the ε system's results.

UQLM approach (Black-Box):
  - Generate N responses to the same prompt (default: 5)
  - Measure semantic consistency across responses
  - Confidence score: 0–1 (higher = more consistent = less uncertain)
  - "Fires" when confidence < threshold (default: 0.5)

ε system approach:
  - Single generation with logprobs
  - Token-level entropy, filtered for consequential decisions (not naming)
  - "Fires" when max code-token ε > soft threshold (0.30)

Key methodological difference: UQLM detects cross-generation inconsistency;
ε detects within-generation token-level uncertainty. UQLM is 5x more expensive
and provides no localization (which token, which line, which function).

Usage:
    python benchmark_uqlm.py
    python benchmark_uqlm.py --num-responses 3   # faster / cheaper
    python benchmark_uqlm.py --fire-threshold 0.4
    python benchmark_uqlm.py --high-only
    python benchmark_uqlm.py --skip-epsilon       # UQLM only (reads prior ε results)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ------------------------------------------------------------------ #
# Prompts — identical to benchmark_calibration.py
# ------------------------------------------------------------------ #

LOW_PROMPTS = [
    {"id": "low_01", "label": "Sort integers",
     "prompt": "Write a Python function that takes a list of integers and returns it sorted in ascending order.",
     "context": "Python 3.11"},
    {"id": "low_02", "label": "Palindrome check",
     "prompt": "Write a Python function that returns True if a given string is a palindrome, False otherwise. Ignore case.",
     "context": "Python 3.11"},
    {"id": "low_03", "label": "Fibonacci (nth)",
     "prompt": "Write a Python function that returns the nth Fibonacci number. n is a non-negative integer.",
     "context": "Python 3.11"},
    {"id": "low_04", "label": "List maximum",
     "prompt": "Write a Python function that returns the maximum value in a list of numbers without using the built-in max().",
     "context": "Python 3.11"},
    {"id": "low_05", "label": "Word frequency",
     "prompt": "Write a Python function that takes a string and returns a dictionary mapping each word to its frequency.",
     "context": "Python 3.11"},
    {"id": "low_06", "label": "Prime check",
     "prompt": "Write a Python function that returns True if a given integer is prime, False otherwise.",
     "context": "Python 3.11"},
    {"id": "low_07", "label": "Celsius to Fahrenheit",
     "prompt": "Write a Python function that converts a temperature in Celsius to Fahrenheit.",
     "context": "Python 3.11"},
    {"id": "low_08", "label": "Deduplicate list",
     "prompt": "Write a Python function that removes duplicates from a list while preserving the original order.",
     "context": "Python 3.11"},
    {"id": "low_09", "label": "Flatten one level",
     "prompt": "Write a Python function that flattens a list of lists by one level and returns the result.",
     "context": "Python 3.11"},
    {"id": "low_10", "label": "Binary search",
     "prompt": "Write a Python function that performs binary search on a sorted list and returns the index of the target, or -1 if not found.",
     "context": "Python 3.11"},
]

HIGH_PROMPTS = [
    {"id": "high_01", "label": "Stripe payment (A)",
     "prompt": "Write a Python function to charge a customer's card for $50 using the Stripe Python SDK in a FastAPI application. The customer_id is passed in as a parameter.",
     "context": "Python 3.11, Stripe SDK installed"},
    {"id": "high_02", "label": "OpenAI SDK call (B)",
     "prompt": "Write a Python function that calls GPT-4 with a user message and returns the response text.",
     "context": "Python 3.11, openai package installed"},
    {"id": "high_03", "label": "SQLAlchemy query (C)",
     "prompt": "Write a Python function that queries all active users from a SQLAlchemy database session and returns them as a list. The codebase has been around for a few years.",
     "context": "Python 3.11, SQLAlchemy installed"},
    {"id": "high_04", "label": "FastAPI async HTTP (D)",
     "prompt": "Write a FastAPI endpoint that accepts a city name as a query parameter, fetches the current weather for that city from an external weather API, and returns the result as JSON.",
     "context": "Python 3.11, FastAPI installed, requests library used throughout the existing codebase"},
    {"id": "high_05", "label": "Password hashing",
     "prompt": "Write a Python function that securely hashes a password for storage in a database and a function to verify it.",
     "context": "Python 3.11, security-sensitive application"},
    {"id": "high_06", "label": "JWT token creation",
     "prompt": "Write a Python function that creates a signed JWT token for a user containing their user_id, with a 1-hour expiry.",
     "context": "Python 3.11, building an authentication service"},
    {"id": "high_07", "label": "Current UTC datetime",
     "prompt": "Write a Python function that returns the current UTC time as a timezone-aware datetime object for storage in a database.",
     "context": "Python 3.11"},
    {"id": "high_08", "label": "HTTP GET to JSON",
     "prompt": "Write a Python function that makes an HTTP GET request to a given URL and returns the response body as a parsed JSON dict.",
     "context": "Python 3.11"},
    {"id": "high_09", "label": "Redis cache get",
     "prompt": "Write a Python function that retrieves a cached value from Redis by key, returning None if the key does not exist.",
     "context": "Python 3.11, Redis server running, async FastAPI application"},
    {"id": "high_10", "label": "PostgreSQL query",
     "prompt": "Write a Python function that connects to a PostgreSQL database and retrieves all rows from a 'users' table.",
     "context": "Python 3.11"},
]

SYSTEM_PROMPT = (
    "You are an expert software engineer. "
    "Respond with raw Python code only — no markdown fences, "
    "no explanation, no prose. Output only the function(s) requested. "
    "Use concise conventional parameter names (n for integers, s for strings, "
    "lst for lists, d for dicts, f for floats). "
    "Name functions using the most direct verb-noun form from the prompt."
)


# ------------------------------------------------------------------ #
# UQLM runner
# ------------------------------------------------------------------ #

async def run_uqlm(prompts_list: list[dict], model: str, num_responses: int) -> list[dict]:
    """Run all prompts through UQLM BlackBoxUQ and return scored results."""
    try:
        from langchain_openai import ChatOpenAI
        from uqlm import BlackBoxUQ
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install dependencies: pip install uqlm langchain-openai")
        sys.exit(1)

    # Build full prompt strings (inject context into system-style prefix)
    formatted = []
    for entry in prompts_list:
        ctx = entry.get("context", "")
        full = f"Context: {ctx}\n\n{entry['prompt']}" if ctx else entry["prompt"]
        formatted.append(full)

    llm    = ChatOpenAI(model=model, temperature=0)
    scorer = BlackBoxUQ(llm=llm, scorers=["semantic_negentropy"])

    # generate_and_score accepts lists of LangChain messages — use that to
    # inject the system prompt, since system_prompt= is not a supported kwarg.
    from langchain_core.messages import SystemMessage, HumanMessage
    messages_list = [
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=p)]
        for p in formatted
    ]

    print(f"  Running UQLM (BlackBoxUQ / semantic_negentropy) ...")
    print(f"  {len(formatted)} prompts × {num_responses} responses = "
          f"{len(formatted) * num_responses} API calls")
    print()

    results_df = await scorer.generate_and_score(
        prompts=messages_list,
        num_responses=num_responses,
    )

    df = results_df.to_df()

    out = []
    for i, entry in enumerate(prompts_list):
        conf = float(df.iloc[i]["semantic_negentropy"])  # 0–1, higher = less uncertain
        out.append({
            "id":         entry["id"],
            "label":      entry["label"],
            "category":   "LOW" if entry["id"].startswith("low") else "HIGH",
            "confidence": round(conf, 4),
        })
    return out


# ------------------------------------------------------------------ #
# ε runner (reuses benchmark_calibration logic)
# ------------------------------------------------------------------ #

def run_epsilon(prompts_list: list[dict], model: str) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai not installed. pip install openai")
        sys.exit(1)

    from epsilon.core import EpsilonWrapper

    THRESHOLDS = {
        "threshold_soft":    0.30,
        "threshold_hard":    0.65,
        "threshold_abort":   0.95,
        "accumulation_floor": 0.30,
        "peak_min_epsilon":  0.40,
        "peak_max_count":    3,
    }

    client  = OpenAI()
    wrapper = EpsilonWrapper(client, config=THRESHOLDS)

    out = []
    fires = {"FLAGGED", "PAUSED", "ABORTED"}
    for i, entry in enumerate(prompts_list, 1):
        cat = "LOW" if entry["id"].startswith("low") else "HIGH"
        print(f"  [{i:02d}/{len(prompts_list)}] {cat} — {entry['label']} ... ", end="", flush=True)
        result = wrapper.generate_code(
            prompt=entry["prompt"],
            context=entry.get("context", ""),
            model=model,
        )
        fired = result.status in fires
        print(f"{result.status:<8}  ε={result.epsilon_file:.3f}")
        out.append({
            "id":       entry["id"],
            "label":    entry["label"],
            "category": cat,
            "epsilon":  round(result.epsilon_file, 4),
            "status":   result.status,
            "fired":    fired,
        })
    return out


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def print_sep(label: str = "", width: int = 62) -> None:
    if label:
        pad = max(0, (width - len(label) - 2) // 2)
        print(f"{'━' * pad} {label} {'━' * (width - pad - len(label) - 2)}")
    else:
        print("━" * width)


def main():
    parser = argparse.ArgumentParser(description="UQLM vs ε comparison benchmark")
    parser.add_argument("--model",          default="gpt-4o")
    parser.add_argument("--num-responses",  type=int, default=5,
                        help="UQLM: responses per prompt (default: 5, min: 3)")
    parser.add_argument("--fire-threshold", type=float, default=0.5,
                        help="UQLM confidence below which we count as 'fired' (default: 0.5)")
    parser.add_argument("--high-only",      action="store_true")
    parser.add_argument("--low-only",       action="store_true")
    parser.add_argument("--skip-epsilon",   action="store_true",
                        help="Skip ε run; load prior results from benchmark_calibration.json")
    parser.add_argument("--skip-uqlm",      action="store_true",
                        help="Skip UQLM run; load prior results from benchmark_uqlm.json if present")
    args = parser.parse_args()

    if args.high_only:
        prompts = HIGH_PROMPTS
    elif args.low_only:
        prompts = LOW_PROMPTS
    else:
        prompts = LOW_PROMPTS + HIGH_PROMPTS

    print()
    print_sep("UQLM vs ε COMPARISON BENCHMARK")
    print()
    print(f"  Model          : {args.model}")
    print(f"  Prompts        : {len(prompts)} "
          f"({sum(1 for p in prompts if p['id'].startswith('low'))} low / "
          f"{sum(1 for p in prompts if p['id'].startswith('high'))} high)")
    print(f"  UQLM responses : {args.num_responses} per prompt")
    print(f"  UQLM fires when: confidence < {args.fire_threshold}")
    print()

    # ---- ε results ----
    eps_run_cache = Path("benchmark_epsilon_run.json")
    if args.skip_epsilon and eps_run_cache.exists():
        print(f"Loading prior ε results from {eps_run_cache} ...")
        prior = json.loads(eps_run_cache.read_text(encoding="utf-8"))
        eps_results = [r for r in prior["epsilon_results"]
                       if any(p["id"] == r["id"] for p in prompts)]
        print(f"  Loaded {len(eps_results)} ε results.")
        print()
    else:
        print_sep("ε SYSTEM")
        print()
        eps_results = run_epsilon(prompts, args.model)
        print()

    # Persist ε results immediately so a UQLM crash doesn't lose the work
    Path("benchmark_epsilon_run.json").write_text(
        json.dumps({"epsilon_results": eps_results}, indent=2), encoding="utf-8"
    )

    # ---- UQLM results ----
    uqlm_cache = Path("benchmark_uqlm.json")
    if args.skip_uqlm and uqlm_cache.exists():
        print("Loading prior UQLM results from benchmark_uqlm.json ...")
        cached = json.loads(uqlm_cache.read_text(encoding="utf-8"))
        uqlm_results = [r for r in cached["uqlm_results"]
                        if any(p["id"] == r["id"] for p in prompts)]
        print(f"  Loaded {len(uqlm_results)} UQLM results.")
        print()
    else:
        print_sep("UQLM")
        print()
        uqlm_results = asyncio.run(run_uqlm(prompts, args.model, args.num_responses))

    # Tag UQLM fires
    fire_thr = args.fire_threshold
    for r in uqlm_results:
        r["fired"] = r["confidence"] < fire_thr

    # ---- Side-by-side comparison ----
    print()
    print_sep("SIDE-BY-SIDE RESULTS")
    print()

    # Merge by id
    eps_by_id  = {r["id"]: r for r in eps_results}
    uqlm_by_id = {r["id"]: r for r in uqlm_results}

    col_w = 28
    print(f"  {'Prompt':<{col_w}}  {'Cat':4}  {'ε-status':<8}  {'ε-fired':7}  {'UQLM-conf':9}  {'UQLM-fired':10}  {'Agree?'}")
    print(f"  {'-'*col_w}  {'----':4}  {'--------':<8}  {'-------':7}  {'---------':9}  {'----------':10}  {'------'}")

    agree = disagree = 0
    for p in prompts:
        pid  = p["id"]
        er   = eps_by_id.get(pid, {})
        ur   = uqlm_by_id.get(pid, {})
        cat  = er.get("category", "?")
        e_fired  = er.get("fired", None)
        u_fired  = ur.get("fired", None)
        e_status = er.get("status", "—")
        u_conf   = ur.get("confidence", float("nan"))
        agreed   = (e_fired == u_fired) if (e_fired is not None and u_fired is not None) else None
        if agreed is True:  agree    += 1
        if agreed is False: disagree += 1

        agree_str  = "✓" if agreed is True else ("✗" if agreed is False else "?")
        e_fire_str = ("YES" if e_fired else "no") if e_fired is not None else "—"
        u_fire_str = ("YES" if u_fired else "no") if u_fired is not None else "—"
        print(f"  {p['label']:<{col_w}}  {cat:<4}  {e_status:<8}  {e_fire_str:<7}  {u_conf:<9.4f}  {u_fire_str:<10}  {agree_str}")

    # ---- Metrics ----
    print()
    print_sep("METRICS")
    print()

    for label, cat in [("LOW (false positives)", "LOW"), ("HIGH (detections)", "HIGH")]:
        e_cat   = [r for r in eps_results  if r["category"] == cat]
        u_cat   = [r for r in uqlm_results if r["category"] == cat]
        e_fires = sum(1 for r in e_cat  if r["fired"])
        u_fires = sum(1 for r in u_cat  if r["fired"])
        n       = max(len(e_cat), len(u_cat), 1)
        print(f"  {label} ({n} prompts)")
        print(f"    ε system : {e_fires}/{n} = {e_fires/n:.0%}")
        print(f"    UQLM     : {u_fires}/{n} = {u_fires/n:.0%}")
        print()

    total = agree + disagree
    print(f"  Agreement: {agree}/{total} prompts ({agree/total:.0%})" if total else "  No agreement data.")
    print()

    # ---- Advantage summary ----
    print_sep("STRUCTURAL COMPARISON")
    print()
    print("  ε system:")
    print("    - Single API call per prompt")
    print("    - Token-level localization (which line, which function)")
    print("    - Zero added latency — logprobs from same generation")
    print("    - Pipeline-propagating (ε accumulates across runs)")
    print()
    print("  UQLM (BlackBoxUQ / semantic_negentropy):")
    print(f"    - {args.num_responses} API calls per prompt ({args.num_responses}x cost)")
    print("    - File-level score only — no localization")
    print("    - Post-hoc: requires N complete generations before scoring")
    print("    - Designed for factual hallucination, not API version decisions")
    print()

    # ---- Save ----
    output = {
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
        "model":         args.model,
        "num_responses": args.num_responses,
        "fire_threshold": fire_thr,
        "epsilon_results": eps_results,
        "uqlm_results":  uqlm_results,
    }
    Path("benchmark_uqlm.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print("  Results saved to: benchmark_uqlm.json")
    print()


if __name__ == "__main__":
    main()
