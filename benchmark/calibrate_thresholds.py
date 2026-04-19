#!/usr/bin/env python3
"""
Domain threshold calibration runner for Scenario E.

Runs the authentication module prompt N times and tracks the cold start → MAD
transition, printing domain status after each run. Use this to:

  1. Observe when the adaptive threshold first activates (MAD domain).
  2. Assess whether the MAD threshold is stable and reasonable.
  3. Decide whether knn_cold_start_tokens / knn_conformal_tokens need adjustment.

Usage:
    python calibrate_thresholds.py             # 12 runs (default)
    python calibrate_thresholds.py --runs 20
    python calibrate_thresholds.py --report    # just analyze existing log, no API calls
    python calibrate_thresholds.py --model gpt-4o-mini  # cheaper model for bulk runs
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import math
import statistics
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LOG_PATH = "epsilon_session.log"

DEMO_PROMPT = """\
Write a complete Python authentication module with the following six functions:

  register_user(db, username, password, email)
    Create a new user with a securely hashed password. Raise an appropriate
    error if the username or email already exists.

  login(db, username, password)
    Verify credentials against the stored hash. Return a signed JWT access
    token on success; raise an appropriate error on failure.

  verify_token(token)
    Decode and validate a JWT. Return the payload dict. Raise an error if
    the token is expired or its signature is invalid.

  refresh_token(token)
    If the token is valid and within the refresh window, issue a new token
    with a reset expiry. Raise an error if the token cannot be refreshed.

  request_password_reset(db, email)
    Generate a short-lived signed reset token and return it. Assume the
    caller handles delivery. Raise an error if the email is not found.

  reset_password(db, reset_token, new_password)
    Validate the reset token and update the user's stored password hash.
    Raise an error if the token is expired or already used.\
"""

DEMO_CONTEXT = (
    "Python 3.11, FastAPI backend, PostgreSQL database accessed via SQLAlchemy. "
    "No existing authentication library has been committed to in this codebase."
)

THRESHOLDS = {
    "threshold_soft":   0.30,
    "threshold_hard":   0.65,
    "threshold_abort":  0.95,
}


# ------------------------------------------------------------------ #
# Report mode: analyze existing log without making API calls
# ------------------------------------------------------------------ #

def _load_scenario_e_entries(log_path: str) -> list[dict]:
    """Return all log entries that match the Scenario E auth module prompt."""
    p = Path(log_path)
    if not p.exists():
        return []
    entries = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Scenario E entries contain "authentication module" in the prompt
        if "authentication" in e.get("prompt", "").lower():
            entries.append(e)
    return entries


def _report(log_path: str, cfg: dict) -> None:
    """Print analysis of existing Scenario E entries in the log."""
    all_entries = _load_scenario_e_entries(log_path)
    embedded    = [e for e in all_entries if e.get("embedding")]

    print()
    print("=" * 62)
    print(" Scenario E log analysis (authentication module)")
    print("=" * 62)
    print(f"  Total Scenario E entries : {len(all_entries)}")
    print(f"  With embeddings          : {len(embedded)}")
    print()

    if not embedded:
        print("  No embedded entries yet — run without --report to accumulate data.")
        return

    epsilons     = [e["epsilon_file"]  for e in embedded]
    code_tokens  = [e.get("n_code_tokens", 0) for e in embedded]
    total_tokens = sum(code_tokens)

    print(f"  n_code_tokens per run: {code_tokens}")
    print(f"  Total n_code_tokens  : {total_tokens}")
    print()

    # Simulate the domain switch as runs accumulated
    cold_start_tok = cfg["knn_cold_start_tokens"]
    conformal_tok  = cfg["knn_conformal_tokens"]
    min_n_mad      = cfg["knn_min_n_mad"]
    min_n_conformal= cfg["knn_min_n_conformal"]

    print(f"  Domain thresholds in use:")
    print(f"    cold_start_tokens = {cold_start_tok}  (min_n_mad = {min_n_mad})")
    print(f"    conformal_tokens  = {conformal_tok}  (min_n_conformal = {min_n_conformal})")
    print()
    print(f"  {'Run':>4}  {'eps':>6}  {'n_code':>6}  {'cum_tok':>7}  {'n_nbrs':>6}  {'domain':12}  {'threshold'}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*7}  {'-'*6}  {'-'*12}  {'-'*9}")

    mad_first = None
    for i in range(len(embedded)):
        # Use all entries up to and including i as the simulated neighborhood
        # (in production the K-NN would filter by similarity; here all are identical)
        nbr_eps    = [e["epsilon_file"] for e in embedded[:i+1]]
        nbr_tokens = sum(e.get("n_code_tokens", 0) for e in embedded[:i+1])
        n          = len(nbr_eps)
        eps_i      = embedded[i]["epsilon_file"]

        if nbr_tokens >= conformal_tok and n >= min_n_conformal:
            domain = "CONFORMAL"
            sorted_e = sorted(nbr_eps)
            idx      = min(int(0.95 * n), n - 1)
            thresh   = sorted_e[idx]
        elif nbr_tokens >= cold_start_tok and n >= min_n_mad:
            domain = "MAD"
            if mad_first is None:
                mad_first = i + 1
            median = statistics.median(nbr_eps)
            mad    = statistics.median([abs(e - median) for e in nbr_eps])
            thresh = None if mad == 0 else min(1.0, median + 3.0 * mad / 0.6745)
        else:
            domain = "COLD START"
            thresh = None

        thresh_str = f"{thresh:.4f}" if thresh is not None else "—"
        print(f"  {i+1:>4}  {eps_i:>6.4f}  {embedded[i].get('n_code_tokens',0):>6}  "
              f"{nbr_tokens:>7}  {n:>6}  {domain:12}  {thresh_str}")

    print()

    if mad_first is not None:
        print(f"  MAD domain first activated at run {mad_first}.")
    else:
        runs_needed_tok = math.ceil(cold_start_tok / max(1, statistics.mean(code_tokens) if code_tokens else 1))
        runs_needed     = max(min_n_mad, runs_needed_tok)
        print(f"  MAD domain not yet reached. Estimated {runs_needed} total runs needed "
              f"(have {len(embedded)}).")

    # Stability check: how much does the MAD threshold vary run-to-run?
    mad_thresholds = []
    for i in range(len(embedded)):
        nbr_eps    = [e["epsilon_file"] for e in embedded[:i+1]]
        nbr_tokens = sum(e.get("n_code_tokens", 0) for e in embedded[:i+1])
        n          = len(nbr_eps)
        if nbr_tokens >= cold_start_tok and n >= min_n_mad:
            median = statistics.median(nbr_eps)
            mad    = statistics.median([abs(e - median) for e in nbr_eps])
            if mad > 0:
                mad_thresholds.append(min(1.0, median + 3.0 * mad / 0.6745))

    if len(mad_thresholds) >= 2:
        print()
        print(f"  MAD threshold stability across {len(mad_thresholds)} MAD-domain runs:")
        print(f"    range  : {min(mad_thresholds):.4f} – {max(mad_thresholds):.4f}")
        print(f"    std dev: {statistics.stdev(mad_thresholds):.4f}")
        print(f"    mean   : {statistics.mean(mad_thresholds):.4f}")
        spread = max(mad_thresholds) - min(mad_thresholds)
        if spread < 0.10:
            verdict = "STABLE — threshold is consistent; thresholds look well-calibrated."
        elif spread < 0.20:
            verdict = "MODERATE — some run-to-run variation; may need more data."
        else:
            verdict = "UNSTABLE — high variance; consider raising knn_min_n_mad."
        print(f"    verdict: {verdict}")

    print()


# ------------------------------------------------------------------ #
# Run mode: call API N times and append to log
# ------------------------------------------------------------------ #

def _run_calibration(n_runs: int, model: str, log_path: str) -> None:
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai not installed.  pip install openai")
        sys.exit(1)

    from epsilon.core import EpsilonWrapper

    client  = OpenAI()
    wrapper = EpsilonWrapper(client, config=THRESHOLDS, log_path=log_path)
    cfg     = wrapper.config

    # How many Scenario E embedded entries exist already?
    prior = _load_scenario_e_entries(log_path)
    prior_emb = [e for e in prior if e.get("embedding")]
    print()
    print(f"  Existing Scenario E embedded entries: {len(prior_emb)}")
    print(f"  Running {n_runs} more runs. Model: {model}")
    print(f"  Log: {log_path}")
    print()
    print(f"  {'Run':>4}  {'eps':>6}  {'n_code':>6}  {'domain':12}  {'threshold':10}  status")
    print(f"  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*10}  {'-'*10}")

    total_cost = 0.0

    for i in range(n_runs):
        run_num = len(prior_emb) + i + 1
        result  = wrapper.generate_code(
            prompt=DEMO_PROMPT,
            context=DEMO_CONTEXT,
            model=model,
        )

        # Domain determination (mirrored from EpsilonResult fields)
        if result.ensemble_threshold is not None:
            if result.trigger == "ensemble":
                domain = "MAD/CONF*"  # ensemble threshold was active and triggered
            else:
                domain = "MAD/CONF"   # ensemble active but didn't trigger
        else:
            domain = "COLD START"

        thresh_str = (f"{result.ensemble_threshold:.4f}"
                      if result.ensemble_threshold is not None else "—")

        cost = (result.prompt_tokens * 2.50 + result.completion_tokens * 10.0) / 1_000_000
        total_cost += cost

        print(f"  {run_num:>4}  {result.epsilon_file:>6.4f}  {result.n_code_tokens:>6}  "
              f"{domain:12}  {thresh_str:10}  {result.status}  (${cost:.4f})")

    print()
    print(f"  Total cost: ${total_cost:.4f}")
    print()
    print("  Running report...")
    print()
    _report(log_path, cfg)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate domain thresholds by running Scenario E repeatedly."
    )
    parser.add_argument("--runs",   type=int, default=12,
                        help="Number of Scenario E runs to execute (default: 12)")
    parser.add_argument("--model",  default="gpt-4o",
                        help="OpenAI model (default: gpt-4o)")
    parser.add_argument("--report", action="store_true",
                        help="Print analysis of existing log only — no API calls")
    parser.add_argument("--log",    default=LOG_PATH,
                        help=f"Path to JSONL log file (default: {LOG_PATH})")
    args = parser.parse_args()

    if args.report:
        # Reconstruct a minimal config for the report
        cfg = {
            "knn_cold_start_tokens": 500,
            "knn_conformal_tokens":  3000,
            "knn_min_n_mad":         5,
            "knn_min_n_conformal":   20,
        }
        _report(args.log, cfg)
    else:
        _run_calibration(args.runs, args.model, args.log)


if __name__ == "__main__":
    main()
