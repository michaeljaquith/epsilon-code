#!/usr/bin/env python3
"""
Filtering-Stage Replay Analysis
================================
Computes HIGH detection rate and LOW false-positive rate at each successive
filtering stage, using per-token provenance data stored by benchmark_calibration.py
--store-tokens.

Stages (matching Table tab:filtering in the paper):

  Stage 1 — Naive: all tokens, no filtering
  Stage 2 — Noise: whitespace / comment / backtick fence tokens removed
  Stage 3 — Names: + naming-convention system prompt  (separate --no-naming run)
  Stage 4 — AST:   + AST declaration filter
  Stage 5 — Fence: + fence-format filter (full current config)

Stages 1–2 come from the --no-naming benchmark run.
Stages 3–5 come from the standard (naming-conventions) benchmark run.

Usage:
    python analyze_filtering_stages.py \\
        --with-naming  benchmark_gpt-4o_staged.json \\
        --no-naming    benchmark_gpt-4o_nonaming_staged.json
"""
import argparse
import json
from pathlib import Path

THRESHOLD = 0.30   # flag threshold — matches paper


def max_eps_at_stage(token_data: list, stage: str) -> float:
    """Return the max ε over tokens included at a given filtering stage.

    stage values:
      "raw"   — all tokens
      "ws"    — exclude whitespace/comment/fence noise (is_noise_ws)
      "ast"   — exclude noise + AST declarations (ws + dcl)
      "fence" — exclude noise + AST + fence-format (ws + dcl + fmt)  [current]
    """
    candidates = []
    for t in token_data:
        if stage == "raw":
            pass  # all tokens
        elif stage == "ws":
            if t["ws"]:
                continue
        elif stage == "ast":
            if t["ws"] or t["dcl"]:
                continue
        elif stage == "fence":
            if t["ws"] or t["dcl"] or t["fmt"]:
                continue
        candidates.append(t["eps"])
    above = [e for e in candidates if e > THRESHOLD]
    return max(above) if above else 0.0


def analyze(results: list, stage: str) -> tuple[float, float]:
    """Return (HIGH detection rate, LOW FP rate) for a given stage."""
    low  = [r for r in results if r["category"] == "LOW"]
    high = [r for r in results if r["category"] == "HIGH"]

    if not low or not high:
        return 0.0, 0.0
    if "token_data" not in results[0]:
        raise ValueError("No token_data in results — re-run benchmark with --store-tokens")

    fp  = sum(1 for r in low  if max_eps_at_stage(r["token_data"], stage) >= THRESHOLD)
    det = sum(1 for r in high if max_eps_at_stage(r["token_data"], stage) >= THRESHOLD)

    return det / len(high), fp / len(low)


def pct(x: float) -> str:
    return f"{x:.0%}"


def main():
    parser = argparse.ArgumentParser(description="Filtering-stage replay analysis")
    parser.add_argument("--with-naming", required=True,
                        help="Benchmark JSON run WITH naming-convention system prompt (--store-tokens)")
    parser.add_argument("--no-naming",   required=True,
                        help="Benchmark JSON run WITHOUT naming-convention system prompt (--no-naming --store-tokens)")
    args = parser.parse_args()

    with open(args.with_naming) as f:
        data_naming = json.load(f)
    with open(args.no_naming) as f:
        data_bare   = json.load(f)

    res_naming = data_naming["results"]
    res_bare   = data_bare["results"]

    n_high = len([r for r in res_naming if r["category"] == "HIGH"])
    n_low  = len([r for r in res_naming if r["category"] == "LOW"])

    print()
    print("Filtering-Stage Detection Analysis")
    print(f"Model: {data_naming['model']}   HIGH={n_high}   LOW={n_low}")
    print(f"Flag threshold: eps >= {THRESHOLD}")
    print()

    stages = [
        # (label, dataset, stage_key)
        ("1 — Naive (no filtering)",             res_bare,   "raw"),
        ("2 — + Whitespace/comment filter",       res_bare,   "ws"),
        ("3 — + Naming-convention system prompt", res_naming, "ws"),
        ("4 — + AST declaration filter",          res_naming, "ast"),
        ("5 — + Fence-format filter  [current]",  res_naming, "fence"),
    ]

    # Header
    col = 44
    print(f"  {'Stage':<{col}}  {'HIGH det':>8}  {'LOW FP':>6}")
    print(f"  {'-'*col}  {'--------':>8}  {'------':>6}")

    for label, dataset, stage in stages:
        det, fp = analyze(dataset, stage)
        print(f"  {label:<{col}}  {pct(det):>8}  {pct(fp):>6}")

    print()
    print("Notes:")
    print("  Stages 1–2 use the bare system prompt (no naming conventions).")
    print("  Stages 3–5 use the full system prompt (naming conventions enabled).")
    print("  Stage 5 == current production configuration.")
    print()

    # Also print raw counts for the paper
    print("Raw counts for LaTeX table:")
    for label, dataset, stage in stages:
        det_raw = sum(1 for r in dataset if r["category"] == "HIGH"
                      and max_eps_at_stage(r["token_data"], stage) >= THRESHOLD)
        fp_raw  = sum(1 for r in dataset if r["category"] == "LOW"
                      and max_eps_at_stage(r["token_data"], stage) >= THRESHOLD)
        n_h = len([r for r in dataset if r["category"] == "HIGH"])
        n_l = len([r for r in dataset if r["category"] == "LOW"])
        print(f"  {label}: HIGH {det_raw}/{n_h}  LOW {fp_raw}/{n_l}")
    print()


if __name__ == "__main__":
    main()
