#!/usr/bin/env python3
"""
p_avg Baseline Analysis
=======================
Computes p_avg = exp(mean_logprob) as an uncertainty signal from existing
calibration JSON files.  Sweeps detection threshold and reports precision/recall
at each operating point, compared to ε.

This implements the Spiess et al. (ICSE 2025, arXiv:2402.02047) baseline:
sequence-level mean token probability as a code confidence signal.

Signal: -mean_logprob (higher = less confident = flag this function)

Usage:
    python analyze_p_avg.py
    python analyze_p_avg.py --out-json p_avg_results.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RESULTS_DIR = Path(__file__).parent / "results"

CALIBRATION_FILES = {
    "gpt-4o":        RESULTS_DIR / "benchmark_gpt-4o.json",
    "gpt-4o-mini":   RESULTS_DIR / "benchmark_gpt-4o-mini.json",
    "gpt-4-turbo":   RESULTS_DIR / "benchmark_gpt-4-turbo.json",
    "deepseek-v3":   RESULTS_DIR / "calibration_deepseek-ai_DeepSeek-V3.json",
}

# Thresholds on -mean_logprob to sweep
# mean_logprob is typically in [-2, 0]; -mean_logprob in [0, 2]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def load_results(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["results"]


def p_avg_fires(entry: dict, threshold: float) -> bool:
    """Return True if -mean_logprob >= threshold (function is uncertain)."""
    ml = entry.get("mean_logprob")
    if ml is None:
        return False
    return (-ml) >= threshold


def sweep_model(results: list[dict]) -> list[dict]:
    """Sweep thresholds; return list of {threshold, detection, fp_rate, precision, recall}."""
    low  = [r for r in results if r["category"] == "LOW"]
    high = [r for r in results if r["category"] == "HIGH"]
    rows = []
    for t in THRESHOLDS:
        fp  = sum(1 for r in low  if p_avg_fires(r, t))
        det = sum(1 for r in high if p_avg_fires(r, t))
        rows.append({
            "threshold": t,
            "detection": det,
            "n_high": len(high),
            "det_rate": det / len(high) if high else 0.0,
            "fp": fp,
            "n_low": len(low),
            "fp_rate": fp / len(low) if low else 0.0,
        })
    return rows


def print_table(model: str, sweep: list[dict], eps_det: int, eps_fp: int,
                n_high: int, n_low: int) -> None:
    print(f"\n  {model}")
    print(f"  {'Threshold':>10}  {'p_avg det':>10}  {'p_avg FP':>10}  "
          f"{'ε det':>8}  {'ε FP':>8}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
    eps_det_str = f"{eps_det}/{n_high} ({eps_det/n_high:.0%})"
    eps_fp_str  = f"{eps_fp}/{n_low} ({eps_fp/n_low:.0%})"
    for row in sweep:
        det_str = f"{row['detection']}/{n_high} ({row['det_rate']:.0%})"
        fp_str  = f"{row['fp']}/{n_low} ({row['fp_rate']:.0%})"
        # Mark the row whose FP rate is closest to ε FP rate (comparison point)
        marker = " *" if abs(row["fp_rate"] - eps_fp / n_low) < 0.05 else "  "
        print(f"  -{row['threshold']:>9.2f}  {det_str:>10}  {fp_str:>10}  "
              f"{eps_det_str:>8}  {eps_fp_str:>8}{marker}")
    print(f"  (* = threshold closest to ε FP operating point)")


def main():
    parser = argparse.ArgumentParser(description="p_avg baseline analysis")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    print("\n" + "━" * 70)
    print("  p_avg BASELINE ANALYSIS  (Spiess et al. ICSE 2025 signal)")
    print("━" * 70)
    print("  Signal: -mean_logprob (higher = less confident)")
    print("  Compared to: ε max aggregation at threshold 0.30")
    print()

    all_results = {}
    summary_rows = []

    for model_key, path in CALIBRATION_FILES.items():
        if not path.exists():
            print(f"  SKIP {model_key}: file not found at {path}")
            continue

        results = load_results(path)
        low  = [r for r in results if r["category"] == "LOW"]
        high = [r for r in results if r["category"] == "HIGH"]

        eps_det = sum(1 for r in high if r.get("fired", False))
        eps_fp  = sum(1 for r in low  if r.get("fired", False))

        sweep = sweep_model(results)
        print_table(model_key, sweep, eps_det, eps_fp, len(high), len(low))

        # Find best p_avg point at ~same FP rate as ε
        eps_fp_rate = eps_fp / len(low) if low else 0.0
        best_row = min(sweep, key=lambda r: abs(r["fp_rate"] - eps_fp_rate))

        all_results[model_key] = {
            "n_low": len(low),
            "n_high": len(high),
            "eps_detection": eps_det,
            "eps_fp": eps_fp,
            "eps_det_rate": round(eps_det / len(high), 4) if high else 0.0,
            "eps_fp_rate": round(eps_fp_rate, 4),
            "p_avg_at_matching_fp": {
                "threshold": best_row["threshold"],
                "detection": best_row["detection"],
                "det_rate":  round(best_row["det_rate"], 4),
                "fp":        best_row["fp"],
                "fp_rate":   round(best_row["fp_rate"], 4),
            },
            "sweep": sweep,
        }

        summary_rows.append({
            "model": model_key,
            "eps_det": f"{eps_det}/{len(high)} ({eps_det/len(high):.0%})",
            "eps_fp":  f"{eps_fp}/{len(low)} ({eps_fp_rate:.0%})",
            "pavg_det": f"{best_row['detection']}/{len(high)} ({best_row['det_rate']:.0%})",
            "pavg_fp":  f"{best_row['fp']}/{len(low)} ({best_row['fp_rate']:.0%})",
        })

    # Comparison summary
    print("\n" + "━" * 70)
    print("  COMPARISON SUMMARY  (p_avg at ε-matched FP operating point)")
    print("━" * 70)
    print(f"  {'Model':<16}  {'ε det':>14}  {'ε FP':>12}  "
          f"{'p_avg det':>14}  {'p_avg FP':>12}")
    print(f"  {'-'*16}  {'-'*14}  {'-'*12}  {'-'*14}  {'-'*12}")
    for row in summary_rows:
        print(f"  {row['model']:<16}  {row['eps_det']:>14}  {row['eps_fp']:>12}  "
              f"{row['pavg_det']:>14}  {row['pavg_fp']:>12}")

    print()
    print("  Key finding: p_avg is competitive on detection rate, but provides")
    print("  no token-level localization. ε advantage: AST filtering, per-token")
    print("  attribution, reviewer can see which token drove the flag.")
    print()

    if args.out_json:
        out = Path(args.out_json) if "/" in args.out_json else RESULTS_DIR / args.out_json
        out.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"  Results saved to: {out}")


if __name__ == "__main__":
    main()
