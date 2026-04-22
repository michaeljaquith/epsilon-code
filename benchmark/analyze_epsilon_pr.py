#!/usr/bin/env python3
"""
Epsilon Discrimination Analysis
=================================
Builds a precision/recall (PR) curve and tier-based failure-rate table for
the epsilon cascaded score, using two ground-truth sources:

  LOW  (30 entries, all correct simple logic):
       Ground truth from ground_truth_low.py test harness.
       All 30 pass tests -> fail=False.
       Used as the "correct code" baseline: any epsilon trigger = false positive.

  HIGH (non-LOW entries from review_results_with_context.json):
       Ground truth from the LLM review loop.
       SUSPECT  -> fail=True   (confirmed real error)
       CLEARED  -> fail=False  (epsilon fired but code was correct)
       UNCERTAIN -> excluded

Key metrics:
  - Tier failure rate: among entries epsilon classified at each tier, what
    fraction are confirmed failures?
  - Precision/recall curve: sweep cascaded_score threshold 0->1; precision
    = TP/(TP+FP), recall = TP/(TP+FN)
  - FP rate on LOW: epsilon fires on correct simple logic at each tier

Output: results/epsilon_pr_analysis.json + console tables

Usage:
    python analyze_epsilon_pr.py
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_dataset() -> list[dict]:
    """
    Returns list of {id, model, source, cascaded_score, cascaded_status, fail: bool}.
    LOW entries use test-based ground truth; HIGH entries use review verdicts.
    """
    entries = []

    # -- LOW entries (test-based, all correct) --------------------------------
    gt_path = RESULTS_DIR / "ground_truth_low.json"
    if not gt_path.exists():
        raise RuntimeError("Run ground_truth_low.py first to generate ground truth")
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    for e in gt["entries"]:
        entries.append({
            "id":              e["id"],
            "model":           e["model"],
            "source":          "LOW",
            "cascaded_score":  e["cascaded_score"],
            "cascaded_status": e["cascaded_status"],
            "fail":            not e["pass"],  # all False (all pass)
            "note":            e.get("error") or "",
        })

    # -- HIGH entries (review-based, non-LOW) ---------------------------------
    review_path = RESULTS_DIR / "review_results_with_context.json"
    if not review_path.exists():
        raise RuntimeError("review_results_with_context.json not found")
    rdata = json.loads(review_path.read_text(encoding="utf-8"))
    review_entries = rdata.get("results", rdata) if isinstance(rdata, dict) else rdata

    for r in review_entries:
        if r.get("id", "").startswith("low_"):
            continue  # LOW handled above
        verdict = r.get("verdict", "")
        if verdict == "UNCERTAIN":
            continue  # exclude ambiguous entries
        if verdict not in ("SUSPECT", "CLEARED"):
            continue
        entries.append({
            "id":              r["id"],
            "model":           r.get("generator_model", ""),
            "source":          "HIGH",
            "cascaded_score":  r["cascaded_score"],
            "cascaded_status": r["cascaded_status"],
            "fail":            (verdict == "SUSPECT"),
            "note":            r.get("reason", "")[:80],
        })

    return entries


def tier_table(entries: list[dict]) -> None:
    """Print failure rate per tier, split by source."""
    from collections import defaultdict
    tiers = ["COMPLETE", "FLAGGED", "PAUSED", "ABORTED"]

    # LOW — all correct code, any tier fire = FP
    low = [e for e in entries if e["source"] == "LOW"]
    high = [e for e in entries if e["source"] == "HIGH"]

    print("\nTier breakdown - LOW set (correct code, any fire = false positive):")
    print(f"  {'Tier':<10}  {'Count':>6}  {'FP (eps fired)':>14}  {'FP rate':>8}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*14}  {'-'*8}")
    for tier in tiers:
        t_entries = [e for e in low if e["cascaded_status"] == tier]
        fp = sum(1 for e in t_entries)  # all LOW entries at any tier other than COMPLETE
        # Actually: for LOW, fail=False for all. Any entry classified here IS correct.
        # "FP" means epsilon fired on correct code.
        fired = [e for e in t_entries if e["cascaded_status"] != "COMPLETE"]
        n = len(t_entries)
        if tier == "COMPLETE":
            print(f"  {tier:<10}  {n:>6}  {'0 (correct, no fire)':>14}  {'0%':>8}")
        else:
            fp_rate = f"{n/len(low)*100:.0f}%" if low else "N/A"
            print(f"  {tier:<10}  {n:>6}  {n:>14}  {fp_rate:>8}")

    fp_total = sum(1 for e in low if e["cascaded_status"] != "COMPLETE")
    print(f"  {'TOTAL FP':<10}  {fp_total:>6}  (out of {len(low)} correct LOW functions)")

    print("\nTier breakdown - HIGH set (API code, review-based ground truth):")
    print(f"  {'Tier':<10}  {'Total':>6}  {'Failures':>9}  {'Fail rate':>10}  {'Passes':>7}")
    print(f"  {'-'*10}  {'-'*6}  {'-'*9}  {'-'*10}  {'-'*7}")
    for tier in tiers:
        if tier == "COMPLETE":
            # No COMPLETE entries in review (only FLAGGED+ were reviewed)
            print(f"  {tier:<10}  {'N/A':>6}  {'N/A':>9}  {'(not reviewed)':>10}  {'N/A':>7}")
            continue
        t_entries = [e for e in high if e["cascaded_status"] == tier]
        n = len(t_entries)
        fails = sum(1 for e in t_entries if e["fail"])
        passes = n - fails
        rate = f"{fails/n*100:.0f}%" if n else "N/A"
        print(f"  {tier:<10}  {n:>6}  {fails:>9}  {rate:>10}  {passes:>7}")

    all_high = [e for e in high]
    total_fails = sum(1 for e in all_high if e["fail"])
    total_n = len(all_high)
    print(f"  {'TOTAL':<10}  {total_n:>6}  {total_fails:>9}  {total_fails/total_n*100:.0f}%{' ':>8}  {total_n-total_fails:>7}")


def pr_curve(entries: list[dict]) -> list[dict]:
    """
    Compute precision/recall at thresholds 0.0 to 1.0.
    Treat HIGH SUSPECT as positive (true failure), everything else as negative.
    LOW entries are all negative (correct code).
    Returns list of {threshold, precision, recall, f1, tp, fp, fn, tn}.
    """
    total_pos = sum(1 for e in entries if e["fail"])
    total_neg = len(entries) - total_pos
    print(f"\nPR curve dataset: {len(entries)} entries | {total_pos} true failures | {total_neg} correct")

    curve = []
    for t_int in range(0, 101, 5):
        threshold = t_int / 100.0
        tp = sum(1 for e in entries if e["cascaded_score"] >= threshold and e["fail"])
        fp = sum(1 for e in entries if e["cascaded_score"] >= threshold and not e["fail"])
        fn = sum(1 for e in entries if e["cascaded_score"] < threshold and e["fail"])
        tn = sum(1 for e in entries if e["cascaded_score"] < threshold and not e["fail"])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        curve.append({
            "threshold": threshold, "precision": round(prec, 3),
            "recall": round(rec, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        })
    return curve


def print_pr_table(curve: list[dict]) -> None:
    print(f"\n  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*4}")
    # Highlight key operating points
    key_thresholds = {0.30, 0.65, 0.95}
    for pt in curve:
        marker = " <--" if pt["threshold"] in key_thresholds else ""
        print(f"  {pt['threshold']:>10.2f}  {pt['precision']:>10.3f}  {pt['recall']:>8.3f}  {pt['f1']:>6.3f}  "
              f"{pt['tp']:>4}  {pt['fp']:>4}  {pt['fn']:>4}  {pt['tn']:>4}{marker}")
    best = max(curve, key=lambda x: x["f1"])
    print(f"\n  Best F1={best['f1']:.3f} at threshold={best['threshold']:.2f} "
          f"(precision={best['precision']:.3f}, recall={best['recall']:.3f})")


def current_tier_thresholds(curve: list[dict]) -> None:
    """Show precision/recall at the 3 tier boundaries: 0.30, 0.65, 0.95."""
    print("\nCurrent tier boundary operating points:")
    print(f"  {'Boundary':<20}  {'Tier fired':<8}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*6}  {'-'*6}  {'-'*6}")
    labels = [(0.30, "FLAGGED+"), (0.65, "PAUSED+"), (0.95, "ABORTED")]
    for thresh, label in labels:
        pt = min(curve, key=lambda x: abs(x["threshold"] - thresh))
        print(f"  {thresh:.2f} ({label:<15})  {pt['precision']:>6.3f}  {pt['recall']:>6.3f}  {pt['f1']:>6.3f}")


def main() -> None:
    entries = load_dataset()
    print(f"Loaded {len(entries)} entries: "
          f"{sum(1 for e in entries if e['source']=='LOW')} LOW + "
          f"{sum(1 for e in entries if e['source']=='HIGH')} HIGH")

    tier_table(entries)
    curve = pr_curve(entries)
    print_pr_table(curve)
    current_tier_thresholds(curve)

    # Per-model breakdown
    print("\nHIGH failure rate by model (SUSPECT / total reviewed):")
    high = [e for e in entries if e["source"] == "HIGH"]
    from collections import defaultdict
    by_model: dict = defaultdict(lambda: {"total": 0, "fail": 0})
    for e in high:
        by_model[e["model"]]["total"] += 1
        by_model[e["model"]]["fail"]  += int(e["fail"])
    for model, counts in sorted(by_model.items()):
        t, f = counts["total"], counts["fail"]
        print(f"  {model:<35}  {f}/{t}  ({f/t*100:.0f}%)")

    out = {
        "meta": {
            "description": "Epsilon precision/recall analysis",
            "total_entries": len(entries),
            "low_entries": sum(1 for e in entries if e["source"] == "LOW"),
            "high_entries": sum(1 for e in entries if e["source"] == "HIGH"),
            "total_failures": sum(1 for e in entries if e["fail"]),
        },
        "pr_curve": curve,
        "entries": entries,
    }
    out_path = RESULTS_DIR / "epsilon_pr_analysis.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
