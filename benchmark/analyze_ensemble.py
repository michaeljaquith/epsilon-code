#!/usr/bin/env python3
"""
Ensemble analysis: compare cascaded feature distributions across TP vs FP cases.
Loads the 3 production result JSONs and prints feature statistics to guide
weight calibration of the cascaded_epsilon algorithm.

Usage:
    python analyze_ensemble.py               # use stored cascaded_score from JSON
    python analyze_ensemble.py --recompute   # re-run cascaded algo on saved eps_seq
"""
import json
import math
import sys
from pathlib import Path


RESULT_FILES = [
    "results/production_gpt-4o.json",
    "results/production_gpt-4o-mini.json",
    "results/production_deepseek-ai_DeepSeek-V3.json",
    "results/scenarios_gpt-4o.json",
    "results/scenarios_gpt-4o-mini.json",
    "results/scenarios_deepseek-ai_DeepSeek-V3.json",
]

REVIEW_FILE = "results/review_results.json"

FEATURES = [
    "peak_eps", "cluster_count", "max_run", "re_escalation", "elev_fraction", "total_tokens"
]


def recompute_cascaded(eps_seq: list[float]) -> dict:
    """Re-run cascaded algorithm on a saved eps_seq. Mirrors cascaded_epsilon() logic."""
    n = len(eps_seq)
    peak_eps = max(eps_seq) if eps_seq else 0.0
    if peak_eps < 0.10:
        return {"cascaded_score": 0.0, "evidence": {"peak_eps": round(peak_eps, 4), "total_tokens": n}}
    if peak_eps < 0.30:
        return {"cascaded_score": round(peak_eps * 0.5, 4),
                "evidence": {"peak_eps": round(peak_eps, 4), "total_tokens": n}}

    peak_idx = eps_seq.index(peak_eps)
    cluster_count = sum(1 for e in eps_seq if e > 0.20)
    max_run = cur_run = 0
    for e in eps_seq:
        if e > 0.15:
            cur_run += 1
            max_run = max(max_run, cur_run)
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
                    in_cluster = False
                    gap_len = 0
        else:
            if eps_seq[i] > 0.20:
                re_escalation += 1
                in_cluster = True
                gap_len = 0
    elev_fraction = cluster_count / n if n > 0 else 0.0
    if n <= 6 and re_escalation == 0:
        lone_spike_tier = 1
        lone_spike_penalty = 0.70
    elif n <= 40 and re_escalation == 0 and 3 <= cluster_count <= 5 and max_run <= 1:
        lone_spike_tier = 2
        lone_spike_penalty = 0.40
    else:
        lone_spike_tier = 0
        lone_spike_penalty = 0.0

    score = peak_eps
    score += 0.10 * min(cluster_count / 12.0, 1.0)
    score += 0.10 * min(max_run / 4.0, 1.0)
    if drops_fast:
        score -= 0.15
    score += 0.05 * min(re_escalation, 3)
    score += 0.10 * min(elev_fraction / 0.15, 1.0)
    score -= lone_spike_penalty
    score = max(0.0, min(1.0, score))

    return {
        "cascaded_score": round(score, 4),
        "evidence": {
            "peak_eps": round(peak_eps, 4), "cluster_count": cluster_count,
            "max_run": max_run, "drops_fast": drops_fast,
            "re_escalation": re_escalation, "elev_fraction": round(elev_fraction, 4),
            "lone_spike": lone_spike_tier > 0, "lone_spike_pen": round(lone_spike_penalty, 4),
            "total_tokens": n,
        },
    }


def load_all(recompute: bool = False) -> list[dict]:
    rows = []
    for f in RESULT_FILES:
        p = Path(f)
        if not p.exists():
            print(f"  MISSING: {f}")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        model = data["meta"]["model"]
        for r in data["results"]:
            if r.get("epsilon") is None:
                continue
            ev = r.get("evidence", {})
            if recompute and r.get("eps_seq"):
                recomp = recompute_cascaded(r["eps_seq"])
                casc_score = recomp["cascaded_score"]
                ev = recomp["evidence"]
            else:
                casc_score = r.get("cascaded_score", 0.0)
            casc_fired = casc_score >= 0.30
            expected_fire = r["type"] == "API"
            casc_label = ("TP" if casc_fired and expected_fire else
                          "TN" if not casc_fired and not expected_fire else
                          "FP" if casc_fired and not expected_fire else "FN")
            rows.append({
                "model":       model,
                "name":        r.get("name", r.get("id", "?")),
                "type":        r["type"],
                "label":       r["label"],
                "casc_label":  casc_label,
                "epsilon":     r["epsilon"],
                "casc_score":  casc_score,
                "peak_eps":    ev.get("peak_eps", r["epsilon"]),
                "cluster_count":   ev.get("cluster_count", 0),
                "max_run":         ev.get("max_run", 0),
                "drops_fast":      ev.get("drops_fast", False),
                "re_escalation":   ev.get("re_escalation", 0),
                "elev_fraction":   ev.get("elev_fraction", 0.0),
                "total_tokens":    ev.get("total_tokens", r.get("token_count", 0)),
                "lone_spike":      ev.get("lone_spike", False),
                "lone_spike_pen":  ev.get("lone_spike_pen", 0.0),
            })
    return rows


def stats(vals: list[float]) -> str:
    if not vals:
        return "  n=0"
    n = len(vals)
    mu = sum(vals) / n
    variance = sum((v - mu) ** 2 for v in vals) / n
    sd = math.sqrt(variance)
    mn, mx = min(vals), max(vals)
    sorted_v = sorted(vals)
    med = sorted_v[n // 2]
    return f"  n={n:>3}  mean={mu:>6.3f}  med={med:>6.3f}  sd={sd:>5.3f}  [{mn:.3f}, {mx:.3f}]"


def print_feature_table(rows: list[dict], label_key: str = "label") -> None:
    tp_rows = [r for r in rows if r[label_key] == "TP"]
    tn_rows = [r for r in rows if r[label_key] == "TN"]
    fp_rows = [r for r in rows if r[label_key] == "FP"]
    fn_rows = [r for r in rows if r[label_key] == "FN"]

    print(f"\n  {'Feature':<18}  {'TP':>60}  {'FP':>60}")
    print(f"  {'-'*18}  {'-'*60}  {'-'*60}")
    for feat in FEATURES:
        tp_vals = [r[feat] for r in tp_rows if r[feat] is not None]
        fp_vals = [r[feat] for r in fp_rows if r[feat] is not None]
        print(f"  {feat:<18}  {stats(tp_vals)}  {stats(fp_vals)}")

    print(f"\n  {'Feature':<18}  {'TN':>60}  {'FN':>60}")
    print(f"  {'-'*18}  {'-'*60}  {'-'*60}")
    for feat in FEATURES:
        tn_vals = [r[feat] for r in tn_rows if r[feat] is not None]
        fn_vals = [r[feat] for r in fn_rows if r[feat] is not None]
        print(f"  {feat:<18}  {stats(tn_vals)}  {stats(fn_vals)}")

    print(f"\n  drops_fast rate:  TP={sum(r['drops_fast'] for r in tp_rows)}/{len(tp_rows)}"
          f"  FP={sum(r['drops_fast'] for r in fp_rows)}/{len(fp_rows)}"
          f"  TN={sum(r['drops_fast'] for r in tn_rows)}/{len(tn_rows)}")


def print_logic_detail(rows: list[dict]) -> None:
    """Print all LOGIC-type functions across all 3 models."""
    logic = [r for r in rows if r["type"] == "LOGIC"]
    print(f"\n  {'model':<22} {'name':<44} {'orig':>4}  {'casc_s':>6}  {'casc_l':>5}  "
          f"{'clust':>5}  {'run':>3}  {'re_e':>4}  {'frac':>5}  n")
    print(f"  {'-'*22} {'-'*44} {'-'*4}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*3}  {'-'*4}  {'-'*5}  {'-'*4}")
    for r in sorted(logic, key=lambda x: (x["name"], x["model"])):
        df = "*" if r["drops_fast"] else " "
        print(f"  {r['model']:<22} {r['name']:<44} [{r['label']}]  "
              f"{r['casc_score']:>6.3f}  [{r['casc_label']}]  "
              f"{r['cluster_count']:>5}  {r['max_run']:>3}  {r['re_escalation']:>4}  "
              f"{r['elev_fraction']:>5.3f} {df} {r['total_tokens']:>4}")


def print_cluster_histogram(rows: list[dict]) -> None:
    """Show cluster_count distribution: TP vs FP side by side."""
    tp_rows = [r for r in rows if r["label"] == "TP"]
    fp_rows = [r for r in rows if r["label"] == "FP"]
    max_c = max((r["cluster_count"] for r in rows if r["label"] in ("TP", "FP")), default=0)

    print(f"\n  cluster_count histogram (TP vs FP):")
    print(f"  {'count':>6}  {'TP':>5}  {'FP':>5}")
    print(f"  {'-'*6}  {'-'*5}  {'-'*5}")
    for c in range(0, min(max_c + 2, 25)):
        n_tp = sum(1 for r in tp_rows if r["cluster_count"] == c)
        n_fp = sum(1 for r in fp_rows if r["cluster_count"] == c)
        if n_tp + n_fp > 0:
            bar_tp = "#" * n_tp
            bar_fp = "#" * n_fp
            print(f"  {c:>6}  {bar_tp:<5}  {bar_fp:<5}  ({n_tp} TP, {n_fp} FP)")


def print_max_run_histogram(rows: list[dict]) -> None:
    """Show max_run distribution."""
    tp_rows = [r for r in rows if r["label"] == "TP"]
    fp_rows = [r for r in rows if r["label"] == "FP"]
    max_r = max((r["max_run"] for r in rows if r["label"] in ("TP", "FP")), default=0)

    print(f"\n  max_run (consecutive eps>0.15) histogram:")
    print(f"  {'run':>4}  {'TP':>5}  {'FP':>5}")
    print(f"  {'-'*4}  {'-'*5}  {'-'*5}")
    for r_val in range(0, min(max_r + 2, 20)):
        n_tp = sum(1 for r in tp_rows if r["max_run"] == r_val)
        n_fp = sum(1 for r in fp_rows if r["max_run"] == r_val)
        if n_tp + n_fp > 0:
            bar_tp = "#" * n_tp
            bar_fp = "#" * n_fp
            print(f"  {r_val:>4}  {bar_tp:<5}  {bar_fp:<5}  ({n_tp} TP, {n_fp} FP)")


def suggest_thresholds(rows: list[dict]) -> None:
    """Try different score thresholds and report accuracy."""
    relevant = [r for r in rows if r["label"] in ("TP", "FP", "TN", "FN")]
    api = [r for r in relevant if r["type"] == "API"]
    logic = [r for r in relevant if r["type"] == "LOGIC"]

    print(f"\n  Threshold sweep on cascaded_score:")
    print(f"  {'thresh':>7}  {'det%':>6}  {'fp%':>6}  TP  FN  FP  TN  {'accuracy':>8}")
    print(f"  {'-'*7}  {'-'*6}  {'-'*6}  --  --  --  --  {'-'*8}")

    for t_int in range(20, 80, 5):
        t = t_int / 100.0
        tp = sum(1 for r in api   if r["casc_score"] >= t)
        fn = sum(1 for r in api   if r["casc_score"] <  t)
        fp = sum(1 for r in logic if r["casc_score"] >= t)
        tn = sum(1 for r in logic if r["casc_score"] <  t)
        n_api   = len(api)
        n_logic = len(logic)
        det = tp / n_api   * 100 if n_api   else 0
        fpr = fp / n_logic * 100 if n_logic else 0
        acc = (tp + tn) / (n_api + n_logic) * 100
        print(f"  {t:>7.2f}  {det:>6.1f}  {fpr:>6.1f}  {tp:>2}  {fn:>2}  {fp:>2}  {tn:>2}  {acc:>8.1f}%")


def load_review_verdicts() -> dict:
    """Load review_results.json and return a {(source_file, id): verdict} dict."""
    p = Path(REVIEW_FILE)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    verdicts = {}
    for r in data.get("results", []):
        key = (r.get("source_file", ""), r.get("id", ""))
        verdicts[key] = r.get("verdict", "UNCERTAIN")
    return verdicts


def print_review_analysis(rows: list[dict], verdicts: dict) -> None:
    """Show how review verdicts change effective tier distribution."""
    if not verdicts:
        print("  (no review_results.json found — run review_loop.py first)")
        return

    flagged_paused = [r for r in rows
                      if r.get("casc_label") in ("TP", "FP")
                      and r.get("epsilon", 0) >= 0.30]

    # Attach verdict to each row
    enriched = []
    for r in flagged_paused:
        # Key must match what review_loop.py stored
        src = f"production_{r['model'].replace('/', '_')}.json" \
              if "scenario" not in r.get("casc_label", "zzz") else \
              f"scenarios_{r['model'].replace('/', '_')}.json"
        # Try both filename patterns
        v = (verdicts.get((src, r["name"]))
             or verdicts.get((src.replace("production_", "scenarios_"), r["name"]))
             or verdicts.get((src.replace("scenarios_", "production_"), r["name"]))
             or "NOT_REVIEWED")
        enriched.append({**r, "verdict": v})

    total   = len(enriched)
    cleared = sum(1 for r in enriched if r["verdict"] == "CLEARED")
    uncert  = sum(1 for r in enriched if r["verdict"] == "UNCERTAIN")
    suspect = sum(1 for r in enriched if r["verdict"] == "SUSPECT")
    not_rev = sum(1 for r in enriched if r["verdict"] == "NOT_REVIEWED")

    print(f"\n  Reviewed entries : {total}")
    print(f"  CLEARED          : {cleared}  ({cleared/total*100:.0f}%)")
    print(f"  UNCERTAIN        : {uncert}   ({uncert/total*100:.0f}%)")
    print(f"  SUSPECT          : {suspect}  ({suspect/total*100:.0f}%)")
    if not_rev:
        print(f"  NOT_REVIEWED     : {not_rev}")

    # FP population: how many LOGIC FPs were cleared vs. remain
    fp_rows = [r for r in enriched if r.get("type") == "LOGIC"]
    if fp_rows:
        fp_cleared = sum(1 for r in fp_rows if r["verdict"] == "CLEARED")
        print(f"\n  LOGIC FPs after review: {len(fp_rows)} total")
        print(f"    Cleared (confirmed correct): {fp_cleared}")
        print(f"    Remaining flagged:           {len(fp_rows) - fp_cleared}")
        print(f"    Effective FP rate at REVIEW tier: "
              f"{(len(fp_rows)-fp_cleared)/max(1,sum(1 for r in rows if r['type']=='LOGIC'))*100:.0f}%")

    # Show all SUSPECT entries — these are true concerns
    suspects = [r for r in enriched if r["verdict"] == "SUSPECT"]
    if suspects:
        print(f"\n  SUSPECT entries (genuine issues found by reviewer):")
        for r in suspects:
            print(f"    [{r['type']:<5}] {r['model'][:14]:<14} {r['name']:<40} "
                  f"casc={r['casc_score']:.3f}")


def main() -> None:
    recompute = "--recompute" in sys.argv
    rows = load_all(recompute=recompute)
    if recompute:
        print("  (re-computed cascaded scores from saved eps_seq with updated algorithm)")
    print(f"Loaded {len(rows)} results from {len(RESULT_FILES)} model files.")

    api_rows   = [r for r in rows if r["type"] == "API"]
    logic_rows = [r for r in rows if r["type"] == "LOGIC"]
    print(f"API: {len(api_rows)}  LOGIC: {len(logic_rows)}")

    print(f"\n{'='*80}")
    print("ORIGINAL ALGORITHM — label distribution")
    print(f"{'='*80}")
    for lbl in ("TP", "FN", "FP", "TN"):
        n = sum(1 for r in rows if r["label"] == lbl)
        print(f"  {lbl}: {n}")

    print(f"\n{'='*80}")
    print("CASCADED ALGORITHM — label distribution")
    print(f"{'='*80}")
    for lbl in ("TP", "FN", "FP", "TN"):
        n = sum(1 for r in rows if r["casc_label"] == lbl)
        print(f"  {lbl}: {n}")

    print(f"\n{'='*80}")
    print("FEATURE STATISTICS — original labels")
    print(f"{'='*80}")
    print_feature_table(rows, label_key="label")

    print(f"\n{'='*80}")
    print("CLUSTER_COUNT HISTOGRAM")
    print(f"{'='*80}")
    print_cluster_histogram(rows)

    print(f"\n{'='*80}")
    print("MAX_RUN HISTOGRAM")
    print(f"{'='*80}")
    print_max_run_histogram(rows)

    print(f"\n{'='*80}")
    print("LOGIC FUNCTIONS — detailed evidence (all models)")
    print(f"{'='*80}")
    print_logic_detail(rows)

    print(f"\n{'='*80}")
    print("THRESHOLD SWEEP")
    print(f"{'='*80}")
    suggest_thresholds(rows)

    # Show cases where original and cascaded labels disagree
    print(f"\n{'='*80}")
    print("LABEL CHANGES: original -> cascaded")
    print(f"{'='*80}")
    changed = [r for r in rows if r["label"] != r["casc_label"]]
    if changed:
        for r in sorted(changed, key=lambda x: (x["type"], x["casc_label"])):
            lsp = r.get("lone_spike_pen", 0)
            print(f"  [{r['type']:<5}] {r['model']:<22} {r['name']:<44} "
                  f"{r['label']} -> {r['casc_label']}  "
                  f"eps={r['epsilon']:.3f}  casc={r['casc_score']:.3f}  "
                  f"lone_pen={lsp:.2f}  n={r['total_tokens']}")
    else:
        print("  (no changes)")

    print(f"\n{'='*80}")
    print("REVIEW LOOP VERDICTS")
    print(f"{'='*80}")
    verdicts = load_review_verdicts()
    print_review_analysis(rows, verdicts)

    # Check if genuinely-pure FPs can be separated
    print(f"\n{'='*80}")
    print("TRUE AMBIGUOUS vs GENUINELY PURE LOGIC FPs")
    print(f"{'='*80}")
    logic_fp = [r for r in logic_rows if r["label"] == "FP"]
    pure_logic_fps = ["test_token", "read_user_me", "health_check", "get_current_active_superuser"]
    schema_fps = ["render_email_template", "generate_test_email",
                  "generate_reset_password_email", "generate_new_account_email"]
    print(f"\n  Genuinely pure logic FPs (should be TN — model verbose or wording uncertain):")
    for r in logic_fp:
        if r["name"] in pure_logic_fps:
            df = "*" if r["drops_fast"] else " "
            print(f"  [{r['model'][:12]:<12}] {r['name']:<44} "
                  f"casc={r['casc_score']:.3f} clust={r['cluster_count']} run={r['max_run']}"
                  f" re={r['re_escalation']} frac={r['elev_fraction']:.3f}{df}")
    print(f"\n  Schema-ambiguous FPs (legitimately uncertain — classification may be wrong):")
    for r in logic_fp:
        if r["name"] in schema_fps:
            df = "*" if r["drops_fast"] else " "
            print(f"  [{r['model'][:12]:<12}] {r['name']:<44} "
                  f"casc={r['casc_score']:.3f} clust={r['cluster_count']} run={r['max_run']}"
                  f" re={r['re_escalation']} frac={r['elev_fraction']:.3f}{df}")


if __name__ == "__main__":
    main()
