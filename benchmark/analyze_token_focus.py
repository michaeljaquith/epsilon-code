"""
Token-level focus analysis.

The paper currently shows ε flags 97% of API functions (function-level review rate).
This script computes the complementary token-level picture: within a flagged function,
ε localizes to a small fraction of tokens — the specific decision points a developer
should examine. This answers the "why not review everything?" objection.

Usage:
    python analyze_token_focus.py [--verbose]
"""

import json
import statistics
import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS_DIR = Path(__file__).parent / "results"

SCENARIO_FILES = [
    "scenarios_gpt-4o.json",
    "scenarios_gpt-4o-mini.json",
    "scenarios_gpt-4-turbo.json",
    "scenarios_deepseek-ai_DeepSeek-V3.json",
]

# Epsilon thresholds (must match epsilon/core.py)
THR_FLAG   = 0.30   # FLAGGED tier: review triggered
THR_PAUSED = 0.95   # PAUSED tier: genuinely extreme uncertainty
THR_FLOOR  = 0.70   # accumulation floor: compounds into function-level ε


def load_all_entries():
    entries = []
    for fname in SCENARIO_FILES:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"  [warn] not found: {path}")
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = data["meta"]["model"]
        for r in data["results"]:
            r["_model"] = model
            entries.append(r)
    return entries


def token_counts(eps_seq, threshold):
    return sum(1 for e in eps_seq if e > threshold)


def analyze(entries, label, verbose=False):
    total_fns = len(entries)
    if total_fns == 0:
        return {}

    flagged_fns = [e for e in entries if e.get("fired")]
    complete_fns = [e for e in entries if not e.get("fired")]

    total_tokens = sum(e["token_count"] for e in entries)
    total_tokens_flagged = sum(e["token_count"] for e in flagged_fns)

    # Token counts above each threshold across all entries
    tok_above_flag   = sum(token_counts(e["eps_seq"], THR_FLAG)   for e in entries)
    tok_above_paused = sum(token_counts(e["eps_seq"], THR_PAUSED) for e in entries)
    tok_above_floor  = sum(token_counts(e["eps_seq"], THR_FLOOR)  for e in entries)

    # Per-flagged-function: tokens above threshold
    per_fn_flag   = [token_counts(e["eps_seq"], THR_FLAG)   for e in flagged_fns]
    per_fn_paused = [token_counts(e["eps_seq"], THR_PAUSED) for e in flagged_fns]
    per_fn_tokens = [e["token_count"] for e in flagged_fns]

    avg_fn_size    = statistics.mean(per_fn_tokens) if per_fn_tokens else 0
    avg_flag_tok   = statistics.mean(per_fn_flag)   if per_fn_flag   else 0
    avg_paused_tok = statistics.mean(per_fn_paused) if per_fn_paused else 0

    # Fraction of tokens that are "focus tokens" within a flagged function
    focus_pct_flag   = avg_flag_tok   / avg_fn_size if avg_fn_size else 0
    focus_pct_paused = avg_paused_tok / avg_fn_size if avg_fn_size else 0

    # Review burden: if you reviewed EVERY token in every flagged function vs
    # only the tokens ε highlights
    naive_review = total_tokens_flagged
    epsilon_focus_flag   = sum(per_fn_flag)
    epsilon_focus_paused = sum(per_fn_paused)

    reduction_flag   = 1 - epsilon_focus_flag   / naive_review if naive_review else 0
    reduction_paused = 1 - epsilon_focus_paused / naive_review if naive_review else 0

    result = {
        "label": label,
        "total_fns": total_fns,
        "flagged_fns": len(flagged_fns),
        "fn_review_rate": len(flagged_fns) / total_fns,
        "total_tokens": total_tokens,
        "total_tokens_flagged_fns": total_tokens_flagged,
        "tok_above_flag":   tok_above_flag,
        "tok_above_paused": tok_above_paused,
        "tok_above_floor":  tok_above_floor,
        "tok_rate_flag":    tok_above_flag   / total_tokens,
        "tok_rate_paused":  tok_above_paused / total_tokens,
        "avg_fn_size":      avg_fn_size,
        "avg_focus_tokens_flag":   avg_flag_tok,
        "avg_focus_tokens_paused": avg_paused_tok,
        "focus_pct_flag":   focus_pct_flag,
        "focus_pct_paused": focus_pct_paused,
        "naive_review_tokens":          naive_review,
        "epsilon_review_tokens_flag":   epsilon_focus_flag,
        "epsilon_review_tokens_paused": epsilon_focus_paused,
        "reduction_flag":   reduction_flag,
        "reduction_paused": reduction_paused,
    }

    print(f"\n{'='*55}")
    print(f"  {label}  ({total_fns} entries across all models)")
    print(f"{'='*55}")

    print(f"\n  Function-level")
    print(f"    Flagged for review:     {len(flagged_fns)}/{total_fns} = {len(flagged_fns)/total_fns:.1%}")
    print(f"    COMPLETE (no review):   {len(complete_fns)}/{total_fns} = {len(complete_fns)/total_fns:.1%}")

    print(f"\n  Token-level (all {total_tokens} tokens in corpus)")
    print(f"    Tokens > {THR_FLAG:.2f} (flag):   {tok_above_flag:4d}/{total_tokens} = {tok_above_flag/total_tokens:.1%}")
    print(f"    Tokens > {THR_PAUSED:.2f} (paused): {tok_above_paused:4d}/{total_tokens} = {tok_above_paused/total_tokens:.1%}")
    print(f"    Tokens > {THR_FLOOR:.2f} (floor):  {tok_above_floor:4d}/{total_tokens} = {tok_above_floor/total_tokens:.1%}")

    print(f"\n  Within a flagged function (avg {avg_fn_size:.0f} tokens)")
    print(f"    ε localizes to  ~{avg_flag_tok:.1f} tokens at flag threshold  ({focus_pct_flag:.1%} of function)")
    print(f"    ε localizes to  ~{avg_paused_tok:.1f} tokens at paused threshold ({focus_pct_paused:.1%} of function)")

    print(f"\n  Review burden (flagged functions only, {total_tokens_flagged} total tokens)")
    print(f"    Without ε — read all tokens:      {naive_review:6d} tokens")
    print(f"    With ε at flag threshold:          {epsilon_focus_flag:6d} tokens  ({reduction_flag:.0%} reduction)")
    print(f"    With ε at paused threshold:        {epsilon_focus_paused:6d} tokens  ({reduction_paused:.0%} reduction)")

    if verbose:
        print(f"\n  Per-flagged-function distribution (flag threshold):")
        dist = {}
        for v in per_fn_flag:
            dist[v] = dist.get(v, 0) + 1
        for k in sorted(dist):
            bar = "#" * dist[k]
            print(f"    {k:3d} focus tokens: {dist[k]:3d} fns  {bar}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("Loading scenario benchmark data...")
    all_entries = load_all_entries()
    print(f"Loaded {len(all_entries)} entries from {len(SCENARIO_FILES)} model files")

    api     = [e for e in all_entries if e["type"] == "API"]
    logic   = [e for e in all_entries if e["type"] == "LOGIC"]

    r_api   = analyze(api,   "API prompts (version-split / uncertain)", args.verbose)
    r_logic = analyze(logic, "LOGIC prompts (pure-logic baselines)",    args.verbose)
    r_all   = analyze(all_entries, "ALL prompts",                        args.verbose)

    # Summary table for paper
    print(f"\n\n{'='*55}")
    print("  SUMMARY TABLE  (for paper narrative)")
    print(f"{'='*55}")
    print(f"{'Metric':<42} {'API':>8} {'LOGIC':>8} {'ALL':>8}")
    print("-"*66)

    def row(name, key, fmt=".1%"):
        vals = [r_api.get(key,0), r_logic.get(key,0), r_all.get(key,0)]
        formatted = [f"{v:{fmt}}" for v in vals]
        print(f"  {name:<40} {formatted[0]:>8} {formatted[1]:>8} {formatted[2]:>8}")

    row("Function-level review rate",       "fn_review_rate")
    row("Token rate above flag (0.30)",     "tok_rate_flag")
    row("Token rate above paused (0.65)",   "tok_rate_paused")
    row("Avg fn size (tokens)",             "avg_fn_size",      ".0f")
    row("Avg focus tokens @ flag",          "avg_focus_tokens_flag",   ".1f")
    row("Avg focus tokens @ paused",        "avg_focus_tokens_paused", ".1f")
    row("Focus fraction @ flag",            "focus_pct_flag")
    row("Focus fraction @ paused",          "focus_pct_paused")
    row("Token review burden reduction @ flag",   "reduction_flag")
    row("Token review burden reduction @ paused", "reduction_paused")

    print(f"\n  KEY TAKEAWAY:")
    flag_pct   = r_api["focus_pct_flag"]
    paused_pct = r_api["focus_pct_paused"]
    avg_sz     = r_api["avg_fn_size"]
    avg_f      = r_api["avg_focus_tokens_flag"]
    avg_p      = r_api["avg_focus_tokens_paused"]
    print(f"  A developer reviewing a FLAGGED API function (~{avg_sz:.0f} tokens) with ε")
    print(f"  reads {avg_f:.0f} focus tokens ({flag_pct:.0%} of the function) at the flag threshold,")
    print(f"  or {avg_p:.0f} focus tokens ({paused_pct:.1%}) at the paused threshold —")
    print(f"  not the full function.")


if __name__ == "__main__":
    main()
