#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")
"""
p_avg Baseline — Full Review Loop Comparison
============================================
Tests whether token-level localization is responsible for the ε review loop's
precision gain by running the SAME review loop with p_avg flagging instead of ε.

Design:
  1. Load production + scenario JSONs (must have mean_logprob field — re-run
     benchmark_production.py and benchmark_scenarios.py if missing)
  2. Sweep p_avg thresholds; pick the one whose API/HIGH recall best matches ε
  3. Flag entries based on p_avg at that threshold
  4. Run the identical review + consolidation loop, but WITHOUT token attribution
     in the prompt (reviewer sees the whole function, not the specific token)
  5. Report end-to-end precision and compare to ε's published 94%

Requires:
    python benchmark_production.py --model gpt-4o
    python benchmark_production.py --model gpt-4o-mini
    python benchmark_production.py --provider together --model deepseek-ai/DeepSeek-V3
    python benchmark_scenarios.py  --model gpt-4o
    python benchmark_scenarios.py  --model gpt-4o-mini
    python benchmark_scenarios.py  --provider together --model deepseek-ai/DeepSeek-V3

Usage:
    python benchmark_p_avg_review.py
    python benchmark_p_avg_review.py --reviewer gpt-4o
    python benchmark_p_avg_review.py --workers 10
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

_env = Path(__file__).parent.parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

RESULT_DIR = Path(__file__).parent / "results"

RESULT_FILES = [
    "production_gpt-4o.json",
    "production_gpt-4o-mini.json",
    "production_deepseek-ai_DeepSeek-V3.json",
    "scenarios_gpt-4o.json",
    "scenarios_gpt-4o-mini.json",
    "scenarios_deepseek-ai_DeepSeek-V3.json",
]

P_AVG_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.40]

DOMAIN_FACTS = """\
Known breaking changes — treat as ground truth:
  - openai>=1.0: ChatCompletion.create was removed → must use client.chat.completions.create
  - Python 3.12+: datetime.utcnow() deprecated → use datetime.now(timezone.utc)
  - Pydantic v2: .dict() deprecated → .model_dump() is correct
  - jinja2.Template(x): x is template SOURCE TEXT, not a filename — reading from file requires
    Template(Path(file).read_text())
  - bare `except: pass` silently swallows all errors (SUSPECT when spec implies propagation)
  - Generated output containing markdown fences (```python) or prose explanation is format
    contamination and always SUSPECT"""

SYSTEM_PROMPT = f"""\
You are a senior Python code reviewer assessing LLM-generated code for correctness.

You will be shown:
  1. A function specification
  2. The LLM-generated implementation
  3. A confidence signal indicating overall generation uncertainty (NO token-level detail)

Respond with EXACTLY this format (no other text):
VERDICT: <CLEARED|UNCERTAIN|SUSPECT>
REASON: <one sentence>

CLEARED   = code appears correct given the spec and context
UNCERTAIN = correctness cannot be determined without external schema or runtime access
SUSPECT   = the implementation contains a likely error

DO return SUSPECT for:
  - Deprecated or removed API calls (see domain facts)
  - Wrong method signatures or broken call patterns
  - Logic that produces incorrect output given the spec
  - Silent exception handling (bare except/pass) when spec implies error propagation
  - Output containing markdown fences or prose explanation (format contamination)

Do NOT return SUSPECT for:
  - Variable naming preferences (lst vs list, _items vs _count, etc.)
  - Error message phrasing when the HTTP status code is correct
  - Choice between two equally valid libraries unless the spec mandates one
  - Code style or indentation preferences

{DOMAIN_FACTS}"""

CONSOLIDATION_PROMPT = f"""\
Final calibration pass: you are seeing ALL entries still flagged SUSPECT after
individual reviews. Your job is cross-entry pattern normalization.

{DOMAIN_FACTS}

CRITICAL: The domain facts above are ground truth. A pattern that appears in multiple
entries because all models share the same bug is still a bug — do NOT downgrade entries
solely because the same mistake appears consistently across models. Only downgrade when
the "failure" is a style preference with no correctness impact.

Look for:
  - Multiple entries sharing the same "failure" that is actually a consistent style
    preference → DOWNGRADE
  - Entries where the only issue is a known-correct pattern the reviewer flagged → DOWNGRADE
  - An entry that looks ambiguous alone but is clearly wrong given the full set → CONFIRM
  - Any entry matching a domain fact violation (deprecated API, wrong call sig) → always CONFIRM

For each entry output EXACTLY one line:
  ENTRY_ID ||| CONFIRMED — reason
  OR
  ENTRY_ID ||| DOWNGRADE — reason

Use ||| as the delimiter.
Entries:"""


def build_p_avg_prompt(r: dict) -> str:
    generated = r.get("generated", "").strip()
    if not generated:
        return ""
    spec = r.get("signature", r.get("prompt", r.get("label", ""))).strip()
    p_avg_score = r.get("_p_avg_score", 0.0)
    return f"""\
FUNCTION SPECIFICATION:
{spec}

GENERATED IMPLEMENTATION:
{generated}

UNCERTAINTY SIGNAL:
  Signal         : p_avg (sequence-level mean log-probability baseline)
  -mean_logprob  : {p_avg_score:.4f}  (higher = less confident)
  Token detail   : not available — this is a function-level signal with no attribution

TOP ALTERNATIVES AT TRIGGER POSITION:
(not available — p_avg provides no token localization)"""


def _review_one(client: OpenAI, reviewer_model: str, entry: dict) -> dict:
    prompt = build_p_avg_prompt(entry)
    if not prompt:
        return {**entry, "verdict": "UNCERTAIN",
                "reason": "no generated text", "review_model": reviewer_model}
    resp = client.chat.completions.create(
        model=reviewer_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=120,
    )
    text    = (resp.choices[0].message.content or "").strip()
    verdict = "UNCERTAIN"
    reason  = text
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            if v in ("CLEARED", "UNCERTAIN", "SUSPECT"):
                verdict = v
        elif line.startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return {**entry, "verdict": verdict, "reason": reason,
            "review_model": reviewer_model}


def run_review_phase(client: OpenAI, reviewer_model: str,
                     entries: list[dict], workers: int) -> list[dict]:
    print(f"\n  Review pass — {len(entries)} entries, {workers} workers...")
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_review_one, client, reviewer_model, e): e
                   for e in entries}
        for i, fut in enumerate(as_completed(futures), 1):
            entry = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                print(f"  [{i:03d}/{len(entries)}] {result['verdict']:<9}  "
                      f"[{entry.get('generator_model','?')[:12]:<12}] "
                      f"{entry.get('id','?')[:20]:<20}  {result['reason'][:55]}")
            except Exception as ex:
                results.append({**entry, "verdict": "UNCERTAIN",
                                 "reason": f"error: {ex}",
                                 "review_model": reviewer_model})
                print(f"  [{i:03d}/{len(entries)}] ERROR  {entry.get('id','?')}: {ex}")
    return results


def consolidation_pass(client: OpenAI, reviewer_model: str,
                       suspects: list[dict]) -> dict[tuple, str]:
    if not suspects:
        return {}
    keys = [f"ENTRY_{i:03d}" for i in range(len(suspects))]
    key_to_entry = dict(zip(keys, suspects))
    lines = [CONSOLIDATION_PROMPT, ""]
    for key, r in zip(keys, suspects):
        lines.append(f"{key} ||| id={r.get('id','?')}  generator={r.get('generator_model','?')}  "
                     f"p_avg_score={r.get('_p_avg_score',0):.4f}")
        lines.append(f"  Spec: {r.get('signature', r.get('prompt',''))[:120]}")
        lines.append(f"  Phase-2 reason: {r.get('reason','')}")
        gen = (r.get("generated") or "").strip()[:300]
        lines.append("  Code:\n    " + gen.replace("\n", "\n    "))
        lines.append("")
    out_tokens = max(2000, len(suspects) * 50)
    print(f"\n  Consolidation pass ({len(suspects)} suspects, max_tokens={out_tokens})...")
    resp = client.chat.completions.create(
        model=reviewer_model,
        messages=[
            {"role": "system", "content":
             "You are a senior Python reviewer doing cross-entry calibration. Be decisive."},
            {"role": "user", "content": "\n".join(lines)},
        ],
        temperature=0,
        max_tokens=out_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()
    raw_verdicts: dict[str, str] = {}
    for line in text.splitlines():
        if "|||" in line:
            key_part, rest = line.split("|||", 1)
            k = key_part.strip()
            if k in key_to_entry:
                raw_verdicts[k] = "CONFIRMED" if "CONFIRMED" in rest.upper() else "DOWNGRADE"
    verdicts: dict[tuple, str] = {}
    unparsed = 0
    for key, r in zip(keys, suspects):
        composite = (r.get("source_file", ""), r.get("id", ""))
        decision  = raw_verdicts.get(key, "CONFIRMED")
        if key not in raw_verdicts:
            unparsed += 1
        verdicts[composite] = decision
    if unparsed:
        print(f"  WARNING: {unparsed}/{len(suspects)} not parsed — defaulted CONFIRMED")
    for key, r in zip(keys, suspects):
        decision = verdicts[(r.get("source_file", ""), r.get("id", ""))]
        print(f"  {decision:<10}  {r.get('id','?')[:28]:<28}  [{r.get('generator_model','?')[:15]}]")
    return verdicts


def load_all_entries() -> list[dict]:
    """Load all entries from production + scenario files, keeping ε data for comparison."""
    entries = []
    for fname in RESULT_FILES:
        fpath = RESULT_DIR / fname
        if not fpath.exists():
            print(f"  SKIP (not found): {fname}")
            continue
        data      = json.loads(fpath.read_text(encoding="utf-8"))
        model     = data.get("meta", {}).get("model", fname)
        benchmark = data.get("meta", {}).get("benchmark", "production")
        for r in data.get("results", []):
            if not r.get("generated"):
                continue
            entries.append({
                "source_file":      fname,
                "generator_model":  model,
                "benchmark":        benchmark,
                "id":               r.get("id",    r.get("name", "?")),
                "label":            r.get("label", r.get("name", "?")),
                "type":             r.get("type", "?"),
                "scenario":         r.get("scenario"),
                "mean_logprob":     r.get("mean_logprob"),
                "epsilon":          r.get("epsilon", 0.0),
                "cascaded_score":   r.get("cascaded_score", 0.0),
                "cascaded_status":  r.get("cascaded_status", "COMPLETE"),
                "trigger_token":    r.get("trigger_token"),
                "evidence":         r.get("evidence", {}),
                "sparse_tokens":    r.get("sparse_tokens", []),
                "generated":        r.get("generated", ""),
                "signature":        r.get("signature", ""),
                "prompt":           r.get("prompt", ""),
                # Ground truth labels
                "eps_label":        r.get("casc_label", r.get("label", "?")),
            })
    return entries


def select_threshold(entries: list[dict]) -> tuple[float, int, float]:
    """
    Find p_avg threshold that best matches ε's recall on API/HIGH entries.
    Returns (threshold, n_flagged, recall_on_api).
    """
    api_entries = [e for e in entries if e.get("type") in ("API",)]
    logic_entries = [e for e in entries if e.get("type") == "LOGIC"]

    # ε recall: fraction of API entries that ε flagged (cascaded)
    eps_api_flagged = sum(1 for e in api_entries
                          if e.get("cascaded_status") in ("FLAGGED", "PAUSED", "ABORTED"))
    eps_recall = eps_api_flagged / len(api_entries) if api_entries else 0.0
    eps_total_flagged = sum(1 for e in entries
                            if e.get("cascaded_status") in ("FLAGGED", "PAUSED", "ABORTED"))

    print(f"\n  ε baseline: {eps_api_flagged}/{len(api_entries)} API entries flagged "
          f"({eps_recall:.0%} recall), {eps_total_flagged} total flagged")

    # Entries with mean_logprob available
    has_lp = [e for e in entries if e.get("mean_logprob") is not None]
    missing = len(entries) - len(has_lp)
    if missing:
        print(f"  WARNING: {missing} entries missing mean_logprob — excluded from threshold sweep")

    print(f"\n  p_avg threshold sweep (target recall ≈ {eps_recall:.0%}):")
    print(f"  {'Threshold':>10}  {'API recall':>10}  {'Total flagged':>13}  {'Logic FP':>9}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*13}  {'-'*9}")

    best_threshold = P_AVG_THRESHOLDS[0]
    best_delta = float("inf")

    for t in P_AVG_THRESHOLDS:
        api_flagged = sum(1 for e in has_lp
                         if e.get("type") == "API" and
                         e.get("mean_logprob") is not None and
                         (-e["mean_logprob"]) >= t)
        total_flagged = sum(1 for e in has_lp
                           if e.get("mean_logprob") is not None and
                           (-e["mean_logprob"]) >= t)
        logic_flagged = sum(1 for e in has_lp
                           if e.get("type") == "LOGIC" and
                           e.get("mean_logprob") is not None and
                           (-e["mean_logprob"]) >= t)
        recall = api_flagged / len([e for e in has_lp if e.get("type") == "API"]) \
                 if any(e.get("type") == "API" for e in has_lp) else 0.0
        delta = abs(recall - eps_recall)
        marker = " *" if delta < best_delta else "  "
        if delta < best_delta:
            best_delta = delta
            best_threshold = t
        print(f"  -{t:>9.2f}  {api_flagged:>4}/{len([e for e in has_lp if e.get('type')=='API']):<4}"
              f" ({recall:>5.0%})  {total_flagged:>6}/{len(has_lp):<6}  "
              f"{logic_flagged:>4}/{len([e for e in has_lp if e.get('type')=='LOGIC']):<4}{marker}")

    print(f"\n  Selected threshold: -{best_threshold:.2f}  (closest to ε recall of {eps_recall:.0%})")
    return best_threshold, eps_total_flagged, eps_recall


def apply_p_avg_flag(entries: list[dict], threshold: float) -> list[dict]:
    """Return only entries that p_avg flags at the given threshold."""
    flagged = []
    for e in entries:
        ml = e.get("mean_logprob")
        if ml is None:
            continue
        neg_ml = -ml
        if neg_ml >= threshold:
            entry = dict(e)
            entry["_p_avg_score"] = round(neg_ml, 4)
            flagged.append(entry)
    return flagged


def print_comparison(eps_n: int, eps_precision: float,
                     pavg_n: int, pavg_precision: float,
                     threshold: float) -> None:
    print(f"\n{'='*70}")
    print("ε vs p_avg — END-TO-END REVIEW LOOP COMPARISON")
    print("=" * 70)
    print(f"\n  Signal         {'Flagged':>8}  {'Precision':>10}  {'Notes'}")
    print(f"  {'-'*14} {'-'*8}  {'-'*10}  {'-'*30}")
    print(f"  ε (cascaded)   {eps_n:>8}  {eps_precision:>9.1%}  token attribution + AST filter")
    print(f"  p_avg (>{threshold:.2f})  {pavg_n:>8}  {pavg_precision:>9.1%}  function-level only, no token detail")
    delta = pavg_precision - eps_precision
    print(f"\n  Precision delta: {delta:+.1%}  "
          f"({'p_avg better' if delta > 0 else 'ε better' if delta < 0 else 'equal'})")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="p_avg baseline review loop — compares function-level vs token-level flagging"
    )
    parser.add_argument("--reviewer", default="gpt-4o-mini",
                        help="Reviewer model (default: gpt-4o-mini)")
    parser.add_argument("--workers",  type=int, default=10)
    parser.add_argument("--output",   default="p_avg_review_results.json")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  p_avg BASELINE -- FULL REVIEW LOOP")
    print("=" * 70)

    entries = load_all_entries()
    print(f"\n  Loaded {len(entries)} total entries from {len(RESULT_FILES)} files")

    has_lp = sum(1 for e in entries if e.get("mean_logprob") is not None)
    print(f"  Entries with mean_logprob: {has_lp}/{len(entries)}")
    if has_lp == 0:
        print("\n  ERROR: No entries have mean_logprob. Re-run the benchmark scripts:")
        for f in RESULT_FILES[:2]:
            model = f.replace("production_", "").replace("scenarios_", "").replace(".json", "")
            print(f"    python benchmark_production.py --model {model}")
        sys.exit(1)

    # Threshold selection
    threshold, eps_total_flagged, eps_recall = select_threshold(entries)

    # Apply p_avg flagging
    flagged = apply_p_avg_flag(entries, threshold)
    api_flagged = sum(1 for e in flagged if e.get("type") == "API")
    logic_flagged = sum(1 for e in flagged if e.get("type") == "LOGIC")

    print(f"\n  p_avg flags {len(flagged)} entries at threshold -{threshold:.2f}:")
    print(f"    API/HIGH : {api_flagged}")
    print(f"    LOGIC/LOW: {logic_flagged}")
    print(f"    Naïve precision (pre-review): "
          f"{api_flagged/len(flagged)*100:.1f}%  "
          f"(ε naïve was ~27% — see paper §5)")

    if not flagged:
        print("\n  No entries flagged at this threshold. Try a lower threshold.")
        sys.exit(1)

    client = OpenAI()

    # Review pass (WITHOUT token localization)
    results = run_review_phase(client, args.reviewer, flagged, args.workers)

    # Consolidation pass
    suspects = [r for r in results if r.get("verdict") == "SUSPECT"]
    consolidation: dict[tuple, str] = {}
    if suspects:
        consolidation = consolidation_pass(client, args.reviewer, suspects)
    else:
        print("  No SUSPECT entries — skipping consolidation.")

    # Precision calculation
    confirmed = sum(1 for v in consolidation.values() if v == "CONFIRMED")
    downgraded = sum(1 for v in consolidation.values() if v == "DOWNGRADE")
    v_counts = {"CLEARED": 0, "UNCERTAIN": 0, "SUSPECT": 0}
    for r in results:
        v_counts[r.get("verdict", "UNCERTAIN")] += 1

    true_fail = confirmed
    false_pos_caught = downgraded
    # SUSPECT entries that were CONFIRMED = genuine errors caught
    # CLEARED + downgraded SUSPECT = non-issues = FPs
    total_reviewed = len(results)
    total_fp = total_reviewed - true_fail - (len(suspects) - confirmed - downgraded)
    pavg_precision = true_fail / total_reviewed if total_reviewed else 0.0

    print(f"\n{'='*70}")
    print("p_avg REVIEW LOOP SUMMARY")
    print("=" * 70)
    print(f"\n  {total_reviewed} entries reviewed (no token localization in prompt)")
    print(f"  Cleared immediately : {v_counts['CLEARED']:>3}  ({v_counts['CLEARED']/total_reviewed*100:.1f}%)")
    print(f"  Uncertain           : {v_counts['UNCERTAIN']:>3}  ({v_counts['UNCERTAIN']/total_reviewed*100:.1f}%)")
    print(f"  Forwarded to consol.: {len(suspects):>3}")
    print(f"  Confirmed failures  : {confirmed:>3}  ({confirmed/total_reviewed*100:.1f}%)  ← genuine errors")
    print(f"  FP caught late      : {false_pos_caught:>3}")

    # Published ε numbers (from paper)
    eps_precision_published = 0.94
    print_comparison(
        eps_total_flagged, eps_precision_published,
        total_reviewed, pavg_precision,
        threshold,
    )

    out = RESULT_DIR / args.output
    out.write_text(json.dumps({
        "meta": {
            "reviewer_model":  args.reviewer,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "p_avg_threshold": threshold,
            "n_reviewed":      total_reviewed,
            "v_counts":        v_counts,
            "confirmed":       confirmed,
            "downgraded":      downgraded,
            "pavg_precision":  round(pavg_precision, 4),
            "eps_precision":   eps_precision_published,
            "eps_n_flagged":   eps_total_flagged,
            "consolidation":   {f"{sf}::{eid}": v
                                for (sf, eid), v in consolidation.items()},
        },
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results -> {out}")


if __name__ == "__main__":
    main()
