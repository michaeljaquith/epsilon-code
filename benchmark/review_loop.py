#!/usr/bin/env python3
"""
Review Loop — Single-pass LLM review with optional model-specific context
=========================================================================
One review pass over all FLAGGED/PAUSED/ABORTED entries, followed by a
consolidation pass (single call, all suspects in view).

Two modes for paper comparison:
  2A (baseline):    sharp prompt, no historical context
  2B (with context): sharp prompt + model-specific learned context

Model context files (results/model_contexts/<model>.json) are built from
the 2A run and accumulate across benchmark runs, making 2B more accurate
over time as the system learns each generator model's failure patterns.

Usage:
    # 2A baseline (also builds context files for 2B):
    python review_loop.py --no-context --output review_results_baseline.json

    # 2B with context:
    python review_loop.py --output review_results_with_context.json

    python review_loop.py --model gpt-4o      # stronger reviewer
    python review_loop.py --dry-run           # show prompts without API calls
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

# Load .env
_env = Path(__file__).parent.parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

RESULT_DIR      = Path(__file__).parent / "results"
MODEL_CTX_DIR   = RESULT_DIR / "model_contexts"

RESULT_FILES = [
    "production_gpt-4o.json",
    "production_gpt-4o-mini.json",
    "production_deepseek-ai_DeepSeek-V3.json",
    "scenarios_gpt-4o.json",
    "scenarios_gpt-4o-mini.json",
    "scenarios_deepseek-ai_DeepSeek-V3.json",
]

REVIEW_TIERS = {"FLAGGED", "PAUSED", "ABORTED"}

# ── Prompts ────────────────────────────────────────────────────────────────────

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
  3. The token where maximum generation uncertainty was detected
  4. The top alternative tokens the model considered at that position

Respond with EXACTLY this format (no other text):
VERDICT: <CLEARED|UNCERTAIN|SUSPECT>
REASON: <one sentence>

CLEARED   = chosen token is correct given the spec and context
UNCERTAIN = correctness cannot be determined without external schema or runtime access
SUSPECT   = a different alternative would produce more correct code

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

CONTEXT_EXTRACTION_PROMPT = f"""\
You are analyzing code review results to extract reusable knowledge about a specific
LLM generator model's failure patterns, for use in future reviews.

You will receive:
  - SUSPECT entries: cases where the generator produced genuinely incorrect code
  - CLEARED entries: cases where generation uncertainty was flagged but code was correct

{DOMAIN_FACTS}

Extract two concise, specific lists a future reviewer can apply to NEW outputs from this model.

Respond with EXACTLY this format:
SUSPECT_PATTERNS:
- [concrete failure pattern observed in this model's output]
- [...]

FALSE_POSITIVE_PATTERNS:
- [thing this model does that looks suspicious but is actually correct]
- [...]

Rules:
  - Be specific and actionable: "Uses openai.ChatCompletion.create (removed in v1.0)"
    not "Uses deprecated APIs"
  - Apply the domain facts above when judging direction — e.g. .model_dump() is CORRECT
    in Pydantic v2; flagging it as a failure would be wrong
  - Only include patterns with at least 2 supporting examples across the entries
  - 2–5 bullets per section; write NONE if no patterns meet the threshold"""

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
    preference (variable names, phrasing, library choice between equals) → DOWNGRADE
  - Entries where the only issue is a known-correct pattern the reviewer flagged → DOWNGRADE
  - An entry that looks ambiguous alone but is clearly wrong given the full set → CONFIRM
  - Any entry matching a domain fact violation (deprecated API, wrong call sig) → always CONFIRM

For each entry output EXACTLY one line:
  ENTRY_ID ||| CONFIRMED — reason
  OR
  ENTRY_ID ||| DOWNGRADE — reason

Use ||| as the delimiter (IDs may contain brackets or colons).
Entries:"""


# ── Prompt builders ────────────────────────────────────────────────────────────

def build_review_prompt(r: dict, model_context: str = "") -> str:
    generated = r.get("generated", "").strip()
    if not generated:
        return ""

    spec     = r.get("signature", r.get("prompt", r.get("label", ""))).strip()
    trigger  = r.get("trigger_token", "?")
    ev       = r.get("evidence", {})
    peak_eps = ev.get("peak_eps", r.get("epsilon", 0.0))
    peak_idx = ev.get("peak_idx", -1)
    window   = ev.get("window_eps", [])

    alts_text = "(not available)"
    for st in r.get("sparse_tokens", []):
        if st.get("idx") == peak_idx:
            lines = []
            for rank, (tok, pct) in enumerate(st.get("alts", []), 1):
                chosen = " <-- CHOSEN" if tok == trigger else ""
                lines.append(f"  {rank}. {repr(tok):<22} {pct:>5.1f}%{chosen}")
            if lines:
                alts_text = "\n".join(lines)
            break

    window_str = "  ".join(f"{e:.3f}" for e in window)
    ctx_block  = f"\n{model_context}\n" if model_context else ""

    return f"""\
FUNCTION SPECIFICATION:
{spec}

GENERATED IMPLEMENTATION:
{generated}

UNCERTAINTY SIGNAL:
  Trigger token  : {repr(trigger)}
  Epsilon        : {peak_eps:.3f}
  Window eps     : {window_str}

TOP ALTERNATIVES AT TRIGGER POSITION:
{alts_text}
{ctx_block}"""


# ── Model context ──────────────────────────────────────────────────────────────

def load_model_context(generator_model: str) -> str:
    """Return FP-only context block for the review prompt, or '' if none.

    Only false-positive patterns are injected into individual reviews.
    Suspect patterns are retained in the context file for inspection but
    are not injected here — the sharpened system prompt and domain facts
    already cover known failure modes, and adding more suspect triggers
    causes over-flagging rather than precision improvement.
    """
    slug     = generator_model.replace("/", "_")
    ctx_path = MODEL_CTX_DIR / f"{slug}.json"
    if not ctx_path.exists():
        return ""
    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
    fp  = [p for p in ctx.get("false_positive_patterns", []) if p and p != "NONE"]
    if not fp:
        return ""

    lines = [f"\nLEARNED FALSE-POSITIVE PATTERNS FOR {generator_model}:",
             "  (things this model does that look suspicious but are correct)"]
    for p in fp:
        lines.append(f"    - {p}")
    return "\n".join(lines)


def _extract_patterns(client: OpenAI, reviewer_model: str,
                      generator_model: str, results: list[dict]) -> dict:
    suspects = [r for r in results if r.get("verdict") == "SUSPECT"]
    cleared  = [r for r in results if r.get("verdict") == "CLEARED"]
    if not suspects and not cleared:
        return {}

    lines = []
    if suspects:
        lines.append(f"SUSPECT entries ({len(suspects)}):")
        for r in suspects[:15]:
            lines.append(f"  [{r['id']}] trigger={repr(r.get('trigger_token','?'))} "
                         f"eps={r.get('epsilon',0):.3f}")
            lines.append(f"    Reason: {r.get('reason','')}")
            gen = (r.get("generated") or "").strip()[:250]
            lines.append(f"    Code snippet: {gen}")
    if cleared:
        lines.append(f"\nCLEARED entries ({len(cleared)}):")
        for r in cleared[:10]:
            lines.append(f"  [{r['id']}] trigger={repr(r.get('trigger_token','?'))} "
                         f"— {r.get('reason','')}")

    resp = client.chat.completions.create(
        model=reviewer_model,
        messages=[
            {"role": "system", "content": CONTEXT_EXTRACTION_PROMPT},
            {"role": "user",   "content": f"Generator: {generator_model}\n\n" + "\n".join(lines)},
        ],
        temperature=0,
        max_tokens=600,
    )
    text    = (resp.choices[0].message.content or "").strip()
    sp, fp  = [], []
    current = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("SUSPECT_PATTERNS:"):
            current = "sp"
        elif line.startswith("FALSE_POSITIVE_PATTERNS:"):
            current = "fp"
        elif line.startswith("- ") and current == "sp":
            sp.append(line[2:].strip())
        elif line.startswith("- ") and current == "fp":
            fp.append(line[2:].strip())

    return {"model": generator_model, "suspect_patterns": sp,
            "false_positive_patterns": fp}


def build_model_contexts(client: OpenAI, reviewer_model: str,
                         results: list[dict]) -> None:
    MODEL_CTX_DIR.mkdir(exist_ok=True)
    by_model: dict[str, list] = {}
    for r in results:
        by_model.setdefault(r.get("generator_model", "unknown"), []).append(r)

    print(f"\n  Building model contexts ({len(by_model)} generator models)...")
    for gen_model, model_results in sorted(by_model.items()):
        new_ctx = _extract_patterns(client, reviewer_model, gen_model, model_results)
        if not new_ctx:
            continue

        slug     = gen_model.replace("/", "_")
        ctx_path = MODEL_CTX_DIR / f"{slug}.json"

        if ctx_path.exists():
            existing     = json.loads(ctx_path.read_text(encoding="utf-8"))
            merged_sp    = sorted(set(existing.get("suspect_patterns", [])) |
                                  set(new_ctx["suspect_patterns"]))
            merged_fp    = sorted(set(existing.get("false_positive_patterns", [])) |
                                  set(new_ctx["false_positive_patterns"]))
            new_ctx["suspect_patterns"]       = merged_sp
            new_ctx["false_positive_patterns"] = merged_fp
            new_ctx["run_count"] = existing.get("run_count", 1) + 1
        else:
            new_ctx["run_count"] = 1

        new_ctx["last_updated"] = datetime.now(timezone.utc).isoformat()
        ctx_path.write_text(
            json.dumps(new_ctx, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        sp = len([p for p in new_ctx.get("suspect_patterns", []) if p != "NONE"])
        fp = len([p for p in new_ctx.get("false_positive_patterns", []) if p != "NONE"])
        print(f"    {gen_model:35s}: {sp} suspect patterns, {fp} FP patterns")


# ── Review runners ─────────────────────────────────────────────────────────────

def _review_one(client: OpenAI, reviewer_model: str, entry: dict,
                use_model_context: bool) -> dict:
    model_ctx = load_model_context(entry["generator_model"]) if use_model_context else ""
    prompt    = build_review_prompt(entry, model_ctx)
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


def run_phase(client: OpenAI, reviewer_model: str, entries: list[dict],
              label: str, workers: int, use_model_context: bool = False) -> list[dict]:
    print(f"\n  {label} ({len(entries)} entries, {workers} workers)...")
    results = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_review_one, client, reviewer_model, e, use_model_context): e
            for e in entries
        }
        for i, fut in enumerate(as_completed(futures), 1):
            entry = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                v = result["verdict"]
                print(f"  [{i:03d}/{len(entries)}] {v:<9}  "
                      f"[{entry['generator_model'][:12]:<12}] "
                      f"{entry['id'][:20]:<20}  {result['reason'][:55]}")
            except Exception as ex:
                results.append({**entry, "verdict": "UNCERTAIN",
                                 "reason": f"error: {ex}",
                                 "review_model": reviewer_model})
                print(f"  [{i:03d}/{len(entries)}] ERROR    {entry['id']}: {ex}")

    return results


def consolidation_pass(client: OpenAI, reviewer_model: str,
                       suspects: list[dict]) -> dict[str, str]:
    """One call over all SUSPECT entries. Returns {id: 'CONFIRMED'|'DOWNGRADE'}."""
    if not suspects:
        return {}

    # Unique key per entry: index-based so IDs shared across models don't collide
    keys = [f"ENTRY_{i:03d}" for i in range(len(suspects))]
    key_to_entry = dict(zip(keys, suspects))

    lines = [CONSOLIDATION_PROMPT, ""]
    for key, r in zip(keys, suspects):
        lines.append(f"{key} ||| id={r['id']}  generator={r['generator_model']}  "
                     f"eps={r.get('epsilon',0):.3f}")
        lines.append(f"  Spec: {r.get('signature', r.get('prompt',''))[:120]}")
        lines.append(f"  Trigger: {repr(r.get('trigger_token','?'))}")
        lines.append(f"  Phase-2 reason: {r.get('reason','')}")
        gen = (r.get("generated") or "").strip()[:300]
        lines.append("  Code:\n    " + gen.replace("\n", "\n    "))
        lines.append("")

    # Budget: ~50 tokens per output line + headroom
    out_tokens = max(2000, len(suspects) * 50)
    print(f"\n  Consolidation pass ({len(suspects)} suspects in single context, "
          f"max_tokens={out_tokens})...")
    resp = client.chat.completions.create(
        model=reviewer_model,
        messages=[
            {"role": "system", "content":
             "You are a senior Python reviewer doing a cross-entry calibration. Be decisive."},
            {"role": "user", "content": "\n".join(lines)},
        ],
        temperature=0,
        max_tokens=out_tokens,
    )
    text = (resp.choices[0].message.content or "").strip()

    # Parse keyed verdicts
    raw_verdicts: dict[str, str] = {}
    for line in text.splitlines():
        if "|||" in line:
            key_part, rest = line.split("|||", 1)
            k = key_part.strip()
            if k in key_to_entry:
                raw_verdicts[k] = "CONFIRMED" if "CONFIRMED" in rest.upper() else "DOWNGRADE"

    # Build final verdicts indexed by (source_file, id); default unparsed to CONFIRMED
    verdicts: dict[tuple, str] = {}
    unparsed = 0
    for key, r in zip(keys, suspects):
        composite = (r["source_file"], r["id"])
        decision  = raw_verdicts.get(key, "CONFIRMED")
        if key not in raw_verdicts:
            unparsed += 1
        verdicts[composite] = decision

    if unparsed:
        print(f"  WARNING: {unparsed}/{len(suspects)} entries not parsed "
              f"— defaulted to CONFIRMED")

    for key, r in zip(keys, suspects):
        decision = verdicts[(r["source_file"], r["id"])]
        print(f"  {decision:<10}  {r['id'][:28]:<28}  [{r['generator_model'][:15]}]")

    return verdicts


# ── Data loading ───────────────────────────────────────────────────────────────

def load_entries() -> list[dict]:
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
            status = r.get("cascaded_status") or r.get("status", "")
            if status not in REVIEW_TIERS or not r.get("generated"):
                continue
            entries.append({
                "source_file":     fname,
                "generator_model": model,
                "benchmark":       benchmark,
                "id":              r.get("id",    r.get("name", "?")),
                "label":           r.get("label", r.get("name", "?")),
                "type":            r.get("type", "?"),
                "scenario":        r.get("scenario"),
                "cascaded_score":  r.get("cascaded_score", 0.0),
                "cascaded_status": status,
                "epsilon":         r.get("epsilon", 0.0),
                "trigger_token":   r.get("trigger_token"),
                "evidence":        r.get("evidence", {}),
                "sparse_tokens":   r.get("sparse_tokens", []),
                "generated":       r.get("generated", ""),
                "signature":       r.get("signature", ""),
                "prompt":          r.get("prompt", ""),
            })
    return entries


# ── Summary ────────────────────────────────────────────────────────────────────

def print_final_summary(results: list[dict],
                        consolidation: dict[tuple, str]) -> None:
    n = len(results)
    v_counts = {"CLEARED": 0, "UNCERTAIN": 0, "SUSPECT": 0}
    for r in results:
        v_counts[r.get("verdict", "UNCERTAIN")] += 1

    confirmed  = sum(1 for v in consolidation.values() if v == "CONFIRMED")
    downgraded = sum(1 for v in consolidation.values() if v == "DOWNGRADE")
    forwarded  = v_counts["SUSPECT"]           # suspects sent to consolidation
    true_fail  = confirmed                     # confirmed genuine failures
    fp_caught  = downgraded                    # false positives caught by consolidation
    cleared    = v_counts["CLEARED"] + fp_caught

    print(f"\n{'='*70}")
    print("FINAL REVIEW SUMMARY")
    print("=" * 70)

    # Pipeline view
    print(f"\n  Pipeline ({n} entries total):")
    print(f"  {'':4}  Review pass")
    print(f"  {'':8}  Cleared immediately : {v_counts['CLEARED']:>3}  "
          f"({v_counts['CLEARED']/n*100:5.1f}%)")
    print(f"  {'':8}  Uncertain           : {v_counts['UNCERTAIN']:>3}  "
          f"({v_counts['UNCERTAIN']/n*100:5.1f}%)")
    print(f"  {'':8}  Forwarded to consolidation: {forwarded:>3}  "
          f"({forwarded/n*100:5.1f}%)")
    print(f"  {'':4}  Consolidation pass  (over all {forwarded} suspects in one call)")
    print(f"  {'':8}  Confirmed failures  : {true_fail:>3}  "
          f"({true_fail/n*100:5.1f}%)  <- genuine errors")
    print(f"  {'':8}  FP caught late      : {fp_caught:>3}  "
          f"({fp_caught/forwarded*100 if forwarded else 0:5.1f}% of forwarded)  "
          f"<- slipped past review, caught here")

    print(f"\n  Net result:")
    print(f"  {'':8}  True failures  : {true_fail:>3} / {n}  ({true_fail/n*100:.1f}%)")
    print(f"  {'':8}  Cleared        : {cleared:>3} / {n}  ({cleared/n*100:.1f}%)")

    # By generator model
    print(f"\n  By generator model:")
    by_model: dict[str, list] = {}
    for r in results:
        by_model.setdefault(r["generator_model"], []).append(r)
    for gm, rs in sorted(by_model.items()):
        susp = sum(1 for r in rs if r.get("verdict") == "SUSPECT")
        conf = sum(1 for r in rs
                   if r.get("verdict") == "SUSPECT"
                   and consolidation.get((r["source_file"], r["id"])) == "CONFIRMED")
        fp_c = susp - conf
        print(f"  {'':4}  {gm:35s}  {len(rs):3d} reviewed  "
              f"{susp:2d} forwarded  {conf:2d} confirmed  {fp_c:2d} FP caught")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM review loop: one review pass over all flagged entries + consolidation.\n"
                    "\nWorkflow for paper comparison:\n"
                    "  Step 1 (baseline):      python review_loop.py --no-context "
                    "--output review_results_baseline.json\n"
                    "  Step 2 (with context):  python review_loop.py "
                    "--output review_results_with_context.json\n"
                    "\nStep 1 builds model context files automatically for use in step 2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",         default="gpt-4o-mini", help="Reviewer model")
    parser.add_argument("--workers",       type=int, default=10)
    parser.add_argument("--no-context",    action="store_true",
                        help="Run without model context (2A baseline). "
                             "Also builds context files for future runs.")
    parser.add_argument("--skip-context",  action="store_true",
                        help="Skip saving model context after this run")
    parser.add_argument("--output",        default="review_results.json",
                        help="Output filename (saved to results/)")
    parser.add_argument("--dry-run",       action="store_true")
    args = parser.parse_args()

    entries = load_entries()
    print(f"\nLoaded {len(entries)} FLAGGED/PAUSED/ABORTED entries with generated text.")
    by_status: dict[str, int] = {}
    for e in entries:
        by_status.setdefault(e["cascaded_status"], 0)
        by_status[e["cascaded_status"]] += 1
    for s, n in sorted(by_status.items()):
        print(f"  {s}: {n}")

    mode_label = "no context (baseline 2A)" if args.no_context else "with model context (2B)"
    print(f"  Mode: {mode_label}")

    if args.dry_run:
        print("\n[DRY RUN — first 3 entries]")
        for e in entries[:3]:
            print(f"\n{'='*60}")
            print(f"[{e['generator_model']}] {e['id']}")
            ctx = load_model_context(e["generator_model"])
            if ctx:
                print(f"  MODEL CONTEXT AVAILABLE: {len(ctx)} chars")
            print(build_review_prompt(e, ctx)[:500])
        print(f"\n... {len(entries)} total entries")
        return

    client = OpenAI()

    # ── Review pass (single pass over all entries) ─────────────────────────────
    label = ("Review pass — sharp prompt, no context"
             if args.no_context else
             "Review pass — sharp prompt + model context")
    results = run_phase(
        client, args.model, entries, label,
        args.workers, use_model_context=not args.no_context,
    )

    # Build/update model context from this run (for future use as 2B input)
    if not args.skip_context:
        build_model_contexts(client, args.model, results)

    # ── Consolidation pass (suspects only, single call) ────────────────────────
    consolidation_verdicts: dict[tuple, str] = {}
    suspects = [r for r in results if r.get("verdict") == "SUSPECT"]
    if suspects:
        consolidation_verdicts = consolidation_pass(client, args.model, suspects)
    else:
        print("  No SUSPECT entries — skipping consolidation.")

    print_final_summary(results, consolidation_verdicts)

    out_path = RESULT_DIR / args.output
    out_path.write_text(
        json.dumps({
            "meta": {
                "model_context_used": not args.no_context,
                "review_model":       args.model,
                "timestamp":          datetime.now(timezone.utc).isoformat(),
                "n_reviewed":         len(results),
                "consolidation": {f"{sf}::{eid}": v
                                  for (sf, eid), v in consolidation_verdicts.items()},
            },
            "results": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n  Results -> {out_path}")


if __name__ == "__main__":
    main()
