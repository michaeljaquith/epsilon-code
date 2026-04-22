#!/usr/bin/env python3
"""
Scenario E Collapse Experiment
===============================
Demonstrates ε collapse in combined multi-function generation vs. per-function.

Combined: all 6 auth functions in one prompt → model commits to a library in
function 1; functions 3–6 show ε collapse as uncertainty is already resolved.

Per-function: load peak ε from existing scenarios_*.json results where each
function was generated independently, preserving full library-choice uncertainty.

Output: results/scenario_e_collapse.json + console comparison table.

Usage:
    python benchmark_scenario_e_collapse.py
    python benchmark_scenario_e_collapse.py --model gpt-4o-mini
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent

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

RESULTS_DIR = Path(__file__).parent / "results"
K = 5

# ---------------------------------------------------------------------------
# Scenario E function signatures (mirrored from benchmark_scenarios.py)
# ---------------------------------------------------------------------------
_SCE_E_CTX = "from sqlalchemy.orm import Session\nSECRET_KEY = 'your-secret-key'  # loaded from env"

SCENARIO_E_PROMPTS = [
    {
        "id": "sce_e1", "fn": "register_user",
        "signature": dedent("""\
            def register_user(session: Session, email: str, password: str):
                \"\"\"Create a new user account. Hash the password and persist the user record.
                Raise an exception if the email is already registered. Return the user.\"\"\"
        """),
    },
    {
        "id": "sce_e2", "fn": "login",
        "signature": dedent("""\
            def login(session: Session, email: str, password: str) -> str:
                \"\"\"Verify credentials. Return a signed JWT access token (1-hour expiry).
                Raise an exception if credentials are invalid.\"\"\"
        """),
    },
    {
        "id": "sce_e3", "fn": "verify_token",
        "signature": dedent("""\
            def verify_token(token: str) -> dict | None:
                \"\"\"Decode and validate a JWT access token.
                Return the payload dict on success, or None if invalid or expired.\"\"\"
        """),
    },
    {
        "id": "sce_e4", "fn": "refresh_token",
        "signature": dedent("""\
            def refresh_token(token: str) -> str:
                \"\"\"Validate the given JWT and issue a new token with refreshed 1-hour expiry.
                Raise an exception if the token is invalid or expired.\"\"\"
        """),
    },
    {
        "id": "sce_e5", "fn": "request_password_reset",
        "signature": dedent("""\
            def request_password_reset(session: Session, email: str) -> str:
                \"\"\"Generate a signed password reset token (1-hour expiry).
                Return empty string if email not found (prevent enumeration).\"\"\"
        """),
    },
    {
        "id": "sce_e6", "fn": "reset_password",
        "signature": dedent("""\
            def reset_password(session: Session, reset_token: str, new_password: str) -> bool:
                \"\"\"Validate the reset token, look up the user, and update their hashed password.
                Return True on success. Raise an exception on invalid token or missing user.\"\"\"
        """),
    },
]

COMBINED_SYSTEM = (
    "You are a Python expert. Output ONLY Python code "
    "(no explanation, no markdown fences, no preamble). "
    "Implement all functions. Use a consistent authentication library throughout."
)

COMBINED_HEADER = dedent("""\
    Implement a complete Python authentication module.
    Stack: Python 3.11, FastAPI backend, PostgreSQL via SQLAlchemy.
    No authentication library is committed — choose one consistently across all functions.

    Relevant imports available:
    from sqlalchemy.orm import Session
    SECRET_KEY = 'your-secret-key'  # loaded from env

    Implement all of the following functions in order. Output ONLY Python code.
""")


def build_combined_prompt() -> str:
    parts = [COMBINED_HEADER]
    for p in SCENARIO_E_PROMPTS:
        parts.append(p["signature"].rstrip() + "\n")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ε math
# ---------------------------------------------------------------------------

def token_epsilon(top_logprobs: list) -> float:
    k = len(top_logprobs)
    if k <= 1:
        return 0.0
    probs = [math.exp(lp) for lp in top_logprobs]
    total = sum(probs)
    if total == 0:
        return 0.0
    probs = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return entropy / math.log(k)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_model(client: OpenAI, model: str, prompt: str) -> tuple[list[dict], str]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COMBINED_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=3000,
        logprobs=True,
        top_logprobs=K,
    )
    lp = response.choices[0].logprobs
    tokens = []
    if lp and lp.content:
        for tok in lp.content:
            top_lps  = [alt.logprob for alt in tok.top_logprobs]
            top_toks = [alt.token   for alt in tok.top_logprobs]
            tokens.append({"token": tok.token, "logprob": tok.logprob,
                           "top_logprobs": top_lps, "top_tokens": top_toks})
    return tokens, (response.choices[0].message.content or "")


def get_together_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://api.together.xyz/v1")


# ---------------------------------------------------------------------------
# Split token stream at 'def ' boundaries
# ---------------------------------------------------------------------------

def split_at_def_boundaries(tokens: list[dict]) -> list[list[dict]]:
    """
    Split the token stream at each 'def ' occurrence that starts a new function.
    Returns up to 6 chunks. Scans for 'def' token followed by a space token,
    or a token that is exactly 'def ' or starts with '\ndef'.
    """
    # Build cumulative text to find def positions
    text = ""
    tok_starts = []  # character start position of each token
    for t in tokens:
        tok_starts.append(len(text))
        text += t["token"]

    # Find all 'def ' positions (newline-preceded to avoid nested defs inside classes,
    # but we also take top-level ones)
    import re
    positions = []
    for m in re.finditer(r'(?:^|\n)def ', text):
        char_pos = m.start() if text[m.start()] == 'd' else m.start() + 1
        # find which token index this char_pos falls in
        tok_idx = None
        for i, s in enumerate(tok_starts):
            end = tok_starts[i + 1] if i + 1 < len(tok_starts) else len(text)
            if s <= char_pos < end:
                tok_idx = i
                break
        if tok_idx is not None:
            positions.append(tok_idx)

    if not positions:
        # fallback: treat entire stream as function 1
        return [tokens]

    # Build chunks between consecutive def positions
    chunks = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(tokens)
        chunks.append(tokens[start:end])

    return chunks


def chunk_peak_eps(chunk: list[dict]) -> float:
    if not chunk:
        return 0.0
    eps_vals = [token_epsilon(t["top_logprobs"]) for t in chunk]
    return max(eps_vals) if eps_vals else 0.0


def chunk_mean_eps(chunk: list[dict]) -> float:
    if not chunk:
        return 0.0
    eps_vals = [token_epsilon(t["top_logprobs"]) for t in chunk]
    return sum(eps_vals) / len(eps_vals)


# ---------------------------------------------------------------------------
# Load per-function results from existing scenarios_*.json
# ---------------------------------------------------------------------------

def load_per_function_eps(model_slug: str) -> dict[str, float]:
    """Returns {sce_e1..sce_e6: peak_eps} from existing scenarios results."""
    path = RESULTS_DIR / f"scenarios_{model_slug}.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    results = data.get("results", data) if isinstance(data, dict) else data
    out = {}
    for r in results:
        if isinstance(r, dict) and r.get("scenario") == "E":
            eid = r["id"]
            peak = r.get("evidence", {}).get("peak_eps", r.get("max_eps", 0.0))
            out[eid] = peak
    return out


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

MODELS = [
    {"display": "gpt-4o",                  "slug": "gpt-4o",                       "provider": "openai"},
    {"display": "gpt-4o-mini",             "slug": "gpt-4o-mini",                  "provider": "openai"},
    {"display": "deepseek-ai/DeepSeek-V3", "slug": "deepseek-ai_DeepSeek-V3",      "provider": "together"},
]


def run_experiment(target_model: str | None = None) -> list[dict]:
    models = [m for m in MODELS if target_model is None or m["display"] == target_model]
    if not models:
        print(f"ERROR: model '{target_model}' not in list")
        sys.exit(1)

    combined_prompt = build_combined_prompt()
    all_results = []

    for m in models:
        print(f"\n{'='*60}")
        print(f"Model: {m['display']}")
        print(f"{'='*60}")

        # Client
        if m["provider"] == "together":
            client = get_together_client()
        else:
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

        # Combined call
        print("  Running combined prompt (all 6 functions)...")
        tokens, text = call_model(client, m["display"], combined_prompt)
        print(f"  Generated {len(tokens)} tokens")
        print(f"  Output preview: {repr(text[:120])}")

        # Split into chunks
        chunks = split_at_def_boundaries(tokens)
        print(f"  Detected {len(chunks)} function chunks (expected 6)")

        # Pad/truncate to 6
        while len(chunks) < 6:
            chunks.append([])

        # Per-function ε from existing results
        per_fn_eps = load_per_function_eps(m["slug"])

        fn_results = []
        print(f"\n  {'Fn':<25} {'Combined peak':>14} {'Combined mean':>14} {'Per-fn peak':>12} {'Delta':>8} {'Tokens':>7}")
        print(f"  {'-'*25} {'-'*14} {'-'*14} {'-'*12} {'-'*8} {'-'*7}")

        for i, (p, chunk) in enumerate(zip(SCENARIO_E_PROMPTS, chunks)):
            combined_peak = chunk_peak_eps(chunk)
            combined_mean = chunk_mean_eps(chunk)
            per_peak = per_fn_eps.get(p["id"], None)
            delta = (combined_peak - per_peak) if per_peak is not None else None
            delta_str = f"{delta:+.3f}" if delta is not None else "  N/A"
            per_str = f"{per_peak:.3f}" if per_peak is not None else "   N/A"
            print(f"  {p['fn']:<25} {combined_peak:14.3f} {combined_mean:14.3f} {per_str:>12} {delta_str:>8} {len(chunk):>7}")
            fn_results.append({
                "id": p["id"],
                "fn": p["fn"],
                "combined_peak_eps": round(combined_peak, 4),
                "combined_mean_eps": round(combined_mean, 4),
                "per_fn_peak_eps": round(per_peak, 4) if per_peak is not None else None,
                "delta_peak": round(delta, 4) if delta is not None else None,
                "combined_tokens": len(chunk),
            })

        all_results.append({
            "model": m["display"],
            "total_tokens": len(tokens),
            "chunks_detected": len([c for c in chunks if c]),
            "functions": fn_results,
            "raw_text_preview": text[:500],
        })

    return all_results


def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*70}")
    print("SCENARIO E COLLAPSE SUMMARY -- Combined vs. Per-function Peak eps")
    print(f"{'='*70}")

    for r in results:
        print(f"\nModel: {r['model']}")
        print(f"  {'Function':<28} {'Combined':>10} {'Per-fn':>10} {'Collapse':>10}")
        print(f"  {'-'*28} {'-'*10} {'-'*10} {'-'*10}")
        for f in r["functions"]:
            combined = f["combined_peak_eps"]
            per = f["per_fn_peak_eps"]
            delta = f["delta_peak"]
            per_str = f"{per:.3f}" if per is not None else "    N/A"
            delta_str = f"{delta:+.3f}" if delta is not None else "    N/A"
            print(f"  {f['fn']:<28} {combined:10.3f} {per_str:>10} {delta_str:>10}")

    print()
    print("Interpretation:")
    print("  Combined peak eps should be HIGH for fn1 (library choice), collapsing")
    print("  toward 0 for fn3-fn6 as the model follows its own committed library.")
    print("  Per-fn peak eps should remain HIGH across all functions (independent context).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scenario E collapse experiment")
    parser.add_argument("--model", default=None, help="Run only this model")
    args = parser.parse_args()

    results = run_experiment(target_model=args.model)
    print_summary(results)

    out = {
        "meta": {
            "experiment": "scenario_e_collapse",
            "description": "Combined vs. per-function ε generation; demonstrates ε collapse",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }
    out_path = RESULTS_DIR / "scenario_e_collapse.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
