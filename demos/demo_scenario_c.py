#!/usr/bin/env python3
"""
Demo Scenario C — SQLAlchemy 1.4 vs 2.0 Session API
=====================================================
SQLAlchemy 2.0 changed the core query pattern:

  1.x:  session.query(User).filter(User.active == True).all()
  2.0:  session.execute(select(User).where(User.active == True)).scalars().all()

SQLAlchemy 1.4 was a transition release that supported BOTH syntaxes,
creating a long period where both patterns were valid and widely published.
The model has seen both heavily in training data. At the token decision
point — .query( vs .execute(select( — it has no context about whether
the target codebase is 1.x or 2.x.

Unlike Scenario B (confident wrongness), this should produce genuine
token-level uncertainty — completing the three-scenario picture.

Usage:
    python demo_scenario_c.py

    Options:
      --no-color    Plain-text output
      --token-map   Show full token-level ε map
      --dry-run     Print prompt only
      --model       OpenAI model name (default: gpt-4o)
"""
import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.  pip install openai")
    sys.exit(1)

from epsilon.core     import EpsilonWrapper
from epsilon.renderer import render_result, render_token_map


DEMO_PROMPT = (
    "Write a Python function that queries all active users from a "
    "SQLAlchemy database session and returns them as a list. "
    "The codebase has been around for a few years."
)

DEMO_CONTEXT = "Python 3.11, SQLAlchemy installed"

THRESHOLDS = {
    "threshold_soft":   0.30,
    "threshold_hard":   0.65,
    "threshold_abort":  0.95,
}

# Tokens associated with each SQLAlchemy API version — used in signal summary
V1_SIGNALS = ["query", ".query", "filter", ".filter"]
V2_SIGNALS = ["execute", "select", "scalars", ".scalars", "where", ".where"]


def print_separator(label: str = "") -> None:
    if label:
        pad = max(0, (55 - len(label) - 2) // 2)
        print(f"{'━' * pad} {label} {'━' * (55 - pad - len(label) - 2)}")
    else:
        print("━" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Demo Scenario C: SQLAlchemy 1.4 vs 2.0 query API"
    )
    parser.add_argument("--no-color",  action="store_true")
    parser.add_argument("--token-map", action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--model",     default="gpt-4o")
    args = parser.parse_args()

    print()
    print_separator("ε DEMO — SQLAlchemy 1.4 vs 2.0 Query API")
    print()
    print("The claim: token probabilities at the session method call are split")
    print("between 1.x (.query) and 2.0 (.execute + select) syntax.")
    print()
    print(f"Thresholds: soft={THRESHOLDS['threshold_soft']} | "
          f"hard={THRESHOLDS['threshold_hard']} | "
          f"abort={THRESHOLDS['threshold_abort']}")
    print(f"Model:      {args.model}")
    print()

    if args.dry_run:
        print("Prompt:")
        print(f"  {DEMO_PROMPT}")
        print("\nContext:")
        print(f"  {DEMO_CONTEXT}")
        return

    client  = OpenAI()
    wrapper = EpsilonWrapper(client, config=THRESHOLDS, log_path="epsilon_session.log")

    print("Running: generate_code() with logprobs=True, top_logprobs=5 ...")
    print()

    result = wrapper.generate_code(
        prompt=DEMO_PROMPT,
        context=DEMO_CONTEXT,
        model=args.model,
    )

    if args.no_color:
        from epsilon.renderer import _render_plain
        _render_plain(result)
    else:
        render_result(result)

    if args.token_map:
        print()
        print_separator("Token map  (ε ≥ 0.30)")
        render_token_map(result, min_epsilon=0.30)

    # SQLAlchemy version signal summary
    v1_hits = [
        te for te in result.token_epsilons
        if any(te.token.lower().strip() == s for s in V1_SIGNALS)
    ]
    v2_hits = [
        te for te in result.token_epsilons
        if any(te.token.lower().strip() == s for s in V2_SIGNALS)
    ]

    api_chosen = "1.x (.query)" if v1_hits else "2.0 (.execute/select)" if v2_hits else "unclear"

    print()
    print_separator("SQLAlchemy version signal")
    print()
    print(f"  API chosen: {api_chosen}")
    print()

    # Show alternatives at query/execute decision tokens
    decision_tokens = [
        te for te in result.token_epsilons
        if te.is_code_token
        and any(s in te.token.lower() for s in
                ["query", "execute", "select", "filter", "where", "scalar"])
        and te.epsilon > 0.20
    ]
    if decision_tokens:
        for te in sorted(decision_tokens, key=lambda t: t.epsilon, reverse=True)[:4]:
            print(f"  Token: {repr(te.token):22}  line {te.line}  ε={te.epsilon:.3f}")
            for tok, prob in te.top_alternatives[:3]:
                v_tag = ""
                if any(s in tok.lower() for s in V1_SIGNALS):
                    v_tag = "  ← v1.x"
                elif any(s in tok.lower() for s in V2_SIGNALS):
                    v_tag = "  ← v2.0"
                print(f"    {tok:30}  P={prob:.3f}{v_tag}")
        print()
    else:
        print("  No query/execute tokens found above ε threshold.")
        print()

    print()
    cost_in  = result.prompt_tokens     * 2.50 / 1_000_000
    cost_out = result.completion_tokens * 10.0  / 1_000_000
    print(
        f"Cost estimate: ${cost_in + cost_out:.5f}  "
        f"({result.prompt_tokens} in + {result.completion_tokens} out tokens)"
    )
    print("Log entry written to: epsilon_session.log")
    print()


if __name__ == "__main__":
    main()
