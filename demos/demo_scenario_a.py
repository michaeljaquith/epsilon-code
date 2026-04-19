#!/usr/bin/env python3
"""
Demo Scenario A — Stripe API Deprecation
=========================================
Demonstrates that ε (epsilon) reliably fires when GPT-4o generates code
using the deprecated stripe.Charge.create() API, because the model's
token probabilities at "Charge" are genuinely split with "PaymentIntent".

Usage:
    export OPENAI_API_KEY="sk-..."
    python demo_scenario_a.py

    Options:
      --no-color    Plain-text output (no rich library required)
      --token-map   Show the full token-level ε map after the result
      --dry-run     Skip the API call; print the prompt only
"""
import argparse
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is optional; key can also be set as an env variable

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.  pip install openai")
    sys.exit(1)

from epsilon.core     import EpsilonWrapper
from epsilon.renderer import render_result, render_token_map


DEMO_PROMPT = """\
Write a Python function to charge a customer's card for $50 using \
the Stripe Python SDK in a FastAPI application. \
The customer_id is passed in as a parameter.\
"""

DEMO_CONTEXT = "Python 3.11, FastAPI 0.100+, stripe-python latest"

THRESHOLDS = {
    "threshold_soft":   0.30,
    "threshold_hard":   0.65,
    "threshold_abort":  0.95,
}


def print_separator(label: str = "") -> None:
    line = "━" * 55
    if label:
        pad = (55 - len(label) - 2) // 2
        print(f"{'━' * pad} {label} {'━' * pad}")
    else:
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Demo Scenario A: Stripe API deprecation")
    parser.add_argument("--no-color",   action="store_true", help="Disable rich output")
    parser.add_argument("--token-map",  action="store_true", help="Show token-level ε map")
    parser.add_argument("--dry-run",    action="store_true", help="Print prompt only, skip API call")
    parser.add_argument("--model",      default="gpt-4o",    help="OpenAI model name (default: gpt-4o)")
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    print()
    print_separator("ε DEMO — Stripe API Deprecation")
    print()
    t = THRESHOLDS
    print(f"Thresholds: soft={t['threshold_soft']} | hard={t['threshold_hard']} | abort={t['threshold_abort']}")
    print(f"Model:      {args.model}")
    print()

    if args.dry_run:
        print("Prompt:")
        print(f"  {DEMO_PROMPT}")
        print("\nContext:")
        print(f"  {DEMO_CONTEXT}")
        return

    # ------------------------------------------------------------------ #
    # Run the demo
    # ------------------------------------------------------------------ #
    client  = OpenAI()
    wrapper = EpsilonWrapper(client, config=THRESHOLDS, log_path="epsilon_session.log")

    print("Running: generate_code() with logprobs=True, top_logprobs=5 ...")
    print()

    result = wrapper.generate_code(
        prompt=DEMO_PROMPT,
        context=DEMO_CONTEXT,
        model=args.model,
    )

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #
    if args.no_color:
        from epsilon.renderer import _render_plain
        _render_plain(result)
    else:
        render_result(result)

    if args.token_map:
        print()
        print_separator("Token map  (ε ≥ 0.30)")
        render_token_map(result, min_epsilon=0.30)

    # ------------------------------------------------------------------ #
    # Cost estimate
    # ------------------------------------------------------------------ #
    print()
    cost_in  = result.prompt_tokens     * 2.50 / 1_000_000
    cost_out = result.completion_tokens * 10.0  / 1_000_000
    print(
        f"Cost estimate: ${cost_in + cost_out:.5f}  "
        f"({result.prompt_tokens} in + {result.completion_tokens} out tokens)"
    )
    print(f"Log entry written to: epsilon_session.log")
    print()


if __name__ == "__main__":
    main()
