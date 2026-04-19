#!/usr/bin/env python3
"""
Demo Scenario B — OpenAI SDK v0 vs v1 Syntax
==============================================
OpenAI's Python SDK had a major breaking change in November 2023 (v0 → v1).

  v0:  openai.ChatCompletion.create(model=..., messages=...)
  v1:  openai.chat.completions.create(model=..., messages=...)

Models trained before or during the transition have conflicting training data.
Code using v0 syntax installs cleanly, imports cleanly, then throws:
  AttributeError: module 'openai' has no attribute 'ChatCompletion'
at runtime — on any modern OpenAI SDK installation.

This is self-referential: a tool that detects OpenAI API uncertainty,
demonstrated using the OpenAI API.

Usage:
    export OPENAI_API_KEY="sk-..."   (or use .env file)
    python demo_scenario_b.py

    Options:
      --no-color    Plain-text output
      --token-map   Show full token-level ε map after result
      --dry-run     Print prompt only, no API call
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
    "Write a Python function that calls GPT-4 with a user message "
    "and returns the response text."
)

DEMO_CONTEXT = "Python 3.11, openai package installed"

THRESHOLDS = {
    "threshold_soft":   0.30,
    "threshold_hard":   0.65,
    "threshold_abort":  0.95,
}


def print_separator(label: str = "") -> None:
    if label:
        pad = max(0, (55 - len(label) - 2) // 2)
        print(f"{'━' * pad} {label} {'━' * (55 - pad - len(label) - 2)}")
    else:
        print("━" * 55)


def main():
    parser = argparse.ArgumentParser(description="Demo Scenario B: OpenAI SDK v0 vs v1")
    parser.add_argument("--no-color",  action="store_true", help="Disable rich output")
    parser.add_argument("--token-map", action="store_true", help="Show token-level ε map")
    parser.add_argument("--dry-run",   action="store_true", help="Print prompt only")
    parser.add_argument("--model",     default="gpt-4o",    help="OpenAI model (default: gpt-4o)")
    args = parser.parse_args()

    print()
    print_separator("ε DEMO — OpenAI SDK v0 vs v1")
    print()
    print("The claim: token probabilities at the SDK method chain are split")
    print("between v0 (ChatCompletion.create) and v1 (chat.completions.create).")
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

    print()
    cost_in  = result.prompt_tokens     * 2.50 / 1_000_000
    cost_out = result.completion_tokens * 10.0  / 1_000_000
    print(
        f"Cost estimate: ${cost_in + cost_out:.5f}  "
        f"({result.prompt_tokens} in + {result.completion_tokens} out tokens)"
    )
    print("Log entry written to: epsilon_session.log")
    print()

    # Highlight the specific SDK version signal if present
    sdk_tokens = [
        te for te in result.peak_tokens
        if any(kw in te.token.lower() for kw in
               ["chat", "completion", "create", "openai", "gpt"])
    ]
    if sdk_tokens:
        print_separator("SDK version signal")
        print()
        for te in sdk_tokens:
            print(f"  Token: {repr(te.token):20}  line {te.line}  ε={te.epsilon:.3f}")
            for tok, prob in te.top_alternatives[:3]:
                marker = " ←" if any(
                    kw in tok.lower() for kw in ["chat", "completion", "charge"]
                ) else ""
                print(f"    {tok:30}  P={prob:.3f}{marker}")
        print()


if __name__ == "__main__":
    main()
