#!/usr/bin/env python3
"""
Demo Scenario D — async/sync mismatch in FastAPI (requests vs httpx)
=====================================================================
FastAPI strongly encourages async def endpoints. The requests library is
the most downloaded Python package by a wide margin — models have seen it
in far more examples than httpx or aiohttp.

This creates a two-decision problem at generation time:

  Decision 1 — endpoint type:
    async def get_weather(...)   ← FastAPI default, enables concurrency
    def get_weather(...)         ← FastAPI wraps in thread pool, also fine

  Decision 2 — HTTP client:
    httpx.AsyncClient            ← async-native, correct with async def
    requests.get(...)            ← sync, dominant in training data

  Correct combinations:
    async def  +  httpx          ← fully async
    def        +  requests       ← FastAPI threadpool handles it

  Silent bug:
    async def  +  requests.get() ← blocks the event loop

The mismatch is invisible at test time. It only manifests under concurrent
production load: the sync call holds the event loop for the full duration
of the network request. Everything queued behind it stalls. The symptoms
(cascading timeouts, 503s) look like infrastructure failure, not a
three-line code error.

Unlike Scenarios A/B/C (API version mismatch), this is a concurrency
correctness failure. The code is not deprecated — it works perfectly in
isolation. It fails only when context (async framework) makes the sync
pattern unsafe.

Usage:
    python demo_scenario_d.py

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
    "Write a FastAPI endpoint that accepts a city name as a query parameter, "
    "fetches the current weather for that city from an external weather API, "
    "and returns the result as JSON."
)

DEMO_CONTEXT = "Python 3.11, FastAPI installed, requests library used throughout the existing codebase"

THRESHOLDS = {
    "threshold_soft":   0.30,
    "threshold_hard":   0.65,
    "threshold_abort":  0.95,
}

# Token signals for each pattern — used in signal summary
ASYNC_CLIENT_SIGNALS = ["httpx", "asyncclient", "async_client", "aiohttp", "clientsession"]
SYNC_CLIENT_SIGNALS  = ["requests", "requests.get", "requests.post", "urllib"]
ASYNC_DEF_SIGNALS    = ["async def"]
SYNC_DEF_SIGNALS     = ["def "]


def print_separator(label: str = "") -> None:
    if label:
        pad = max(0, (55 - len(label) - 2) // 2)
        print(f"{'━' * pad} {label} {'━' * (55 - pad - len(label) - 2)}")
    else:
        print("━" * 55)


def _classify_pattern(token_epsilons: list, code: str) -> tuple[str, str]:
    """Return (endpoint_type, http_client) based on generated code content."""
    code_lower = code.lower()

    if any(s in code_lower for s in ASYNC_CLIENT_SIGNALS):
        client = "async (httpx / aiohttp)"
    elif any(s in code_lower for s in SYNC_CLIENT_SIGNALS):
        client = "sync (requests)"
    else:
        client = "unclear"

    if "async def" in code_lower:
        endpoint = "async def"
    else:
        endpoint = "def (sync)"

    return endpoint, client


def main():
    parser = argparse.ArgumentParser(
        description="Demo Scenario D: async/sync mismatch in FastAPI"
    )
    parser.add_argument("--no-color",  action="store_true")
    parser.add_argument("--token-map", action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--model",     default="gpt-4o")
    args = parser.parse_args()

    print()
    print_separator("ε DEMO — FastAPI async/sync mismatch")
    print()
    print("The claim: token probabilities are split at two linked decisions:")
    print("  (1) async def vs def  —  endpoint concurrency model")
    print("  (2) httpx vs requests —  HTTP client library")
    print()
    print("Silent bug: async def + requests.get() blocks the event loop.")
    print("Correct: async def + httpx  OR  def + requests.")
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

    # Async/sync pattern summary
    endpoint_type, http_client = _classify_pattern(result.token_epsilons, result.code)
    is_mismatch = ("async def" in endpoint_type) and ("requests" in http_client)

    print()
    print_separator("async/sync pattern signal")
    print()
    print(f"  Endpoint type: {endpoint_type}")
    print(f"  HTTP client:   {http_client}")

    if is_mismatch:
        print()
        print("  *** MISMATCH DETECTED ***")
        print("  async def endpoint with sync requests.get() blocks the event loop.")
        print("  Passes all unit tests. Fails silently under concurrent load.")
    else:
        print()
        print("  Pattern is internally consistent.")
    print()

    # Show the decision tokens — async/sync and HTTP client choice
    DECISION_KEYWORDS = ["async", "def ", "httpx", "requests", "await", "aiohttp"]
    decision_tokens = [
        te for te in result.token_epsilons
        if te.is_code_token
        and any(kw in te.token.lower() for kw in DECISION_KEYWORDS)
        and te.epsilon > 0.20
    ]
    if decision_tokens:
        print_separator("Decision tokens")
        print()
        for te in sorted(decision_tokens, key=lambda t: t.epsilon, reverse=True)[:5]:
            print(f"  Token: {repr(te.token):22}  line {te.line}  ε={te.epsilon:.3f}")
            for tok, prob in te.top_alternatives[:3]:
                tag = ""
                tok_lower = tok.lower().strip()
                if any(s in tok_lower for s in ASYNC_CLIENT_SIGNALS):
                    tag = "  ← async client"
                elif any(s in tok_lower for s in SYNC_CLIENT_SIGNALS):
                    tag = "  ← sync client"
                elif "async" in tok_lower:
                    tag = "  ← async endpoint"
                elif tok_lower == "def":
                    tag = "  ← sync endpoint"
                print(f"    {tok:30}  P={prob:.3f}{tag}")
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
