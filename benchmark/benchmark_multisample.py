#!/usr/bin/env python3
"""
Multi-Sample Diversity Benchmark
=================================
Implements the Sharma & David (2025) baseline: generate N independent samples
at temperature > 0 and measure behavioral diversity (API/library choice) as an
uncertainty proxy.

For each prompt:
  - Generate N=5 completions at T=0.7
  - Extract which API/library/algorithm variant each sample chose
  - Diversity = 1 - max_count / N  (0 = unanimous, 0.8 = all different)
  - Fire if diversity >= threshold

Uses the SAME system prompt as EpsilonWrapper for a fair comparison.
No logprobs needed — this is a pure sampling approach.

Models: gpt-4o, gpt-4o-mini, gpt-4-turbo (OpenAI), DeepSeek V3 (Together AI)

Usage:
    cd d:/Language/repo
    python benchmark/benchmark_multisample.py
    python benchmark/benchmark_multisample.py --model gpt-4o-mini
    python benchmark/benchmark_multisample.py --model deepseek-ai/DeepSeek-V3 --provider together
    python benchmark/benchmark_multisample.py --all-models

Results saved to benchmark/results/multisample_{model_slug}.json
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_env = Path(__file__).parent.parent / ".env"
if not _env.exists():
    _env = Path(__file__).parent.parent.parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.  pip install openai")
    sys.exit(1)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_SAMPLES = 5
TEMPERATURE = 0.7
DIVERSITY_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]

# Same system prompt as EpsilonWrapper for fair comparison
SYSTEM_PROMPT = (
    "You are an expert software engineer. "
    "Respond with raw Python code only — no markdown fences, "
    "no explanation, no prose. Output only the function(s) requested. "
    "Use concise conventional parameter names (n for integers, s for strings, "
    "lst for lists, d for dicts, f for floats). "
    "Name functions using the most direct verb-noun form from the prompt."
)

ALL_MODELS = [
    ("gpt-4o",                     "openai"),
    ("gpt-4o-mini",                "openai"),
    ("gpt-4-turbo",                "openai"),
    ("deepseek-ai/DeepSeek-V3",    "together"),
]

# ------------------------------------------------------------------ #
# Prompt sets (same as benchmark_calibration.py)
# ------------------------------------------------------------------ #

LOW_PROMPTS = [
    {"id": "low_01", "label": "Sort integers",
     "prompt": "Write a Python function that takes a list of integers and returns it sorted in ascending order.",
     "context": "Python 3.11"},
    {"id": "low_02", "label": "Palindrome check",
     "prompt": "Write a Python function that returns True if a given string is a palindrome, False otherwise. Ignore case.",
     "context": "Python 3.11"},
    {"id": "low_03", "label": "Fibonacci (nth)",
     "prompt": "Write a Python function that returns the nth Fibonacci number. n is a non-negative integer.",
     "context": "Python 3.11"},
    {"id": "low_04", "label": "List maximum",
     "prompt": "Write a Python function that returns the maximum value in a list of numbers without using the built-in max().",
     "context": "Python 3.11"},
    {"id": "low_05", "label": "Word frequency",
     "prompt": "Write a Python function that takes a string and returns a dictionary mapping each word to its frequency.",
     "context": "Python 3.11"},
    {"id": "low_06", "label": "Prime check",
     "prompt": "Write a Python function that returns True if a given integer is prime, False otherwise.",
     "context": "Python 3.11"},
    {"id": "low_07", "label": "Celsius to Fahrenheit",
     "prompt": "Write a Python function that converts a temperature in Celsius to Fahrenheit.",
     "context": "Python 3.11"},
    {"id": "low_08", "label": "Deduplicate list",
     "prompt": "Write a Python function that removes duplicates from a list while preserving the original order.",
     "context": "Python 3.11"},
    {"id": "low_09", "label": "Flatten one level",
     "prompt": "Write a Python function that flattens a list of lists by one level and returns the result.",
     "context": "Python 3.11"},
    {"id": "low_10", "label": "Binary search",
     "prompt": "Write a Python function that performs binary search on a sorted list and returns the index of the target, or -1 if not found.",
     "context": "Python 3.11"},
]

HIGH_PROMPTS = [
    {"id": "high_01", "label": "Stripe payment (Scenario A)",
     "prompt": "Write a Python function to charge a customer's card for $50 using the Stripe Python SDK in a FastAPI application. The customer_id is passed in as a parameter.",
     "context": "Python 3.11, Stripe SDK installed"},
    {"id": "high_02", "label": "OpenAI SDK call (Scenario B)",
     "prompt": "Write a Python function that calls GPT-4 with a user message and returns the response text.",
     "context": "Python 3.11, openai package installed"},
    {"id": "high_03", "label": "SQLAlchemy query (Scenario C)",
     "prompt": "Write a Python function that queries all active users from a SQLAlchemy database session and returns them as a list. The codebase has been around for a few years.",
     "context": "Python 3.11, SQLAlchemy installed"},
    {"id": "high_04", "label": "FastAPI async HTTP (Scenario D)",
     "prompt": "Write a FastAPI endpoint that accepts a city name as a query parameter, fetches the current weather for that city from an external weather API, and returns the result as JSON.",
     "context": "Python 3.11, FastAPI installed, requests library used throughout the existing codebase"},
    {"id": "high_05", "label": "Password hashing",
     "prompt": "Write a Python function that securely hashes a password for storage in a database and a function to verify it.",
     "context": "Python 3.11, security-sensitive application"},
    {"id": "high_06", "label": "JWT token creation",
     "prompt": "Write a Python function that creates a signed JWT token for a user containing their user_id, with a 1-hour expiry.",
     "context": "Python 3.11, building an authentication service"},
    {"id": "high_07", "label": "Current UTC datetime",
     "prompt": "Write a Python function that returns the current UTC time as a timezone-aware datetime object for storage in a database.",
     "context": "Python 3.11"},
    {"id": "high_08", "label": "HTTP GET to JSON",
     "prompt": "Write a Python function that makes an HTTP GET request to a given URL and returns the response body as a parsed JSON dict.",
     "context": "Python 3.11"},
    {"id": "high_09", "label": "Redis cache get",
     "prompt": "Write a Python function that retrieves a cached value from Redis by key, returning None if the key does not exist.",
     "context": "Python 3.11, Redis server running, async FastAPI application"},
    {"id": "high_10", "label": "PostgreSQL query",
     "prompt": "Write a Python function that connects to a PostgreSQL database and retrieves all rows from a 'users' table.",
     "context": "Python 3.11"},
    {"id": "high_11", "label": "Pydantic model serialization",
     "prompt": "Write a Python function that takes a Pydantic model instance representing a user (with id, name, and email fields) and returns it as a plain dictionary.",
     "context": "Python 3.11, Pydantic installed"},
    {"id": "high_12", "label": "pandas DataFrame concat",
     "prompt": "Write a Python function that takes a list of pandas DataFrames and concatenates them into a single DataFrame, dropping duplicate rows.",
     "context": "Python 3.11, pandas installed, existing codebase started 3 years ago"},
    {"id": "high_13", "label": "boto3 S3 upload",
     "prompt": "Write a Python function that uploads a local file to an S3 bucket using boto3 and returns the S3 object key.",
     "context": "Python 3.11, boto3 installed, AWS credentials configured"},
    {"id": "high_14", "label": "Celery async task",
     "prompt": "Write a Celery task that sends a welcome email to a new user. The task should accept a user_id and email address as arguments.",
     "context": "Python 3.11, Celery with Redis broker, Django project"},
    {"id": "high_15", "label": "Token expiry datetime",
     "prompt": "Write a Python function that returns a datetime object representing 24 hours from now, suitable for use as a JWT token expiration time.",
     "context": "Python 3.11, building an authentication service"},
    {"id": "high_16", "label": "Pillow image resize",
     "prompt": "Write a Python function that resizes an image to a maximum width of 800 pixels while preserving aspect ratio, using the Pillow library.",
     "context": "Python 3.11, Pillow installed"},
    {"id": "high_17", "label": "Django REST list view",
     "prompt": "Write a Django REST Framework view that returns a paginated list of all active Product objects, supporting filtering by category.",
     "context": "Python 3.11, Django 4.x, Django REST Framework installed"},
    {"id": "high_18", "label": "Async HTTP GET request",
     "prompt": "Write an async Python function that makes an HTTP GET request to a given URL and returns the response body as a parsed JSON dict.",
     "context": "Python 3.11, async FastAPI application"},
    {"id": "high_19", "label": "Schema validation",
     "prompt": "Write a Python function that validates and deserializes incoming JSON request data for a user registration endpoint, checking that name, email, and password fields are present and correctly typed.",
     "context": "Python 3.11, Flask application, no validation library committed yet"},
    {"id": "high_20", "label": "Parametrized test",
     "prompt": "Write a parametrized test for a function that converts temperatures between Celsius and Fahrenheit, covering at least five cases including freezing point, boiling point, and body temperature.",
     "context": "Python 3.11, existing test suite in the project"},
]

# ------------------------------------------------------------------ #
# API choice fingerprints: (label, [patterns_that_identify_this_choice])
# Each entry is a variant; first match wins.
# ------------------------------------------------------------------ #

API_FINGERPRINTS: dict[str, list[tuple[str, list[str]]]] = {
    "high_01": [  # Stripe
        ("PaymentIntent", ["PaymentIntent", "payment_intent"]),
        ("Charge",        ["Charge.create", "stripe.Charge"]),
    ],
    "high_02": [  # OpenAI SDK
        ("sdk_v1",  ["client.chat.completions", ".chat.completions.create", "OpenAI()"]),
        ("sdk_v0",  ["ChatCompletion.create", "openai.ChatCompletion"]),
    ],
    "high_03": [  # SQLAlchemy
        ("v2_select", ["select(", "session.execute", "scalars("]),
        ("v1_query",  [".query(", "session.query"]),
    ],
    "high_04": [  # FastAPI async/sync
        ("async_httpx",    ["async def", "httpx", "await "]),
        ("async_requests", ["async def", "requests.get", "requests.post"]),
        ("sync_requests",  ["def ", "requests.get", "requests.post"]),
        ("sync_httpx",     ["def ", "httpx"]),
    ],
    "high_05": [  # Password hashing
        ("passlib",  ["passlib", "CryptContext", "pwd_context"]),
        ("bcrypt",   ["bcrypt", "bcrypt.hashpw", "bcrypt.checkpw"]),
        ("argon2",   ["argon2", "PasswordHasher"]),
        ("hashlib",  ["hashlib", "sha256", "pbkdf2"]),
        ("werkzeug", ["werkzeug", "generate_password_hash"]),
    ],
    "high_06": [  # JWT
        ("pyjwt",     ["import jwt", "jwt.encode", "jwt.decode"]),
        ("python_jose",["from jose", "jose.jwt", "from jose import"]),
        ("authlib",   ["authlib", "JsonWebToken"]),
        ("itsdangerous", ["itsdangerous", "URLSafeTimedSerializer"]),
    ],
    "high_07": [  # UTC datetime
        ("timezone_aware", ["timezone.utc", "datetime.now(tz", "datetime.now(timezone"]),
        ("utcnow",         ["datetime.utcnow()", "utcnow()"]),
    ],
    "high_08": [  # HTTP GET
        ("requests",  ["requests.get", "import requests"]),
        ("urllib",    ["urllib.request", "urlopen", "urllib.parse"]),
        ("httpx",     ["httpx.get", "import httpx"]),
    ],
    "high_09": [  # Redis
        ("redis_asyncio", ["redis.asyncio", "aioredis", "await redis"]),
        ("redis_sync",    ["redis.Redis", "redis.StrictRedis", "StrictRedis"]),
    ],
    "high_10": [  # PostgreSQL
        ("sqlalchemy", ["sqlalchemy", "create_engine", "Session"]),
        ("psycopg2",   ["psycopg2", "psycopg2.connect"]),
        ("psycopg3",   ["psycopg", "psycopg.connect"]),
        ("asyncpg",    ["asyncpg", "await asyncpg"]),
    ],
    "high_11": [  # Pydantic
        ("model_dump", ["model_dump()", ".model_dump()"]),
        ("dict_v1",    [".dict()", "model.dict()"]),
        ("jsonable",   ["jsonable_encoder"]),
    ],
    "high_12": [  # pandas concat
        ("pd_concat",  ["pd.concat", "pandas.concat"]),
        ("df_append",  [".append(", "DataFrame.append"]),
    ],
    "high_13": [  # boto3 S3
        ("upload_file",   ["upload_file(", ".upload_file("]),
        ("put_object",    ["put_object(", ".put_object("]),
        ("upload_fileobj",["upload_fileobj", ".upload_fileobj("]),
    ],
    "high_14": [  # Celery
        ("shared_task", ["@shared_task", "shared_task"]),
        ("app_task",    ["@app.task", "@celery_app.task", "@celery.task"]),
    ],
    "high_15": [  # Token expiry
        ("timezone_aware", ["timezone.utc", "datetime.now(tz", "timedelta", "datetime.now(timezone"]),
        ("utcnow",         ["datetime.utcnow()", "utcnow()"]),
    ],
    "high_16": [  # Pillow resize
        ("resampling_lanczos", ["Resampling.LANCZOS", "Image.Resampling"]),
        ("lanczos_old",        ["LANCZOS", "Image.LANCZOS"]),
        ("antialias",          ["ANTIALIAS", "Image.ANTIALIAS"]),
        ("thumbnail",          [".thumbnail("]),
    ],
    "high_17": [  # Django REST
        ("list_api_view",  ["ListAPIView", "generics.ListAPIView"]),
        ("api_view",       ["@api_view", "APIView"]),
        ("model_viewset",  ["ModelViewSet", "viewsets.ModelViewSet"]),
        ("generic_list",   ["generics.ListCreateAPIView"]),
    ],
    "high_18": [  # Async HTTP GET
        ("httpx",   ["httpx", "async with httpx", "AsyncClient"]),
        ("aiohttp", ["aiohttp", "ClientSession", "aiohttp.ClientSession"]),
        ("asyncio_urllib", ["asyncio", "loop.run_in_executor"]),
    ],
    "high_19": [  # Schema validation
        ("pydantic",     ["pydantic", "BaseModel", "model.model_validate", "UserRegistration"]),
        ("marshmallow",  ["marshmallow", "Schema()", "fields.Str"]),
        ("cerberus",     ["cerberus", "Validator("]),
        ("manual",       ["if 'name' not in", "if not data.get", "isinstance(data"]),
        ("wtforms",      ["wtforms", "StringField"]),
    ],
    "high_20": [  # Parametrized test
        ("pytest_parametrize", ["pytest.mark.parametrize", "@pytest.mark"]),
        ("unittest",           ["unittest.TestCase", "class Test"]),
        ("ddt",                ["@ddt", "from ddt"]),
    ],
}

# LOW prompt algorithm variants to track diversity
LOW_FINGERPRINTS: dict[str, list[tuple[str, list[str]]]] = {
    "low_01": [  # Sort
        ("builtin_sorted", ["return sorted("]),
        ("list_sort",      [".sort(", "lst.sort"]),
        ("custom_loop",    ["for i in range", "bubble", "merge", "insertion"]),
    ],
    "low_02": [  # Palindrome
        ("slice",          ["[::-1]", "== s[::-1]"]),
        ("two_pointer",    ["left", "right", "while left < right"]),
        ("reversed",       ["reversed(", "== list(reversed"]),
    ],
    "low_03": [  # Fibonacci
        ("iterative",      ["a, b", "while n", "for i in range"]),
        ("recursive",      ["fib(n-1)", "fibonacci(n - 1)", "return n if"]),
        ("memoized",       ["lru_cache", "memo", "@cache"]),
    ],
    "low_04": [  # Max
        ("loop",           ["max_val", "current", "for num in"]),
        ("reduce",         ["reduce(", "functools.reduce"]),
    ],
    "low_05": [  # Word frequency
        ("counter",        ["Counter(", "from collections import Counter"]),
        ("dict_loop",      ["freq = {}", "freq[word]", "setdefault"]),
        ("defaultdict",    ["defaultdict(", "from collections import defaultdict"]),
    ],
    "low_06": [  # Prime
        ("trial_division", ["range(2", "% i == 0", "sqrt"]),
        ("optimized",      ["range(2, int(", "math.sqrt", "i * i <= n"]),
    ],
    "low_07": [  # Celsius
        ("formula",        ["* 9 / 5 + 32", "* 9/5 + 32", "/ 5 * 9 + 32"]),
    ],
    "low_08": [  # Dedup
        ("seen_set",       ["seen = set()", "seen = []", "not in seen"]),
        ("dict_fromkeys",  ["dict.fromkeys", "list(dict.fromkeys"]),
        ("ordered_dict",   ["OrderedDict", "from collections"]),
    ],
    "low_09": [  # Flatten
        ("comprehension",  ["item for sublist", "for item in sublist"]),
        ("chain",          ["itertools.chain", "chain.from_iterable"]),
        ("extend_loop",    ["result.extend(", ".extend("]),
    ],
    "low_10": [  # Binary search
        ("iterative",      ["while low <= high", "mid = (low + high)"]),
        ("recursive",      ["binary_search(", "def _search"]),
        ("builtin",        ["bisect", "bisect_left"]),
    ],
}


# ------------------------------------------------------------------ #
# Fingerprint extraction
# ------------------------------------------------------------------ #

def extract_choice(code: str, prompt_id: str) -> str:
    """Return the first matching variant label for this code snippet."""
    fingerprints = (
        API_FINGERPRINTS.get(prompt_id) or LOW_FINGERPRINTS.get(prompt_id) or []
    )
    for label, patterns in fingerprints:
        # For multi-pattern variants (like async_requests), ALL patterns must match
        if len(patterns) > 1 and prompt_id == "high_04":
            if all(p in code for p in patterns):
                return label
        else:
            if any(p in code for p in patterns):
                return label
    return "unknown"


def compute_diversity(choices: list[str]) -> float:
    """1 - max_frequency / N.  0 = unanimous, higher = more diverse."""
    if not choices:
        return 0.0
    counts: dict[str, int] = {}
    for c in choices:
        counts[c] = counts.get(c, 0) + 1
    max_count = max(counts.values())
    return round(1.0 - max_count / len(choices), 4)


# ------------------------------------------------------------------ #
# API calls
# ------------------------------------------------------------------ #

def get_client(model: str, provider: str) -> OpenAI:
    if provider == "together":
        key = os.environ.get("TOGETHER_API_KEY", "")
        if not key:
            raise RuntimeError("TOGETHER_API_KEY not set in environment")
        return OpenAI(api_key=key, base_url="https://api.together.xyz/v1")
    return OpenAI()


def generate_sample(client: OpenAI, model: str, prompt: str, context: str) -> str:
    """Generate one sample at T=0.7, no logprobs."""
    system = SYSTEM_PROMPT
    if context:
        system = f"{system}\n\nContext: {context}"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=512,
        logprobs=False,
    )
    return response.choices[0].message.content or ""


def run_prompt(client: OpenAI, model: str, entry: dict, n: int = N_SAMPLES) -> dict:
    """Generate N samples for a single prompt entry, extract choices, compute diversity."""
    samples = []
    for _ in range(n):
        try:
            code = generate_sample(client, model, entry["prompt"], entry.get("context", ""))
        except Exception as e:
            code = f"ERROR: {e}"
        samples.append(code)
        time.sleep(0.1)  # gentle rate limiting

    choices = [extract_choice(s, entry["id"]) for s in samples]
    diversity = compute_diversity(choices)

    return {
        "id":        entry["id"],
        "label":     entry["label"],
        "category":  "HIGH" if entry["id"].startswith("high") else "LOW",
        "prompt":    entry["prompt"][:80],
        "choices":   choices,
        "diversity": diversity,
        "samples":   [s[:200] for s in samples],  # store truncated for debugging
    }


# ------------------------------------------------------------------ #
# Threshold sweep + reporting
# ------------------------------------------------------------------ #

def sweep_thresholds(results: list[dict]) -> list[dict]:
    low  = [r for r in results if r["category"] == "LOW"]
    high = [r for r in results if r["category"] == "HIGH"]
    rows = []
    for t in DIVERSITY_THRESHOLDS:
        fp  = sum(1 for r in low  if r["diversity"] > t)
        det = sum(1 for r in high if r["diversity"] > t)
        rows.append({
            "threshold": t,
            "detection": det, "n_high": len(high),
            "det_rate":  det / len(high) if high else 0.0,
            "fp": fp, "n_low": len(low),
            "fp_rate": fp / len(low) if low else 0.0,
        })
    return rows


def print_results(model: str, results: list[dict], sweep: list[dict]) -> None:
    low  = [r for r in results if r["category"] == "LOW"]
    high = [r for r in results if r["category"] == "HIGH"]

    print(f"\n{'━'*60}")
    print(f"  MODEL: {model}")
    print(f"{'━'*60}")

    # Per-prompt diversity table
    print(f"\n  {'ID':<9}  {'Cat':4}  {'Div':>5}  {'Choices'}")
    print(f"  {'-'*9}  {'-'*4}  {'-'*5}  --------")
    for r in results:
        choices_str = ", ".join(f"{c}({r['choices'].count(c)})" for c in sorted(set(r['choices'])))
        div_flag = " <-- diverse" if r["diversity"] > 0.0 else ""
        print(f"  {r['id']:<9}  {r['category']:4}  {r['diversity']:>5.2f}  {choices_str}{div_flag}")

    # Threshold sweep
    print(f"\n  Threshold sweep:")
    print(f"  {'threshold':>10}  {'det':>12}  {'FP':>10}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}")
    for row in sweep:
        det_str = f"{row['detection']}/{row['n_high']} ({row['det_rate']:.0%})"
        fp_str  = f"{row['fp']}/{row['n_low']} ({row['fp_rate']:.0%})"
        print(f"  {row['threshold']:>10.1f}  {det_str:>12}  {fp_str:>10}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def run_model(model: str, provider: str, n_samples: int = N_SAMPLES) -> dict:
    print(f"\n  Running {model} ({provider}) — {n_samples} samples each, T={TEMPERATURE}")
    client = get_client(model, provider)

    all_prompts = LOW_PROMPTS + HIGH_PROMPTS
    results = []
    for i, entry in enumerate(all_prompts, 1):
        cat = "LOW " if entry["id"].startswith("low") else "HIGH"
        print(f"  [{i:02d}/{len(all_prompts)}] {cat} — {entry['label']} ... ", end="", flush=True)
        rec = run_prompt(client, model, entry, n_samples)
        print(f"div={rec['diversity']:.2f}  choices={set(rec['choices'])}")
        results.append(rec)

    sweep = sweep_thresholds(results)
    print_results(model, results, sweep)

    model_slug = model.replace("/", "_").replace("-", "_")
    out_path = RESULTS_DIR / f"multisample_{model_slug}.json"
    payload = {
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "model":       model,
        "provider":    provider,
        "n_samples":   N_SAMPLES,
        "temperature": TEMPERATURE,
        "results":     results,
        "sweep":       sweep,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")
    return payload


def print_cross_model_summary(all_payloads: list[dict]) -> None:
    print("\n" + "━" * 70)
    print("  MULTI-SAMPLE COMPARISON SUMMARY  (diversity > 0.0 threshold)")
    print("━" * 70)
    print(f"  {'Model':<30}  {'HIGH det':>12}  {'LOW FP':>10}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*10}")
    for payload in all_payloads:
        # Find threshold closest to ε operating point (~50% FP)
        # Use diversity > 0 as the most permissive threshold
        sweep = payload["sweep"]
        for row in sweep:
            if row["threshold"] == 0.0:
                det_str = f"{row['detection']}/{row['n_high']} ({row['det_rate']:.0%})"
                fp_str  = f"{row['fp']}/{row['n_low']} ({row['fp_rate']:.0%})"
                print(f"  {payload['model']:<30}  {det_str:>12}  {fp_str:>10}")
                break
    print()
    print("  Note: diversity > 0.0 means any disagreement across samples fires.")
    print("  Compare with ε at 50% FP operating point in analyze_p_avg.py output.")


def main():
    parser = argparse.ArgumentParser(description="Multi-sample diversity benchmark")
    parser.add_argument("--model",     default="gpt-4o-mini")
    parser.add_argument("--provider",  default="openai", choices=["openai", "together"])
    parser.add_argument("--all-models", action="store_true",
                        help="Run all four models sequentially")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    args = parser.parse_args()

    n_samples = args.n_samples

    if args.all_models:
        payloads = []
        for model, provider in ALL_MODELS:
            payload = run_model(model, provider, n_samples)
            payloads.append(payload)
        print_cross_model_summary(payloads)
    else:
        payload = run_model(args.model, args.provider, n_samples)
        print_cross_model_summary([payload])


if __name__ == "__main__":
    main()
