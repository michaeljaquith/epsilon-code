#!/usr/bin/env python3
"""
LOW Ground Truth Harness
=========================
Exec()s each generated LOW function from scenarios_*.json and runs
assertions to get binary pass/fail ground truth.

All 10 LOW prompts are pure logic — no external libraries. All 30
generations (10 prompts x 3 models) should pass. Any that fail are
genuine model errors despite being simple functions.

Output: results/ground_truth_low.json

Usage:
    python ground_truth_low.py
"""
import json
import re
import traceback
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
MODELS = [
    ("gpt-4o",                 "scenarios_gpt-4o.json"),
    ("gpt-4o-mini",            "scenarios_gpt-4o-mini.json"),
    ("deepseek-ai/DeepSeek-V3","scenarios_deepseek-ai_DeepSeek-V3.json"),
]

# ---------------------------------------------------------------------------
# Test suite — one entry per LOW prompt id
# Each entry: list of (args, expected) tuples, or a callable(fn)->None that
# raises AssertionError on failure.
# ---------------------------------------------------------------------------

def _run_tests(fn, cases):
    """Run (args, expected) or callable cases. Raises on first failure."""
    for case in cases:
        if callable(case):
            case(fn)
        else:
            args, expected = case
            got = fn(*args)
            assert got == expected, f"fn({', '.join(repr(a) for a in args)}) -> {got!r}, expected {expected!r}"


LOW_TESTS = {
    "low_01": [
        (([3, 1, 2],),        [1, 2, 3]),
        (([],),               []),
        (([5],),              [5]),
        (([2, 2, 1],),        [1, 2, 2]),
        (([-3, 0, -1],),      [-3, -1, 0]),
    ],
    "low_02": [
        (("racecar",),   True),
        (("hello",),     False),
        (("Racecar",),   True),
        (("A",),         True),
        (("",),          True),
        (("AbBa",),      True),
    ],
    "low_03": [
        ((0,),  0),
        ((1,),  1),
        ((2,),  1),
        ((7,),  13),
        ((10,), 55),
    ],
    "low_04": [
        # "without using the built-in max()" — tests correctness only
        (([3, 1, 4, 1, 5, 9],), 9),
        (([-1, -5, -2],),       -1),
        (([42],),               42),
        (([0, 0, 0],),          0),
    ],
    "low_05": [
        lambda fn: None if fn("") == {} else (_ for _ in ()).throw(
            AssertionError(f'fn("") -> {fn("")!r}, expected {{}}')),
        lambda fn: _assert_word_freq(fn, "the cat sat", {"the": 1, "cat": 1, "sat": 1}),
        lambda fn: _assert_word_freq(fn, "the the cat", {"the": 2, "cat": 1}),
        lambda fn: _assert_word_freq(fn, "a a a", {"a": 3}),
    ],
    "low_06": [
        ((2,),  True),
        ((7,),  True),
        ((13,), True),
        ((1,),  False),
        ((4,),  False),
        ((0,),  False),
    ],
    "low_07": [
        lambda fn: _assert_close(fn(0),    32.0,   "0C"),
        lambda fn: _assert_close(fn(100),  212.0,  "100C"),
        lambda fn: _assert_close(fn(-40),  -40.0,  "-40C"),
        lambda fn: _assert_close(fn(37),   98.6,   "37C"),
    ],
    "low_08": [
        (([1, 2, 1, 3, 2],), [1, 2, 3]),
        (([],),              []),
        (([1, 1, 1],),       [1]),
        (([3, 1, 2],),       [3, 1, 2]),
        (([4, 5, 4, 6, 5],), [4, 5, 6]),
    ],
    "low_09": [
        (([[1, 2], [3, 4]],),          [1, 2, 3, 4]),
        (([[]],),                       []),
        (([[1], [], [2]],),             [1, 2]),
        (([[1, [2]], [3]],),            [1, [2], 3]),   # 1-level only
    ],
    "low_10": [
        (([1, 3, 5, 7], 3),   1),
        (([1, 3, 5, 7], 1),   0),
        (([1, 3, 5, 7], 7),   3),
        (([1, 3, 5, 7], 4),  -1),
        (([],              5), -1),
    ],
}


def _assert_word_freq(fn, text, expected):
    got = fn(text)
    for word, count in expected.items():
        assert got.get(word) == count, (
            f'fn({text!r}): expected {word!r}={count}, got {got.get(word)!r}')


def _assert_close(got, expected, label, tol=0.01):
    assert abs(got - expected) <= tol, (
        f'fn({label}): got {got}, expected {expected} (tol={tol})')


# ---------------------------------------------------------------------------
# Exec harness
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"^```(?:python)?\s*\n?(.*?)\n?```\s*$", re.DOTALL)


def strip_fences(code: str) -> str:
    m = FENCE_RE.match(code.strip())
    return m.group(1) if m else code


def find_function(namespace: dict, preferred_names: list[str]) -> object | None:
    """Return the best callable from an exec'd namespace."""
    for name in preferred_names:
        if name in namespace and callable(namespace[name]):
            return namespace[name]
    # fallback: first callable that isn't a class or module
    for name, val in namespace.items():
        if callable(val) and not name.startswith("_") and not isinstance(val, type):
            return val
    return None


PREFERRED_NAMES = {
    "low_01": ["sort_list", "sort_integers", "sort_numbers", "sort_ints"],
    "low_02": ["is_palindrome", "palindrome", "check_palindrome"],
    "low_03": ["fibonacci", "fib", "nth_fibonacci", "get_fibonacci"],
    "low_04": ["find_max", "list_max", "max_value", "maximum", "find_maximum"],
    "low_05": ["word_frequency", "word_freq", "count_words", "word_count"],
    "low_06": ["is_prime", "prime", "check_prime"],
    "low_07": ["celsius_to_fahrenheit", "c_to_f", "convert", "to_fahrenheit"],
    "low_08": ["remove_duplicates", "deduplicate", "unique", "dedup"],
    "low_09": ["flatten", "flatten_one_level", "flatten_list"],
    "low_10": ["binary_search", "bsearch", "search"],
}


def run_entry(low_id: str, code: str) -> tuple[bool, str | None]:
    """Return (passed, error_message). error_message is None on success."""
    code = strip_fences(code.strip())
    if not code:
        return False, "empty generated code"
    namespace: dict = {}
    try:
        exec(compile(code, "<generated>", "exec"), namespace)
    except Exception as e:
        return False, f"exec error: {e}"
    fn = find_function(namespace, PREFERRED_NAMES.get(low_id, []))
    if fn is None:
        return False, "no callable found in generated code"
    tests = LOW_TESTS.get(low_id)
    if tests is None:
        return False, f"no tests defined for {low_id}"
    try:
        _run_tests(fn, tests)
        return True, None
    except AssertionError as e:
        return False, str(e)
    except Exception as e:
        return False, f"runtime error: {e}\n{traceback.format_exc(limit=3)}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    entries = []
    all_pass = 0
    all_fail = 0

    print(f"{'Model':<30} {'ID':<10} {'Status':<10} {'Score':>7}  {'Result'}")
    print("-" * 80)

    for model_name, fname in MODELS:
        path = RESULTS_DIR / fname
        if not path.exists():
            print(f"  MISSING: {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        results = data.get("results", data) if isinstance(data, dict) else data
        low_entries = [r for r in results if isinstance(r, dict) and r.get("id", "").startswith("low_")]

        for r in low_entries:
            low_id    = r["id"]
            code      = r.get("generated", "")
            casc_stat = r.get("cascaded_status", "UNKNOWN")
            casc_sc   = r.get("cascaded_score",  0.0)
            passed, err = run_entry(low_id, code)
            result_str  = "PASS" if passed else f"FAIL: {err}"
            if passed:
                all_pass += 1
            else:
                all_fail += 1
            print(f"  {model_name:<28} {low_id:<10} {casc_stat:<10} {casc_sc:7.3f}  {result_str}")
            entries.append({
                "model":           model_name,
                "id":              low_id,
                "label":          r.get("label", ""),
                "cascaded_status": casc_stat,
                "cascaded_score":  round(casc_sc, 4),
                "pass":            passed,
                "error":           err,
                "generated_snippet": code[:120],
            })

    print("-" * 80)
    print(f"  Total: {all_pass} pass, {all_fail} fail  ({all_pass}/{all_pass+all_fail})")
    print()

    # Tier breakdown
    from collections import defaultdict
    tier_pass   = defaultdict(int)
    tier_total  = defaultdict(int)
    for e in entries:
        t = e["cascaded_status"]
        tier_total[t] += 1
        if e["pass"]:
            tier_pass[t] += 1

    print("Pass rate by epsilon tier (LOW functions only):")
    for tier in ["COMPLETE", "FLAGGED", "PAUSED", "ABORTED"]:
        total = tier_total.get(tier, 0)
        passed = tier_pass.get(tier, 0)
        if total:
            print(f"  {tier:<10}  {passed}/{total}  ({passed/total*100:.0f}% pass)")

    out = {
        "meta": {
            "description": "Ground truth pass/fail for all 30 LOW scenario generations",
            "total": all_pass + all_fail,
            "pass": all_pass,
            "fail": all_fail,
        },
        "entries": entries,
    }
    out_path = RESULTS_DIR / "ground_truth_low.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
