#!/usr/bin/env python3
"""
Demo Scenario E — Complete Authentication Module
================================================
Scenarios A–D each present one consequential API decision per run. Scenario E
presents four simultaneous decisions that are also interdependent:

  Password hashing:   bcrypt / argon2 / passlib / hashlib.pbkdf2_hmac
  JWT library:        PyJWT / python-jose / authlib
  SQLAlchemy style:   1.x (.query) / 2.0 (.execute + select)
  Datetime handling:  datetime.utcnow() (deprecated) / datetime.now(timezone.utc)

Each decision is made token-by-token with no global reasoning about cross-
function compatibility. The JWT library committed in login() must be used
consistently in verify_token() and refresh_token() — the model does not
guarantee this.

This is also the statistical anchor scenario: because it generates ~400-700
ε-contributing code tokens per run (vs ~50-150 for A-D), it reaches the MAD
adaptive domain after ~8 similar runs and the conformal domain after ~50.
The first run uses absolute thresholds — that is expected and correct.

Usage:
    python demo_scenario_e.py

    Options:
      --no-color    Plain-text output
      --token-map   Show full token-level ε map
      --dry-run     Print prompt only, no API call
      --model       OpenAI model name (default: gpt-4o)
"""
import argparse
import ast
import math
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


# ------------------------------------------------------------------ #
# Prompt
# ------------------------------------------------------------------ #

DEMO_PROMPT = """\
Write a complete Python authentication module with the following six functions:

  register_user(db, username, password, email)
    Create a new user with a securely hashed password. Raise an appropriate
    error if the username or email already exists.

  login(db, username, password)
    Verify credentials against the stored hash. Return a signed JWT access
    token on success; raise an appropriate error on failure.

  verify_token(token)
    Decode and validate a JWT. Return the payload dict. Raise an error if
    the token is expired or its signature is invalid.

  refresh_token(token)
    If the token is valid and within the refresh window, issue a new token
    with a reset expiry. Raise an error if the token cannot be refreshed.

  request_password_reset(db, email)
    Generate a short-lived signed reset token and return it. Assume the
    caller handles delivery. Raise an error if the email is not found.

  reset_password(db, reset_token, new_password)
    Validate the reset token and update the user's stored password hash.
    Raise an error if the token is expired or already used.\
"""

DEMO_CONTEXT = (
    "Python 3.11, FastAPI backend, PostgreSQL database accessed via SQLAlchemy. "
    "No existing authentication library has been committed to in this codebase."
)

THRESHOLDS = {
    "threshold_soft":   0.30,
    "threshold_hard":   0.65,
    "threshold_abort":  0.95,
}

# ------------------------------------------------------------------ #
# Library / style fingerprints for consistency analysis
# ------------------------------------------------------------------ #

JWT_LIBS = {
    # python-jose checked first: "from jose" is unambiguous.
    # PyJWT's "jwt.encode" pattern also matches python-jose (which aliases the
    # submodule as jwt), so specificity order matters here.
    "python-jose": [
        "from jose", "jose import", "jose.jwt", "JWTError",
        "from jose.exceptions",
    ],
    "PyJWT": [
        "import jwt", "from jwt",
        "jwt.ExpiredSignatureError", "jwt.InvalidTokenError",
        "jwt.exceptions",
        "jwt.encode", "jwt.decode",   # checked after jose to avoid false match
    ],
    "authlib": [
        "from authlib", "authlib.jose", "JsonWebToken", "authlib.integrations",
    ],
}

PASSWORD_LIBS = {
    "bcrypt": [
        "import bcrypt", "from bcrypt", "bcrypt.hashpw", "bcrypt.checkpw",
        "bcrypt.gensalt", "bcrypt.hash(",
    ],
    "argon2": [
        "from argon2", "import argon2", "PasswordHasher", "argon2.verify",
        "ph.verify", "ph.hash",
    ],
    "passlib": [
        "CryptContext", "from passlib", "import passlib",
        "pwd_context", "passlib.hash",
    ],
    "hashlib (pbkdf2)": [
        "pbkdf2_hmac", "hashlib.sha256", "hashlib.sha512",
        "hmac.compare_digest",
    ],
}

ORM_STYLES = {
    "1.x  (.query)": [
        "db.query(", "session.query(", ".query(User",
        ".filter(", ".filter_by(",
    ],
    "2.0  (.execute+select)": [
        "db.execute(select", "session.execute(select",
        ".execute(select(", ".scalars()", "scalars().all()",
    ],
}

DATETIME_STYLES = {
    "deprecated  (utcnow)": [
        "datetime.utcnow()", ".utcnow()",
    ],
    "correct  (timezone-aware)": [
        "datetime.now(timezone", "now(tz=timezone",
        "timezone.utc)", "datetime.now(tz",
    ],
}

# Which functions are relevant per decision
JWT_FUNCTIONS      = ["login", "verify_token", "refresh_token",
                      "request_password_reset", "reset_password"]
PASSWORD_FUNCTIONS = ["register_user", "login"]
ORM_FUNCTIONS      = ["register_user", "request_password_reset",
                      "reset_password", "login"]
DATETIME_FUNCTIONS = ["login", "verify_token", "refresh_token",
                      "request_password_reset"]


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

# ORM signal tokens used to detect db-touching functions dynamically
_ORM_TOUCH_SIGNALS = [
    "db.query(", "session.query(", ".query(", "db.execute(", "session.execute(",
    ".filter(", ".filter_by(", ".scalars()", "select(",
]


def _db_touching_functions(func_bodies: dict[str, str], primary: list[str]) -> list[str]:
    """Return primary functions plus any generated helpers that contain ORM patterns.

    When the model refactors db queries into helper functions (e.g.
    get_user_by_email), those helpers are the actual locus of the ORM style
    decision. Including them prevents a false 'unclear' result for every
    primary function that delegates to a helper.
    """
    extras = [
        fn for fn in func_bodies
        if fn not in primary
        and any(sig in func_bodies[fn] for sig in _ORM_TOUCH_SIGNALS)
    ]
    return primary + extras


def print_separator(label: str = "", width: int = 62) -> None:
    if label:
        pad = max(0, (width - len(label) - 2) // 2)
        print(f"{'━' * pad} {label} {'━' * (width - pad - len(label) - 2)}")
    else:
        print("━" * width)


def _extract_function_bodies(code: str) -> dict[str, str]:
    """Return {func_name: source_text} for each function defined in code.

    Module-level import lines are prepended to every function body so that
    library detection can distinguish aliases like 'from jose import jwt'
    (python-jose) from 'import jwt' (PyJWT) even when both call jwt.encode()
    inside their function bodies.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    lines = code.split("\n")

    # Collect top-level import lines only (col_offset == 0 = module scope)
    import_lines: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end = getattr(node, "end_lineno", node.lineno)
            import_lines.extend(lines[node.lineno - 1 : end])
    module_imports = "\n".join(import_lines)

    bodies: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            func_text = "\n".join(lines[node.lineno - 1 : end])
            bodies[node.name] = (
                module_imports + "\n" + func_text if module_imports else func_text
            )
    return bodies


def _detect(text: str, signals: dict[str, list[str]]) -> str:
    """Return first matching library/style name, or 'unclear'."""
    text_lower = text.lower()
    for name, patterns in signals.items():
        if any(p.lower() in text_lower for p in patterns):
            return name
    return "unclear"


def _consistency_block(
    label: str,
    func_bodies: dict[str, str],
    signals: dict[str, list[str]],
    func_names: list[str],
    col_w: int = 28,
) -> list[str]:
    """Return display lines for one consistency check.

    Only includes functions that were actually generated (body is non-empty).
    Reports INCONSISTENCY when multiple different non-unclear values appear.
    """
    rows = [
        (fn, _detect(func_bodies[fn], signals))
        for fn in func_names
        if fn in func_bodies and func_bodies[fn].strip()
    ]
    if not rows:
        return []

    non_unclear = [v for _, v in rows if v != "unclear"]
    has_conflict = len(set(non_unclear)) > 1

    lines = [f"  {label}"]
    first_val = non_unclear[0] if non_unclear else None
    for fn, det in rows:
        flag = ""
        if det == "unclear":
            flag = "  ?"
        elif has_conflict and first_val and det != first_val:
            flag = "  ← INCONSISTENCY"
        lines.append(f"    {fn:<{col_w}}  {det}{flag}")

    if has_conflict:
        lines.append("    *** cross-function inconsistency — may cause runtime errors ***")
    lines.append("")
    return lines


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Demo Scenario E: Complete authentication module"
    )
    parser.add_argument("--no-color",     action="store_true")
    parser.add_argument("--token-map",    action="store_true")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--debug-tokens", action="store_true",
                        help="Show token classification breakdown (why n_code_tokens is what it is)")
    parser.add_argument("--model",        default="gpt-4o")
    args = parser.parse_args()

    print()
    print_separator("ε DEMO — Complete Authentication Module")
    print()
    print("  Six functions. Four simultaneous version-split decisions.")
    print("  Each decision made token-by-token — no cross-function reasoning.")
    print()
    print(f"  Thresholds: soft={THRESHOLDS['threshold_soft']} | "
          f"hard={THRESHOLDS['threshold_hard']} | "
          f"abort={THRESHOLDS['threshold_abort']}")
    print(f"  Model:      {args.model}")
    print()

    if args.dry_run:
        print("Prompt:")
        for line in DEMO_PROMPT.split("\n"):
            print(f"  {line}")
        print("\nContext:")
        print(f"  {DEMO_CONTEXT}")
        return

    # ------------------------------------------------------------------ #
    # Generate
    # ------------------------------------------------------------------ #
    client  = OpenAI()
    wrapper = EpsilonWrapper(client, config=THRESHOLDS, log_path="epsilon_session.log")

    print("Running: generate_code() with logprobs=True, top_logprobs=5 ...")
    print("         Embedding call starts in parallel — zero added latency.")
    print()

    result = wrapper.generate_code(
        prompt=DEMO_PROMPT,
        context=DEMO_CONTEXT,
        model=args.model,
    )

    # ------------------------------------------------------------------ #
    # Standard ε output
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
    # Per-function ε breakdown
    # ------------------------------------------------------------------ #
    print()
    print_separator("Per-function ε breakdown")
    print()
    if result.epsilon_by_func:
        expected = ["register_user", "login", "verify_token",
                    "refresh_token", "request_password_reset", "reset_password"]
        # Show expected order first, then any unexpected functions
        ordered = [(fn, result.epsilon_by_func[fn])
                   for fn in expected if fn in result.epsilon_by_func]
        ordered += [(fn, eps) for fn, eps in result.epsilon_by_func.items()
                    if fn not in expected]

        col_w = max(len(fn) for fn, _ in ordered) if ordered else 20
        for fn, eps in ordered:
            bar    = "█" * int(eps * 28) + "░" * (28 - int(eps * 28))
            status = ""
            if eps > THRESHOLDS["threshold_hard"]:
                status = "  PAUSED"
            elif eps > THRESHOLDS["threshold_soft"]:
                status = "  FLAGGED"
            else:
                status = "  ok"
            print(f"  {fn:<{col_w}}  ε={eps:.3f}  {bar}{status}")

        missing = [fn for fn in expected if fn not in result.epsilon_by_func]
        if missing:
            print()
            print(f"  Not generated: {', '.join(missing)}")
    else:
        print("  (no function boundaries detected in generated code)")
    print()

    # ------------------------------------------------------------------ #
    # Cross-function consistency analysis
    # ------------------------------------------------------------------ #
    func_bodies = _extract_function_bodies(result.code)

    print_separator("Cross-function consistency")
    print()

    all_lines: list[str] = []
    all_lines += _consistency_block(
        "Password hashing  (register_user → login must match)",
        func_bodies, PASSWORD_LIBS, PASSWORD_FUNCTIONS,
    )
    all_lines += _consistency_block(
        "JWT library  (login / verify_token / refresh_token must match)",
        func_bodies, JWT_LIBS, JWT_FUNCTIONS,
    )
    orm_funcs = _db_touching_functions(func_bodies, ORM_FUNCTIONS)
    orm_lines = _consistency_block(
        "SQLAlchemy style  (consistent across all db-touching functions)",
        func_bodies, ORM_STYLES, orm_funcs,
    )
    # When every checked function returns 'unclear', the model likely used raw SQL
    # via db.execute("SELECT ...") or delegated queries to helpers not in the scan.
    # "unclear" here means "not ORM style 1.x or 2.0" — which is itself a finding.
    if orm_lines and all("unclear" in ln for ln in orm_lines if ln.strip() and not ln.startswith("  S")):
        orm_lines.insert(-1,
            "    (all unclear — model may have used raw SQL strings via db.execute(); "
            "check for db.execute(\"SELECT...\") pattern)")
    all_lines += orm_lines
    all_lines += _consistency_block(
        "Datetime handling  (token creation and expiry check must agree)",
        func_bodies, DATETIME_STYLES, DATETIME_FUNCTIONS,
    )

    if all_lines:
        for line in all_lines:
            print(line)
    else:
        print("  (could not parse function bodies — syntax error in generated code?)")
        print()

    # ------------------------------------------------------------------ #
    # Token budget and statistical domain
    # ------------------------------------------------------------------ #
    print_separator("Token budget & statistical domain")
    print()
    print(f"  completion_tokens  {result.completion_tokens:>6}")
    print(f"  n_code_tokens      {result.n_code_tokens:>6}  "
          f"(ε-contributing code tokens; drives K-NN token budget)")
    print()

    if result.ensemble_threshold is not None:
        print(f"  Adaptive threshold : {result.ensemble_threshold:.4f}  "
              f"(trigger: {result.trigger})")
        print("  Statistical domain : MAD or conformal — neighborhood populated")
    else:
        print("  Adaptive threshold : none")
        print("  Statistical domain : cold start — neighborhood not yet built")
        print()
        tokens = max(result.n_code_tokens, 1)
        cfg    = wrapper.config
        mad_runs  = max(cfg["knn_min_n_mad"],
                        math.ceil(cfg["knn_cold_start_tokens"] / tokens))
        conf_runs = max(cfg["knn_min_n_conformal"],
                        math.ceil(cfg["knn_conformal_tokens"]  / tokens))
        print(f"  With {tokens} code tokens per run and k={cfg['knn_k']} neighbors:")
        print(f"    MAD domain       after ~{mad_runs:>3} similar runs  "
              f"(needs ≥{cfg['knn_cold_start_tokens']} neighborhood tokens "
              f"+ ≥{cfg['knn_min_n_mad']} runs)")
        print(f"    Conformal domain after ~{conf_runs:>3} similar runs  "
              f"(needs ≥{cfg['knn_conformal_tokens']} neighborhood tokens "
              f"+ ≥{cfg['knn_min_n_conformal']} runs)")
        print()
        print("  Each run of this scenario writes an embedding to epsilon_session.log.")
        print("  Subsequent similar runs find it as a neighbor and contribute to")
        print("  the token budget. Run this scenario repeatedly to observe the")
        print("  cold start → MAD → conformal transition.")
    print()

    # ------------------------------------------------------------------ #
    # Token classification debug
    # ------------------------------------------------------------------ #
    if args.debug_tokens:
        floor = THRESHOLDS.get("accumulation_floor", 0.30)
        all_te = result.token_epsilons

        n_total      = len(all_te)
        n_non_code   = sum(1 for te in all_te if not te.is_code_token)
        n_code_all   = sum(1 for te in all_te if te.is_code_token)
        n_code_below = sum(1 for te in all_te if te.is_code_token and te.epsilon <= floor)
        n_code_above = sum(1 for te in all_te if te.is_code_token and te.epsilon > floor)

        print_separator("Token classification breakdown")
        print()
        print(f"  Total tokens          {n_total:>5}")
        print(f"  Non-code (filtered)   {n_non_code:>5}  "
              f"({100*n_non_code/n_total:.0f}%)  whitespace, comments, name declarations")
        print(f"  Code tokens           {n_code_all:>5}  ({100*n_code_all/n_total:.0f}%)")
        print(f"    ε ≤ {floor:.2f} (below floor)  {n_code_below:>5}  "
              f"({100*n_code_below/n_total:.0f}%)  model is certain — not ε-contributing")
        print(f"    ε >  {floor:.2f} (above floor)  {n_code_above:>5}  "
              f"({100*n_code_above/n_total:.0f}%)  ← n_code_tokens")
        print()

        # ε distribution of all code tokens
        buckets = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30),
                   (0.30, 0.40), (0.40, 0.50), (0.50, 0.65), (0.65, 1.01)]
        print("  ε distribution across all code tokens:")
        for lo, hi in buckets:
            count = sum(1 for te in all_te if te.is_code_token and lo <= te.epsilon < hi)
            bar   = "█" * count if count < 40 else "█" * 39 + "+"
            mark  = "  ← floor" if abs(lo - floor) < 0.01 else ""
            print(f"    [{lo:.2f}–{hi:.2f})  {count:>4}  {bar}{mark}")
        print()

        # Show the top-20 non-code tokens by ε to reveal what's being filtered
        excluded_high = sorted(
            [te for te in all_te if not te.is_code_token and te.epsilon > 0.20],
            key=lambda t: t.epsilon, reverse=True
        )[:10]
        if excluded_high:
            print("  Highest-ε filtered-out tokens (non-code):")
            for te in excluded_high:
                if not te.token.strip():
                    reason = "whitespace"
                else:
                    reason = "comment" if te.token.strip().startswith("#") else "name declaration"
                print(f"    line {te.line:>3}  {repr(te.token):20}  ε={te.epsilon:.3f}  [{reason}]")
            print()

    # ------------------------------------------------------------------ #
    # Cost
    # ------------------------------------------------------------------ #
    cost_in  = result.prompt_tokens     * 2.50 / 1_000_000
    cost_out = result.completion_tokens * 10.0  / 1_000_000
    print(
        f"Cost estimate: ${cost_in + cost_out:.4f}  "
        f"({result.prompt_tokens} in + {result.completion_tokens} out tokens)"
    )
    print("Log entry + embedding written to: epsilon_session.log")
    print()


if __name__ == "__main__":
    main()
