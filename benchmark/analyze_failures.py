#!/usr/bin/env python3
"""
Deep token-level analysis of failure cases from benchmark_production.py.
Re-runs the specific failing functions and prints the full token probability
sequence, highlighting high-epsilon tokens and the surrounding context.
"""
import math, os, sys
from pathlib import Path
from textwrap import dedent

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai"); sys.exit(1)

# Load .env
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

K = 5
DELTA = 0.30

CONTEXT_HEADER = dedent("""\
    You are implementing a Python function for a production FastAPI web application.
    Stack: FastAPI, SQLModel (ORM over PostgreSQL), JWT authentication (PyJWT),
    pwdlib for password hashing, Jinja2 for email templates, emails library for SMTP.

    Key types available:
      Session/SessionDep, CurrentUser, User, Item, UserCreate/Update/Register/UpdateMe,
      ItemCreate/Update, UserPublic/ItemPublic/UsersPublic/ItemsPublic, Token, Message,
      NewPassword, UpdatePassword, EmailData, settings (with SECRET_KEY, SMTP_*, etc.)
      security.ALGORITHM = "HS256", password_hash = PasswordHash((Argon2Hasher(), BcryptHasher()))

    Implement ONLY the function body. Do not repeat the signature or imports.
""")

FAILURE_CASES = [
    {
        "id": "get_password_hash",
        "label": "consensus FN — eps=0.000 all 3 models",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "context": "from pwdlib import PasswordHash\nfrom pwdlib.hashers.argon2 import Argon2Hasher\nfrom pwdlib.hashers.bcrypt import BcryptHasher\npassword_hash = PasswordHash((Argon2Hasher(), BcryptHasher()))",
        "signature": dedent("""\
            def get_password_hash(password: str) -> str:
                \"\"\"Hash a plain-text password using pwdlib PasswordHash.hash. Return the hash string.\"\"\"
        """),
    },
    {
        "id": "verify_password",
        "label": "FN for gpt-4o-mini only — eps=0.000 mini, 0.886 gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "context": "from pwdlib import PasswordHash\nfrom pwdlib.hashers.argon2 import Argon2Hasher\nfrom pwdlib.hashers.bcrypt import BcryptHasher\npassword_hash = PasswordHash((Argon2Hasher(), BcryptHasher()))",
        "signature": dedent("""\
            def verify_password(plain_password: str, hashed_password: str) -> tuple[bool, str | None]:
                \"\"\"Verify password against hash. Uses pwdlib PasswordHash.verify_and_update.
                Returns (verified: bool, updated_hash: str | None).\"\"\"
        """),
    },
    {
        "id": "get_current_active_superuser",
        "label": "consensus FP — eps 0.443-0.561 all 3 models, trigger='The'",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "context": "from fastapi import HTTPException\nfrom app.models import User",
        "signature": dedent("""\
            def get_current_active_superuser(current_user: CurrentUser) -> User:
                \"\"\"FastAPI dependency: raise 403 if current user is not a superuser. Return user if ok.\"\"\"
        """),
    },
    {
        "id": "render_email_template",
        "label": "consensus FP — eps 0.558-0.741 all 3 models, trigger='(\"'",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "context": "from pathlib import Path\nfrom typing import Any\nfrom jinja2 import Template",
        "signature": dedent("""\
            def render_email_template(*, template_name: str, context: dict[str, Any]) -> str:
                \"\"\"Read HTML template from email-templates/build/<template_name>, render with Jinja2
                using the given context dict. Return rendered HTML string.\"\"\"
        """),
    },
    {
        "id": "generate_reset_password_email",
        "label": "consensus FP — eps 0.665-0.985, trigger='subject'",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "context": "from app.core.config import settings",
        "signature": dedent("""\
            def generate_reset_password_email(email_to: str, email: str, token: str) -> EmailData:
                \"\"\"Build password reset email. Subject: '{PROJECT_NAME} - Password recovery for user {email}'.
                Reset link: {FRONTEND_HOST}/reset-password?token={token}.
                Render reset_password.html with project_name, username, email, valid_hours, link. Return EmailData.\"\"\"
        """),
    },
    {
        "id": "health_check",
        "label": "TN for OpenAI, FP for DeepSeek — eps=0.000 vs 0.521",
        "models": ["gpt-4o", "deepseek-ai/DeepSeek-V3"],
        "context": "",
        "signature": dedent("""\
            async def health_check() -> bool:
                \"\"\"Health check endpoint. Return True if service is up.\"\"\"
        """),
    },
]


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


def call_model(client, model, prompt):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a Python expert. Output ONLY the function body (indented, no signature, no markdown fences)."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=512,
        logprobs=True,
        top_logprobs=K,
    )
    lp = resp.choices[0].logprobs
    tokens = []
    if lp.content:
        for tok in lp.content:
            top_lps = [alt.logprob for alt in tok.top_logprobs]
            top_toks = [alt.token for alt in tok.top_logprobs]
            tokens.append({
                "token": tok.token,
                "logprob": tok.logprob,
                "top_logprobs": top_lps,
                "top_tokens": top_toks,
            })
    return tokens, resp.choices[0].message.content


def print_token_sequence(tokens, case_id, model):
    """Print full token sequence with eps values and top alternatives at high-eps positions."""
    eps_vals = [token_epsilon(t["top_logprobs"]) for t in tokens]
    max_eps = max(eps_vals) if eps_vals else 0.0

    print(f"\n  --- Token sequence [{model}] (max eps={max_eps:.3f}) ---")
    print(f"  {'#':>3}  {'TOKEN':<20} {'eps':>6}  {'top alternatives'}")
    print(f"  {'-'*3}  {'-'*20} {'-'*6}  {'-'*50}")

    for i, (t, eps) in enumerate(zip(tokens, eps_vals)):
        tok_display = repr(t["token"])[:18]
        marker = " <-- MAX" if eps == max_eps and eps > DELTA else ""
        flag = "***" if eps > DELTA else "   "

        if eps > DELTA:
            # Show top alternatives at this position
            alts = []
            for tok_alt, lp in zip(t["top_tokens"][:K], t["top_logprobs"][:K]):
                prob = math.exp(lp) * 100
                alts.append(f"{repr(tok_alt)[:12]}({prob:.1f}%)")
            alt_str = "  |  ".join(alts[:3])
            print(f"  {i:>3}  {tok_display:<20} {eps:>6.3f}  {flag} {alt_str}{marker}")
        else:
            # For low-eps tokens, just show the token and eps
            # But only print every token if the sequence is short, else skip near-zero
            if eps > 0.05 or len(tokens) < 20:
                print(f"  {i:>3}  {tok_display:<20} {eps:>6.3f}")
            # else skip near-zero tokens in long sequences

    # Print the actual generated code
    generated = "".join(t["token"] for t in tokens)
    print(f"\n  Generated body:\n")
    for line in generated.splitlines():
        print(f"    {line}")


def main():
    clients = {
        "gpt-4o": OpenAI(),
        "gpt-4o-mini": OpenAI(),
        "deepseek-ai/DeepSeek-V3": OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY", ""),
            base_url="https://api.together.xyz/v1"
        ),
    }

    for case in FAILURE_CASES:
        print(f"\n{'='*70}")
        print(f"CASE: {case['id']}")
        print(f"      {case['label']}")
        print(f"{'='*70}")

        ctx = case["context"].strip()
        prompt = (
            CONTEXT_HEADER
            + ("\nRelevant imports:\n" + ctx + "\n\n" if ctx else "\n")
            + case["signature"].rstrip()
            + "\n"
        )

        for model in case["models"]:
            print(f"\n[{model}]")
            try:
                tokens, generated = call_model(clients[model], model, prompt)
                print_token_sequence(tokens, case["id"], model)
            except Exception as e:
                print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
