#!/usr/bin/env python3
"""
Production FP Rate Benchmark — ε on real production code
=========================================================
Tests ε against 40 real functions from tiangolo/full-stack-fastapi-template
(commit 13652b51ea0acca7dfe243ac25e2bbdc066f3c4f) to validate that the 50%
false-positive rate measured on synthetic LOW prompts drops substantially
in real production API-integration code.

Functions are classified as:
  API   — calls external libraries (SQLModel/ORM, JWT, password hashing, email)
          ε should fire (detection test)
  LOGIC — pure logic, no external API calls
          ε should stay COMPLETE (FP test)

Usage:
    python benchmark_production.py
    python benchmark_production.py --model gpt-4o-mini
    python benchmark_production.py --provider together --model meta-llama/Llama-3.1-70B-Instruct-Turbo

Outputs:
    results/production_<model>.json
"""
import argparse
import ast
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# API client setup
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.  pip install openai")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Prompt context header (shared across all 40 prompts)
# ---------------------------------------------------------------------------
CONTEXT_HEADER = dedent("""\
    You are implementing a Python function for a production FastAPI web application.
    Stack: FastAPI, SQLModel (ORM over PostgreSQL), JWT authentication (PyJWT),
    pwdlib for password hashing, Jinja2 for email templates, emails library for SMTP.

    Key types available:
      Session          — SQLModel database session (from sqlmodel import Session)
      SessionDep       — Annotated[Session, Depends(get_db)]
      CurrentUser      — Annotated[User, Depends(get_current_user)]
      User             — SQLModel ORM model with id (UUID), email, hashed_password,
                         is_active, is_superuser, full_name, created_at, items
      Item             — SQLModel ORM model with id (UUID), title, description,
                         owner_id (UUID FK), created_at, owner
      UserCreate/Update/Register/UpdateMe — Pydantic input schemas
      ItemCreate/Update — Pydantic input schemas
      UserPublic/ItemPublic/UsersPublic/ItemsPublic — Pydantic output schemas
      Token            — {access_token: str, token_type: str}
      Message          — {message: str}
      NewPassword      — {token: str, new_password: str}
      UpdatePassword   — {current_password: str, new_password: str}
      EmailData        — dataclass(html_content: str, subject: str)
      settings         — app config object with SECRET_KEY, ACCESS_TOKEN_EXPIRE_MINUTES,
                         SMTP_HOST/PORT/TLS/SSL/USER/PASSWORD, EMAILS_FROM_NAME,
                         EMAILS_FROM_EMAIL, emails_enabled, FRONTEND_HOST,
                         EMAIL_RESET_TOKEN_EXPIRE_HOURS, PROJECT_NAME
      security.ALGORITHM — "HS256"

    Implement ONLY the function body. Do not repeat the signature or imports.
    Write idiomatic, production-quality Python.
""")

# ---------------------------------------------------------------------------
# 40 production prompts
# Each entry: id, file, name, type ("API" or "LOGIC"), prompt_suffix
# prompt_suffix is the function signature + docstring block
# ---------------------------------------------------------------------------
PROMPTS = [
    # ── items.py ────────────────────────────────────────────────────────────
    {
        "id": "items_read_items",
        "file": "backend/app/api/routes/items.py",
        "name": "read_items",
        "type": "API",
        "context": "from sqlmodel import col, func, select\nfrom app.models import Item, ItemsPublic, ItemPublic",
        "signature": dedent("""\
            def read_items(
                session: SessionDep, current_user: CurrentUser, skip: int = 0, limit: int = 100
            ) -> ItemsPublic:
                \"\"\"Retrieve items. Superusers see all items; regular users see only their own.
                Returns ItemsPublic with data list and total count. Ordered by created_at desc.\"\"\"
        """),
    },
    {
        "id": "items_read_item",
        "file": "backend/app/api/routes/items.py",
        "name": "read_item",
        "type": "API",
        "context": "import uuid\nfrom app.models import Item, ItemPublic",
        "signature": dedent("""\
            def read_item(session: SessionDep, current_user: CurrentUser, id: uuid.UUID) -> ItemPublic:
                \"\"\"Get item by ID. Raise 404 if not found. Raise 403 if not owner and not superuser.\"\"\"
        """),
    },
    {
        "id": "items_create_item",
        "file": "backend/app/api/routes/items.py",
        "name": "create_item",
        "type": "API",
        "context": "from app.models import Item, ItemCreate, ItemPublic",
        "signature": dedent("""\
            def create_item(*, session: SessionDep, current_user: CurrentUser, item_in: ItemCreate) -> ItemPublic:
                \"\"\"Create new item owned by the current user. Persist and refresh.\"\"\"
        """),
    },
    {
        "id": "items_update_item",
        "file": "backend/app/api/routes/items.py",
        "name": "update_item",
        "type": "API",
        "context": "import uuid\nfrom app.models import Item, ItemUpdate, ItemPublic",
        "signature": dedent("""\
            def update_item(
                *, session: SessionDep, current_user: CurrentUser, id: uuid.UUID, item_in: ItemUpdate
            ) -> ItemPublic:
                \"\"\"Update item. Raise 404 if not found. Raise 403 if not owner and not superuser.
                Apply partial update via model_dump(exclude_unset=True) + sqlmodel_update.\"\"\"
        """),
    },
    {
        "id": "items_delete_item",
        "file": "backend/app/api/routes/items.py",
        "name": "delete_item",
        "type": "API",
        "context": "import uuid\nfrom app.models import Item, Message",
        "signature": dedent("""\
            def delete_item(session: SessionDep, current_user: CurrentUser, id: uuid.UUID) -> Message:
                \"\"\"Delete item by ID. Raise 404 if not found. Raise 403 if not owner and not superuser.\"\"\"
        """),
    },

    # ── login.py ─────────────────────────────────────────────────────────────
    {
        "id": "login_login_access_token",
        "file": "backend/app/api/routes/login.py",
        "name": "login_access_token",
        "type": "API",
        "context": "from datetime import timedelta\nfrom fastapi import Depends\nfrom fastapi.security import OAuth2PasswordRequestForm\nfrom typing import Annotated\nfrom app import crud\nfrom app.core import security\nfrom app.models import Token",
        "signature": dedent("""\
            def login_access_token(
                session: SessionDep,
                form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
            ) -> Token:
                \"\"\"OAuth2 token login. Authenticate user, raise 400 on bad credentials or inactive user.
                Create access token with expiry from settings.ACCESS_TOKEN_EXPIRE_MINUTES.\"\"\"
        """),
    },
    {
        "id": "login_test_token",
        "file": "backend/app/api/routes/login.py",
        "name": "test_token",
        "type": "LOGIC",
        "context": "from typing import Any\nfrom app.models import UserPublic",
        "signature": dedent("""\
            def test_token(current_user: CurrentUser) -> Any:
                \"\"\"Test access token. Return the current authenticated user.\"\"\"
        """),
    },
    {
        "id": "login_recover_password",
        "file": "backend/app/api/routes/login.py",
        "name": "recover_password",
        "type": "API",
        "context": "from app import crud\nfrom app.models import Message\nfrom app.utils import generate_password_reset_token, generate_reset_password_email, send_email",
        "signature": dedent("""\
            def recover_password(email: str, session: SessionDep) -> Message:
                \"\"\"Send password recovery email if user exists (silent no-op if not, to prevent enumeration).
                Generate token, build email content, send. Always return same success message.\"\"\"
        """),
    },
    {
        "id": "login_reset_password",
        "file": "backend/app/api/routes/login.py",
        "name": "reset_password",
        "type": "API",
        "context": "from app import crud\nfrom app.models import Message, NewPassword, UserUpdate\nfrom app.utils import verify_password_reset_token",
        "signature": dedent("""\
            def reset_password(session: SessionDep, body: NewPassword) -> Message:
                \"\"\"Reset password using token. Raise 400 for invalid token, missing user, or inactive user.
                Validate token, look up user, update password via crud.update_user.\"\"\"
        """),
    },
    {
        "id": "login_recover_password_html_content",
        "file": "backend/app/api/routes/login.py",
        "name": "recover_password_html_content",
        "type": "API",
        "context": "from fastapi.responses import HTMLResponse\nfrom app import crud\nfrom app.utils import generate_password_reset_token, generate_reset_password_email",
        "signature": dedent("""\
            def recover_password_html_content(email: str, session: SessionDep) -> Any:
                \"\"\"Return HTML content for password recovery email (for superuser preview).
                Raise 404 if user not found. Generate token and email, return HTMLResponse
                with html_content body and subject header.\"\"\"
        """),
    },

    # ── users.py ─────────────────────────────────────────────────────────────
    {
        "id": "users_read_users",
        "file": "backend/app/api/routes/users.py",
        "name": "read_users",
        "type": "API",
        "context": "from sqlmodel import col, func, select\nfrom app.models import User, UserPublic, UsersPublic",
        "signature": dedent("""\
            def read_users(session: SessionDep, skip: int = 0, limit: int = 100) -> Any:
                \"\"\"Retrieve all users (superuser only). Return UsersPublic with data list and count.
                Ordered by created_at desc.\"\"\"
        """),
    },
    {
        "id": "users_create_user",
        "file": "backend/app/api/routes/users.py",
        "name": "create_user",
        "type": "API",
        "context": "from app import crud\nfrom app.models import UserCreate, UserPublic\nfrom app.utils import generate_new_account_email, send_email",
        "signature": dedent("""\
            def create_user(*, session: SessionDep, user_in: UserCreate) -> Any:
                \"\"\"Create new user (superuser only). Raise 400 if email already exists.
                Send welcome email if emails enabled. Return created user.\"\"\"
        """),
    },
    {
        "id": "users_update_user_me",
        "file": "backend/app/api/routes/users.py",
        "name": "update_user_me",
        "type": "API",
        "context": "from app import crud\nfrom app.models import UserPublic, UserUpdateMe",
        "signature": dedent("""\
            def update_user_me(
                *, session: SessionDep, user_in: UserUpdateMe, current_user: CurrentUser
            ) -> Any:
                \"\"\"Update own user profile. Raise 409 if new email is already taken by another user.
                Apply partial update via sqlmodel_update, persist, refresh.\"\"\"
        """),
    },
    {
        "id": "users_update_password_me",
        "file": "backend/app/api/routes/users.py",
        "name": "update_password_me",
        "type": "API",
        "context": "from app.core.security import get_password_hash, verify_password\nfrom app.models import Message, UpdatePassword",
        "signature": dedent("""\
            def update_password_me(
                *, session: SessionDep, body: UpdatePassword, current_user: CurrentUser
            ) -> Any:
                \"\"\"Update own password. Raise 400 if current password wrong or new == current.
                Hash new password, update hashed_password, persist.\"\"\"
        """),
    },
    {
        "id": "users_read_user_me",
        "file": "backend/app/api/routes/users.py",
        "name": "read_user_me",
        "type": "LOGIC",
        "context": "from typing import Any\nfrom app.models import UserPublic",
        "signature": dedent("""\
            def read_user_me(current_user: CurrentUser) -> Any:
                \"\"\"Get current authenticated user.\"\"\"
        """),
    },
    {
        "id": "users_delete_user_me",
        "file": "backend/app/api/routes/users.py",
        "name": "delete_user_me",
        "type": "API",
        "context": "from typing import Any\nfrom app.models import Message",
        "signature": dedent("""\
            def delete_user_me(session: SessionDep, current_user: CurrentUser) -> Any:
                \"\"\"Delete own user account. Raise 403 if superuser (not allowed to self-delete).
                Delete from session and commit.\"\"\"
        """),
    },
    {
        "id": "users_register_user",
        "file": "backend/app/api/routes/users.py",
        "name": "register_user",
        "type": "API",
        "context": "from app import crud\nfrom app.models import UserCreate, UserPublic, UserRegister",
        "signature": dedent("""\
            def register_user(session: SessionDep, user_in: UserRegister) -> Any:
                \"\"\"Public signup (no auth required). Raise 400 if email already registered.
                Validate to UserCreate and create via crud.\"\"\"
        """),
    },
    {
        "id": "users_read_user_by_id",
        "file": "backend/app/api/routes/users.py",
        "name": "read_user_by_id",
        "type": "API",
        "context": "import uuid\nfrom app.models import User, UserPublic",
        "signature": dedent("""\
            def read_user_by_id(
                user_id: uuid.UUID, session: SessionDep, current_user: CurrentUser
            ) -> Any:
                \"\"\"Get user by ID. Non-superusers can only access their own record (return directly).
                Raise 403 if non-superuser requesting another user. Raise 404 if not found.\"\"\"
        """),
    },
    {
        "id": "users_update_user",
        "file": "backend/app/api/routes/users.py",
        "name": "update_user",
        "type": "API",
        "context": "import uuid\nfrom app import crud\nfrom app.models import UserPublic, UserUpdate",
        "signature": dedent("""\
            def update_user(
                *, session: SessionDep, user_id: uuid.UUID, user_in: UserUpdate
            ) -> Any:
                \"\"\"Update user by ID (superuser only). Raise 404 if user not found.
                Raise 409 if new email already exists for a different user. Update via crud.\"\"\"
        """),
    },
    {
        "id": "users_delete_user",
        "file": "backend/app/api/routes/users.py",
        "name": "delete_user",
        "type": "API",
        "context": "import uuid\nfrom sqlmodel import col, delete\nfrom app.models import Item, Message, User",
        "signature": dedent("""\
            def delete_user(
                session: SessionDep, current_user: CurrentUser, user_id: uuid.UUID
            ) -> Message:
                \"\"\"Delete user by ID (superuser only). Raise 404 if not found.
                Raise 403 if trying to delete self. Delete owned items first, then user.\"\"\"
        """),
    },

    # ── routes/utils.py ──────────────────────────────────────────────────────
    {
        "id": "rutils_test_email",
        "file": "backend/app/api/routes/utils.py",
        "name": "test_email",
        "type": "API",
        "context": "from pydantic.networks import EmailStr\nfrom app.models import Message\nfrom app.utils import generate_test_email, send_email",
        "signature": dedent("""\
            def test_email(email_to: EmailStr) -> Message:
                \"\"\"Send a test email to the given address (superuser only). Return success message.\"\"\"
        """),
    },
    {
        "id": "rutils_health_check",
        "file": "backend/app/api/routes/utils.py",
        "name": "health_check",
        "type": "LOGIC",
        "context": "",
        "signature": dedent("""\
            async def health_check() -> bool:
                \"\"\"Health check endpoint. Return True if service is up.\"\"\"
        """),
    },

    # ── deps.py ──────────────────────────────────────────────────────────────
    {
        "id": "deps_get_db",
        "file": "backend/app/api/deps.py",
        "name": "get_db",
        "type": "API",
        "context": "from collections.abc import Generator\nfrom sqlmodel import Session\nfrom app.core.db import engine",
        "signature": dedent("""\
            def get_db() -> Generator[Session, None, None]:
                \"\"\"FastAPI dependency: yield a SQLModel Session, close on exit.\"\"\"
        """),
    },
    {
        "id": "deps_get_current_user",
        "file": "backend/app/api/deps.py",
        "name": "get_current_user",
        "type": "API",
        "context": "import jwt\nfrom fastapi import Depends, HTTPException, status\nfrom fastapi.security import OAuth2PasswordBearer\nfrom jwt.exceptions import InvalidTokenError\nfrom pydantic import ValidationError\nfrom app.core import security\nfrom app.models import TokenPayload, User",
        "signature": dedent("""\
            def get_current_user(session: SessionDep, token: TokenDep) -> User:
                \"\"\"FastAPI dependency: decode JWT, look up user, raise 403/404/400 as needed.
                Raise 403 on invalid token or validation error. Raise 404 if user not found.
                Raise 400 if user is inactive.\"\"\"
        """),
    },
    {
        "id": "deps_get_current_active_superuser",
        "file": "backend/app/api/deps.py",
        "name": "get_current_active_superuser",
        "type": "LOGIC",
        "context": "from fastapi import HTTPException\nfrom app.models import User",
        "signature": dedent("""\
            def get_current_active_superuser(current_user: CurrentUser) -> User:
                \"\"\"FastAPI dependency: raise 403 if current user is not a superuser. Return user if ok.\"\"\"
        """),
    },

    # ── core/security.py ─────────────────────────────────────────────────────
    {
        "id": "security_create_access_token",
        "file": "backend/app/core/security.py",
        "name": "create_access_token",
        "type": "API",
        "context": "from datetime import datetime, timedelta, timezone\nfrom typing import Any\nimport jwt\nALGORITHM = 'HS256'",
        "signature": dedent("""\
            def create_access_token(subject: str | Any, expires_delta: timedelta) -> str:
                \"\"\"Create a JWT access token. Payload: {exp, sub}. Sign with settings.SECRET_KEY
                using ALGORITHM. Return encoded string.\"\"\"
        """),
    },
    {
        "id": "security_verify_password",
        "file": "backend/app/core/security.py",
        "name": "verify_password",
        "type": "API",
        "context": "from pwdlib import PasswordHash\nfrom pwdlib.hashers.argon2 import Argon2Hasher\nfrom pwdlib.hashers.bcrypt import BcryptHasher\npassword_hash = PasswordHash((Argon2Hasher(), BcryptHasher()))",
        "signature": dedent("""\
            def verify_password(plain_password: str, hashed_password: str) -> tuple[bool, str | None]:
                \"\"\"Verify password against hash. Uses pwdlib PasswordHash.verify_and_update.
                Returns (verified: bool, updated_hash: str | None).\"\"\"
        """),
    },
    {
        "id": "security_get_password_hash",
        "file": "backend/app/core/security.py",
        "name": "get_password_hash",
        "type": "API",
        "context": "from pwdlib import PasswordHash\nfrom pwdlib.hashers.argon2 import Argon2Hasher\nfrom pwdlib.hashers.bcrypt import BcryptHasher\npassword_hash = PasswordHash((Argon2Hasher(), BcryptHasher()))",
        "signature": dedent("""\
            def get_password_hash(password: str) -> str:
                \"\"\"Hash a plain-text password using pwdlib PasswordHash.hash. Return the hash string.\"\"\"
        """),
    },

    # ── crud.py ──────────────────────────────────────────────────────────────
    {
        "id": "crud_create_user",
        "file": "backend/app/crud.py",
        "name": "create_user",
        "type": "API",
        "context": "from sqlmodel import Session\nfrom app.core.security import get_password_hash\nfrom app.models import User, UserCreate",
        "signature": dedent("""\
            def create_user(*, session: Session, user_create: UserCreate) -> User:
                \"\"\"Create a User ORM object: validate UserCreate, hash password, add/commit/refresh.\"\"\"
        """),
    },
    {
        "id": "crud_update_user",
        "file": "backend/app/crud.py",
        "name": "update_user",
        "type": "API",
        "context": "from typing import Any\nfrom sqlmodel import Session\nfrom app.core.security import get_password_hash\nfrom app.models import User, UserUpdate",
        "signature": dedent("""\
            def update_user(*, session: Session, db_user: User, user_in: UserUpdate) -> Any:
                \"\"\"Update a User ORM object. If password in update data, hash it.
                Apply via sqlmodel_update, add/commit/refresh.\"\"\"
        """),
    },
    {
        "id": "crud_get_user_by_email",
        "file": "backend/app/crud.py",
        "name": "get_user_by_email",
        "type": "API",
        "context": "from sqlmodel import Session, select\nfrom app.models import User",
        "signature": dedent("""\
            def get_user_by_email(*, session: Session, email: str) -> User | None:
                \"\"\"Query user by email using sqlmodel select. Return first result or None.\"\"\"
        """),
    },
    {
        "id": "crud_authenticate",
        "file": "backend/app/crud.py",
        "name": "authenticate",
        "type": "API",
        "context": "from sqlmodel import Session\nfrom app.core.security import verify_password\nfrom app.models import User\nDUMMY_HASH = '$argon2id$v=19$m=65536,t=3,p=4$MjQyZWE1MzBjYjJlZTI0Yw$YTU4NGM5ZTZmYjE2NzZlZjY0ZWY3ZGRkY2U2OWFjNjk'",
        "signature": dedent("""\
            def authenticate(*, session: Session, email: str, password: str) -> User | None:
                \"\"\"Authenticate user by email/password. Use DUMMY_HASH to prevent timing attacks
                when user not found. Return User if verified, None otherwise.
                If verify_password returns an updated hash, persist it.\"\"\"
        """),
    },
    {
        "id": "crud_create_item",
        "file": "backend/app/crud.py",
        "name": "create_item",
        "type": "API",
        "context": "import uuid\nfrom sqlmodel import Session\nfrom app.models import Item, ItemCreate",
        "signature": dedent("""\
            def create_item(*, session: Session, item_in: ItemCreate, owner_id: uuid.UUID) -> Item:
                \"\"\"Create an Item ORM object: validate ItemCreate with owner_id, add/commit/refresh.\"\"\"
        """),
    },

    # ── utils.py ─────────────────────────────────────────────────────────────
    {
        "id": "utils_render_email_template",
        "file": "backend/app/utils.py",
        "name": "render_email_template",
        "type": "LOGIC",
        "context": "from pathlib import Path\nfrom typing import Any\nfrom jinja2 import Template",
        "signature": dedent("""\
            def render_email_template(*, template_name: str, context: dict[str, Any]) -> str:
                \"\"\"Read HTML template from email-templates/build/<template_name>, render with Jinja2
                using the given context dict. Return rendered HTML string.\"\"\"
        """),
    },
    {
        "id": "utils_send_email",
        "file": "backend/app/utils.py",
        "name": "send_email",
        "type": "API",
        "context": "import emails\nfrom app.core.config import settings",
        "signature": dedent("""\
            def send_email(*, email_to: str, subject: str = '', html_content: str = '') -> None:
                \"\"\"Send an email via SMTP using the emails library. Assert emails_enabled.
                Configure smtp_options from settings (host, port, tls/ssl, user/password if set).
                Call message.send(to=email_to, smtp=smtp_options).\"\"\"
        """),
    },
    {
        "id": "utils_generate_test_email",
        "file": "backend/app/utils.py",
        "name": "generate_test_email",
        "type": "LOGIC",
        "context": "from app.core.config import settings",
        "signature": dedent("""\
            def generate_test_email(email_to: str) -> EmailData:
                \"\"\"Build test email content. Subject: '{PROJECT_NAME} - Test email'.
                Render test_email.html template with project_name and email context. Return EmailData.\"\"\"
        """),
    },
    {
        "id": "utils_generate_reset_password_email",
        "file": "backend/app/utils.py",
        "name": "generate_reset_password_email",
        "type": "LOGIC",
        "context": "from app.core.config import settings",
        "signature": dedent("""\
            def generate_reset_password_email(email_to: str, email: str, token: str) -> EmailData:
                \"\"\"Build password reset email. Subject: '{PROJECT_NAME} - Password recovery for user {email}'.
                Reset link: {FRONTEND_HOST}/reset-password?token={token}.
                Render reset_password.html with project_name, username, email, valid_hours, link. Return EmailData.\"\"\"
        """),
    },
    {
        "id": "utils_generate_new_account_email",
        "file": "backend/app/utils.py",
        "name": "generate_new_account_email",
        "type": "LOGIC",
        "context": "from app.core.config import settings",
        "signature": dedent("""\
            def generate_new_account_email(email_to: str, username: str, password: str) -> EmailData:
                \"\"\"Build new account email. Subject: '{PROJECT_NAME} - New account for user {username}'.
                Render new_account.html with project_name, username, password, email, link. Return EmailData.\"\"\"
        """),
    },
    {
        "id": "utils_generate_password_reset_token",
        "file": "backend/app/utils.py",
        "name": "generate_password_reset_token",
        "type": "API",
        "context": "from datetime import datetime, timedelta, timezone\nimport jwt\nfrom app.core import security\nfrom app.core.config import settings",
        "signature": dedent("""\
            def generate_password_reset_token(email: str) -> str:
                \"\"\"Generate a JWT token for password reset. Payload: {exp, nbf, sub: email}.
                Sign with settings.SECRET_KEY using security.ALGORITHM. Return encoded string.\"\"\"
        """),
    },
    {
        "id": "utils_verify_password_reset_token",
        "file": "backend/app/utils.py",
        "name": "verify_password_reset_token",
        "type": "API",
        "context": "import jwt\nfrom jwt.exceptions import InvalidTokenError\nfrom app.core import security\nfrom app.core.config import settings",
        "signature": dedent("""\
            def verify_password_reset_token(token: str) -> str | None:
                \"\"\"Decode a password reset JWT. Return the subject (email) string, or None on any error.\"\"\"
        """),
    },
]

# ---------------------------------------------------------------------------
# ε math (self-contained, no dependency on epsilon/core.py)
# ---------------------------------------------------------------------------
DELTA = 0.30   # noise floor — tokens below this are excluded from aggregation
K = 5          # top-k alternatives

# AST declaration node types to filter (cosmetic tokens)
_COSMETIC_TYPES = {
    ast.FunctionDef, ast.AsyncFunctionDef,  # function name
    ast.arg,                                 # parameter names
}


def _cosmetic_ranges(source: str) -> list[tuple[int, int]]:
    """Return (col_start, col_end) ranges of cosmetic identifier tokens."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # function name token is at node.col_offset + len("def ") or "async def "
            ranges.append((node.col_offset, node.col_offset + len(node.name) + 10))
        elif isinstance(node, ast.arg):
            ranges.append((node.col_offset, node.col_offset + len(node.arg) + 2))
        elif isinstance(node, ast.Name) and isinstance(getattr(node, 'ctx', None), ast.Store):
            ranges.append((node.col_offset, node.col_offset + len(node.id) + 2))
    return ranges


def token_epsilon(top_logprobs: list) -> float:
    """Normalized partial Shannon entropy over top-k alternatives."""
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


def file_epsilon(token_data: list[dict]) -> tuple[float, str | None]:
    """
    Max epsilon over consequential tokens (above DELTA noise floor).
    token_data: list of {token: str, logprob: float, top_logprobs: [float, ...]}
    Returns (eps_file, trigger_token | None).
    """
    max_eps = 0.0
    trigger = None
    for td in token_data:
        raw = td["token"].strip()
        # Skip whitespace-only tokens
        if not raw:
            continue
        # Skip obvious keywords that are structural rather than semantic
        eps = token_epsilon(td["top_logprobs"])
        if eps > DELTA and eps > max_eps:
            max_eps = eps
            trigger = td["token"]
    return max_eps, trigger


def eps_to_status(eps: float) -> str:
    if eps < 0.30:
        return "COMPLETE"
    elif eps < 0.65:
        return "FLAGGED"
    elif eps < 0.95:
        return "PAUSED"
    else:
        return "ABORTED"


def cascaded_epsilon(token_data: list[dict]) -> dict:
    """
    Multi-check cascaded uncertainty classifier.

    Distinguishes Type 2 (API/library decision clusters) from Type 3 (naming
    spikes) using five checks beyond the raw peak value:

      A  cluster_count    -- tokens with eps > 0.20 (density indicator)
      B  max_run          -- longest consecutive run with eps > 0.15
                            (API call chains sustain elevation; naming spikes don't)
      C  drops_fast       -- next 2 tokens after peak both drop below 0.15
                            (penalises isolated spikes)
      D  re_escalation    -- times eps rises > 0.20 after 3-token gap below 0.10
                            (compound API: multiple independent decision points)
      E  elev_fraction    -- cluster_count / total_tokens

    Window context saved for inspection: 5 tokens before + trigger + 12 after.

    Returns dict with cascaded_score, cascaded_status, and evidence sub-dict.
    """
    eps_seq = [token_epsilon(t["top_logprobs"]) for t in token_data]
    n = len(eps_seq)

    # Gate 1 — sentinel
    peak_eps = max(eps_seq) if eps_seq else 0.0
    if peak_eps < 0.10:
        return {
            "cascaded_score": 0.0,
            "cascaded_status": "COMPLETE",
            "evidence": {"peak_eps": round(peak_eps, 4)},
        }

    # Gate 2 — peak floor
    if peak_eps < 0.30:
        return {
            "cascaded_score": round(peak_eps * 0.5, 4),
            "cascaded_status": "COMPLETE",
            "evidence": {"peak_eps": round(peak_eps, 4)},
        }

    peak_idx = eps_seq.index(peak_eps)

    # Check A — cluster count (tokens > 0.20)
    cluster_count = sum(1 for e in eps_seq if e > 0.20)

    # Check B — longest consecutive run above 0.15
    max_run = 0
    cur_run = 0
    for e in eps_seq:
        if e > 0.15:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    # Check C — post-peak resolution speed
    post_peak = eps_seq[peak_idx + 1: peak_idx + 3]
    drops_fast = (len(post_peak) >= 2 and all(e < 0.15 for e in post_peak))

    # Check D — re-escalation after sustained drop (3+ tokens below 0.10)
    re_escalation = 0
    in_cluster = eps_seq[peak_idx] > 0.20
    gap_len = 0
    for i in range(peak_idx + 1, n):
        if in_cluster:
            if eps_seq[i] < 0.10:
                gap_len += 1
                if gap_len >= 3:
                    in_cluster = False
                    gap_len = 0
        else:
            if eps_seq[i] > 0.20:
                re_escalation += 1
                in_cluster = True
                gap_len = 0

    # Check E — elevated fraction
    elev_fraction = cluster_count / n if n > 0 else 0.0

    # Check F — short-body lone spike (wording/model-verbosity uncertainty, not API chains)
    #
    # Two tiers:
    #   Tier 1 (n ≤ 6): ANY function generating ≤6 tokens cannot be a real API call.
    #     TP min observed = 7 tokens → this tier is safe to penalise aggressively.
    #     Fixes DeepSeek trivial-function FPs (test_token, read_user_me, health_check).
    #
    #   Tier 2 (7 ≤ n ≤ 40, re=0, cluster ≤ 5, run ≤ 1): short generation,
    #     isolated uncertainty spike, zero re-escalation (no API chaining).
    #     The run ≤ 1 condition prevents penalising short functions that DO have
    #     a sustained API-call run (which implies a real library call).
    #     Fixes get_current_active_superuser (gpt-4o, deepseek).
    if n <= 6 and re_escalation == 0:
        # Tier 1: ≤6 tokens — no real API call possible in this space.
        # TP min observed = 7 tokens → completely safe threshold.
        lone_spike_tier = 1
        lone_spike_penalty = 0.70
    elif (n <= 40 and re_escalation == 0 and 3 <= cluster_count <= 5 and max_run <= 1):
        # Tier 2: short function, zero chaining, isolated multi-point uncertainty.
        # cluster ≥ 3 excludes single-spike short TPs (get_db, get_password_hash).
        # cluster ≤ 5 excludes schema-ambiguous FPs (generate_* cluster 7–16).
        # run ≤ 1 excludes sustained API-call runs.
        lone_spike_tier = 2
        lone_spike_penalty = 0.40
    else:
        lone_spike_tier = 0
        lone_spike_penalty = 0.0

    # Weighted score
    score = peak_eps
    score += 0.10 * min(cluster_count / 12.0, 1.0)
    score += 0.10 * min(max_run / 4.0, 1.0)
    if drops_fast:
        score -= 0.15
    score += 0.05 * min(re_escalation, 3)
    score += 0.10 * min(elev_fraction / 0.15, 1.0)
    score -= lone_spike_penalty
    score = max(0.0, min(1.0, score))

    # Window context (5 before + trigger + 12 after)
    w_start = max(0, peak_idx - 5)
    w_end   = min(n, peak_idx + 13)
    window_eps = eps_seq[w_start:w_end]

    evidence = {
        "peak_eps":         round(peak_eps, 4),
        "peak_idx":         peak_idx,
        "cluster_count":    cluster_count,
        "max_run":          max_run,
        "drops_fast":       drops_fast,
        "re_escalation":    re_escalation,
        "elev_fraction":    round(elev_fraction, 4),
        "lone_spike":       lone_spike_tier > 0,
        "lone_spike_pen":   round(lone_spike_penalty, 4),
        "window_eps":       [round(e, 4) for e in window_eps],
        "total_tokens":     n,
    }

    return {
        "cascaded_score":  round(score, 4),
        "cascaded_status": eps_to_status(score),
        "evidence":        evidence,
    }


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

def build_prompt(p: dict) -> str:
    ctx = p["context"].strip()
    sig = p["signature"].rstrip()
    return (
        CONTEXT_HEADER
        + ("\nRelevant imports available:\n" + ctx + "\n\n" if ctx else "\n")
        + sig
        + "\n"
    )


def call_openai(client: OpenAI, prompt: str, model: str) -> tuple[list[dict], str]:
    """
    Call chat completions with logprobs=True.
    Handles two response formats:
      - Standard (OpenAI/DeepSeek): logprobs.content is a list of token objects
      - Legacy dict (Together/Llama): logprobs.content is None; data in
        logprobs.token_logprobs + logprobs.top_logprobs (list of {tok: lp} dicts)
    Returns list of {token, logprob, top_logprobs: [float]} for each token.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Python expert. Output ONLY the function body "
                    "(indented, no signature line, no markdown fences)."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=512,
        logprobs=True,
        top_logprobs=K,
    )
    lp = response.choices[0].logprobs
    tokens = []

    if lp.content:
        # Standard format (OpenAI, DeepSeek-V3)
        for tok in lp.content:
            top_lps  = [alt.logprob for alt in tok.top_logprobs]
            top_toks = [alt.token   for alt in tok.top_logprobs]
            tokens.append({
                "token":       tok.token,
                "logprob":     tok.logprob,
                "top_logprobs": top_lps,
                "top_tokens":  top_toks,
            })
    elif lp.tokens and lp.token_logprobs:
        # Legacy dict format (Together.ai Llama variants)
        for tok, lp_val, top_dict in zip(lp.tokens, lp.token_logprobs, lp.top_logprobs or []):
            if top_dict:
                sorted_items = sorted(top_dict.items(), key=lambda x: x[1], reverse=True)
                top_toks = [k for k, _ in sorted_items]
                top_lps  = [v for _, v in sorted_items]
            else:
                top_toks = [tok]
                top_lps  = [lp_val]
            tokens.append({
                "token":       tok,
                "logprob":     lp_val,
                "top_logprobs": top_lps,
                "top_tokens":  top_toks,
            })

    return tokens, (response.choices[0].message.content or "")


# ---------------------------------------------------------------------------
# Together.ai / Llama (OpenAI-compatible, different base_url)
# ---------------------------------------------------------------------------

def get_together_client() -> OpenAI:
    key = os.environ.get("TOGETHER_API_KEY", "")
    if not key:
        raise RuntimeError("TOGETHER_API_KEY not set in environment")
    return OpenAI(api_key=key, base_url="https://api.together.xyz/v1")

# Default cross-vendor model (serverless, logprobs confirmed working)
TOGETHER_DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3"


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(client: OpenAI, model: str, provider: str = "openai") -> list[dict]:
    results = []
    api_count = sum(1 for p in PROMPTS if p["type"] == "API")
    logic_count = sum(1 for p in PROMPTS if p["type"] == "LOGIC")
    print(f"\nProduction benchmark: {len(PROMPTS)} functions "
          f"({api_count} API, {logic_count} LOGIC) | model={model} | provider={provider}")
    print("-" * 70)

    for i, p in enumerate(PROMPTS, 1):
        prompt = build_prompt(p)
        try:
            token_data, generated = call_openai(client, prompt, model)
        except Exception as e:
            print(f"  [{i:02d}/{len(PROMPTS)}] ERROR {p['name']}: {e}")
            results.append({**p, "error": str(e), "epsilon": None, "status": "ERROR"})
            continue

        eps, trigger = file_epsilon(token_data)
        status = eps_to_status(eps)
        fired = eps >= 0.30
        casc = cascaded_epsilon(token_data)
        casc_fired = casc["cascaded_score"] >= 0.30

        # Determine correctness: API should fire, LOGIC should not
        expected_fire = p["type"] == "API"
        correct       = fired      == expected_fire
        casc_correct  = casc_fired == expected_fire
        label = ("TP" if fired and expected_fire else
                 "TN" if not fired and not expected_fire else
                 "FP" if fired and not expected_fire else
                 "FN")
        casc_label = ("TP" if casc_fired and expected_fire else
                      "TN" if not casc_fired and not expected_fire else
                      "FP" if casc_fired and not expected_fire else
                      "FN")

        # Build sparse token record: only tokens with eps > 0.10 (with window context)
        eps_seq = [token_epsilon(t["top_logprobs"]) for t in token_data]
        sparse_tokens = []
        for j, (td, e) in enumerate(zip(token_data, eps_seq)):
            if e > 0.10:
                alts = list(zip(
                    td.get("top_tokens", [])[:5],
                    [round(math.exp(lp) * 100, 1) for lp in td["top_logprobs"][:5]],
                ))
                sparse_tokens.append({
                    "idx": j, "token": td["token"], "eps": round(e, 4),
                    "alts": alts,
                })

        casc_marker = "+" if casc_correct else "!"
        marker = "+" if correct else "!"
        trig_safe = trigger.encode("ascii", "replace").decode("ascii") if trigger else None
        print(f"  [{i:02d}/{len(PROMPTS)}] {marker} [{label}] [{p['type']}] "
              f"{p['name']:<35} eps={eps:.3f} {status:<10}"
              + f"  casc={casc['cascaded_score']:.3f} {casc_label} {casc_marker}"
              + (f" << '{trig_safe}'" if trig_safe else ""))

        results.append({
            **{k: v for k, v in p.items()},
            "generated":        generated,
            "epsilon":          round(eps, 4),
            "status":           status,
            "fired":            fired,
            "trigger_token":    trigger,
            "correct":          correct,
            "label":            label,
            "token_count":      len(token_data),
            "eps_seq":          [round(e, 4) for e in eps_seq],
            "cascaded_score":   casc["cascaded_score"],
            "cascaded_status":  casc["cascaded_status"],
            "casc_fired":       casc_fired,
            "casc_correct":     casc_correct,
            "casc_label":       casc_label,
            "evidence":         casc["evidence"],
            "sparse_tokens":    sparse_tokens,
        })

    return results


def print_summary(results: list[dict]) -> None:
    api_results   = [r for r in results if r["type"] == "API"   and r.get("epsilon") is not None]
    logic_results = [r for r in results if r["type"] == "LOGIC" and r.get("epsilon") is not None]

    # Original algorithm
    tp = sum(1 for r in api_results   if r["label"] == "TP")
    fn = sum(1 for r in api_results   if r["label"] == "FN")
    fp = sum(1 for r in logic_results if r["label"] == "FP")
    tn = sum(1 for r in logic_results if r["label"] == "TN")
    det_rate = tp / len(api_results)   * 100 if api_results   else 0
    fp_rate  = fp / len(logic_results) * 100 if logic_results else 0

    # Cascaded algorithm
    c_tp = sum(1 for r in api_results   if r.get("casc_label") == "TP")
    c_fn = sum(1 for r in api_results   if r.get("casc_label") == "FN")
    c_fp = sum(1 for r in logic_results if r.get("casc_label") == "FP")
    c_tn = sum(1 for r in logic_results if r.get("casc_label") == "TN")
    c_det  = c_tp / len(api_results)   * 100 if api_results   else 0
    c_fp_r = c_fp / len(logic_results) * 100 if logic_results else 0

    print("\n" + "=" * 70)
    print("PRODUCTION BENCHMARK SUMMARY")
    print("-" * 70)
    print(f"  Source:  tiangolo/full-stack-fastapi-template @ 13652b51")
    print(f"  Total functions: {len(results)}")
    print()
    print(f"  {'Algorithm':<20} {'Detection':<12} {'FP rate':<10} TP  FN  FP  TN")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*3} {'-'*3} {'-'*3} {'-'*3}")
    print(f"  {'Original (peak eps)':<20} {det_rate:>6.0f}%       {fp_rate:>5.0f}%      {tp:>3} {fn:>3} {fp:>3} {tn:>3}")
    print(f"  {'Cascaded (v2)':<20} {c_det:>6.0f}%       {c_fp_r:>5.0f}%      {c_tp:>3} {c_fn:>3} {c_fp:>3} {c_tn:>3}")
    print()
    print(f"  Reference (synthetic LOW benchmark FP rate): 50%")
    delta_orig = 50 - fp_rate
    delta_casc = 50 - c_fp_r
    print(f"  FP improvement vs reference: "
          f"original {delta_orig:+.0f} pp  |  cascaded {delta_casc:+.0f} pp")
    print("=" * 70)

    # Per-function detail on LOGIC set (both algorithms)
    print("\n  LOGIC function detail:")
    print(f"  {'name':<42} {'eps':>6}  {'orig':>4}  {'casc_score':>10}  {'casc':>4}")
    print(f"  {'-'*42} {'-'*6}  {'-'*4}  {'-'*10}  {'-'*4}")
    for r in logic_results:
        cs = r.get("cascaded_score", 0.0)
        print(f"  {r['name']:<42} {r['epsilon']:>6.3f}  [{r['label']}]  "
              f"{cs:>10.3f}  [{r.get('casc_label', '?')}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Production FP rate benchmark")
    parser.add_argument("--model",    default="gpt-4o",
                        help="Model name (default: gpt-4o)")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "together"],
                        help="API provider (default: openai)")
    parser.add_argument("--dry-run",  action="store_true",
                        help="Print prompts without calling the API")
    args = parser.parse_args()

    if args.dry_run:
        for p in PROMPTS:
            print(f"\n{'='*60}")
            print(f"[{p['type']}] {p['name']} ({p['file']})")
            print(build_prompt(p))
        return

    # Load .env from project root
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    if args.provider == "together":
        client = get_together_client()
    else:
        client = OpenAI()

    results = run_benchmark(client, args.model, args.provider)
    print_summary(results)

    # Save results
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    safe_model = args.model.replace("/", "_").replace(":", "_")
    out_file = out_dir / f"production_{safe_model}.json"
    payload = {
        "meta": {
            "source_repo": "tiangolo/full-stack-fastapi-template",
            "commit": "13652b51ea0acca7dfe243ac25e2bbdc066f3c4f",
            "model": args.model,
            "provider": args.provider,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "n_functions": len(PROMPTS),
            "n_api": sum(1 for p in PROMPTS if p["type"] == "API"),
            "n_logic": sum(1 for p in PROMPTS if p["type"] == "LOGIC"),
        },
        "results": results,
    }
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Results -> {out_file}")


if __name__ == "__main__":
    main()
