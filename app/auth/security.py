import os
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from dotenv import load_dotenv

load_dotenv()

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
_configured_secret = os.getenv("JWT_SECRET_KEY", "").strip()

if APP_ENV == "production":
    if len(_configured_secret) < 32:
        raise RuntimeError(
            "JWT_SECRET_KEY must be configured with at least 32 characters in production."
        )
    SECRET_KEY = _configured_secret
else:
    # Development-only fallback. Never use this value for a deployed environment.
    SECRET_KEY = _configured_secret or "development-only-insecure-jwt-key-change-before-deployment"
    if not _configured_secret:
        logging.warning("JWT_SECRET_KEY is unset; using the development-only fallback.")

ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))


def hash_password(password: str) -> str:
    """Hashes a password using bcrypt with a per-password random salt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies bcrypt hashes and supports legacy SHA-256 hashes during migration."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        salt = "app_secure_salt_"
        return hashlib.sha256((salt + plain_password).encode("utf-8")).hexdigest() == hashed_password


def needs_rehash(hashed_password: str) -> bool:
    """True when the stored hash is not bcrypt and should be upgraded on next login."""
    return not (hashed_password or "").startswith("$2")


def create_jwt_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a signed JWT token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_jwt_token(token: str) -> Optional[dict]:
    """Validates the token signature and returns its payload."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.InvalidTokenError:
        return None
