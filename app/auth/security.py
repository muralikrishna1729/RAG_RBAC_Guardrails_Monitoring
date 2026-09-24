import os 
import hashlib 

import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional 

from dotenv import load_dotenv

import jwt 

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "enterprise-super-secret-key-change-in-prod-2026")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))


def hash_password(password: str) -> str:
    """Hashes a password using bcrypt with a per-password random salt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies against a bcrypt hash (falls back to legacy SHA-256 static-salt hashes)."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except ValueError:
        # Legacy fallback: pre-bcrypt hashes were SHA-256 with a static salt.
        salt = "app_secure_salt_"
        return hashlib.sha256((salt + plain_password).encode("utf-8")).hexdigest() == hashed_password


def needs_rehash(hashed_password: str) -> bool:
    """True when the stored hash is not bcrypt and should be upgraded on next login."""
    return not (hashed_password or "").startswith("$2")
    
def create_jwt_token(data:dict, expires_delta: Optional[timedelta]=None)->str:
    """Generates signed JWT token (or structured token payload)."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt 

def decode_jwt_token(token:str)->Optional[dict]:
    """Validates signature and returns payload claims."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        print("Token has expired.")
        return None
    except jwt.InvalidTokenError:
        print("Invalid token signature or structure.")
        return None


