import os 
import hashlib 
from datetime import datetime, timedelta, timezone
from typing import Optional 
import jwt 

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "enterprise-super-secret-key-change-in-prod-2026")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60)


def hash_password(password: str) -> str:
    """Hashes a password using SHA-256 with a static salt."""
    salt = "app_secure_salt_"
    salted_password = salt + password
    return hashlib.sha256(salted_password.encode("utf-8")).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password
    
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


