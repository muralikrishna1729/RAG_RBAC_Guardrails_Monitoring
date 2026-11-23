import os
from typing import Dict, List

from app.auth.database import Base, engine, SessionLocal
from app.auth.models import User, RoleEnum
from app.auth.security import hash_password, needs_rehash, verify_password, create_jwt_token

# Keep the current lightweight local bootstrap. Schema evolution will be handled
# separately through migrations; create_all does not replace a migration system.
Base.metadata.create_all(bind=engine)

APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
ENABLE_DEMO_USERS = os.getenv(
    "ENABLE_DEMO_USERS", "true" if APP_ENV in {"development", "test"} else "false"
).strip().lower() in {"1", "true", "yes", "on"}


def seed_default_users():
    """Seed demo users only when explicitly enabled for a non-production environment."""
    if APP_ENV == "production" or not ENABLE_DEMO_USERS:
        return

    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            demo_accounts = [
                ("alice", "finance123", RoleEnum.FINANCE),
                ("bob", "hr123", RoleEnum.HR),
                ("david", "engineering123", RoleEnum.ENGINEERING),
                ("charlie", "marketing123", RoleEnum.MARKETING),
                ("eve", "general123", RoleEnum.GENERAL),
                ("sagar", "admin123", RoleEnum.ADMIN),
            ]
            for username, password, role in demo_accounts:
                db.add(User(
                    username=username,
                    hashed_password=hash_password(password),
                    role=role,
                ))
            db.commit()
            print("Successfully seeded development demo users.")
    finally:
        db.close()


# Auto-seed is restricted to development/test and can be disabled with ENABLE_DEMO_USERS=false.
seed_default_users()


def create_access_token(username: str, role: str) -> str:
    """Creates a signed access token carrying identity and an informational role claim."""
    return create_jwt_token({"sub": username, "role": role})


def verify_credentials(username: str, password: str) -> bool:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username.lower()).first()
        if not user or not verify_password(password, user.hashed_password):
            return False
        if needs_rehash(user.hashed_password):
            user.hashed_password = hash_password(password)
            db.commit()
        return True
    finally:
        db.close()


def get_user_role(username: str) -> str:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username.lower()).first()
        return user.role.value if user else None
    finally:
        db.close()


def get_all_demo_users() -> List[Dict[str, str]]:
    """Retrieves usernames and roles. Does not return credentials or password hashes."""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return [{"username": user.username, "role": user.role.value} for user in users]
    finally:
        db.close()


if __name__ == "__main__":
    seed_default_users()
    for username in ["alice", "sagar", "unknown_user"]:
        print(f"User: {username} -> Role: {get_user_role(username)}")
