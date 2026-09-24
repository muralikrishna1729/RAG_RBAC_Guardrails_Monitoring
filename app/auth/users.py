import hashlib
import time
from typing import  Dict, List
from app.auth.database import Base, engine, SessionLocal
from app.auth.models import User, RoleEnum
from app.auth.security import hash_password, verify_password, create_jwt_token

# Automatically create tables if not present
Base.metadata.create_all(bind = engine)

def seed_default_users():
    """Seeds default enterprise test users into SQLite/PostgreSQL on initial run."""
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
            for uname, pwd, role in demo_accounts:
                user = User(username=uname, hashed_password= hash_password(pwd), role = role)
                db.add(user)
            db.commit()
            print("Successfully seeded default enterprise users into database.")
    finally:
        db.close()

def verify_credentials(username:str, password:str)->bool:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username.lower()).first()
        if not user:
            return False 
        return verify_password(password, user.hashed_password)
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
    """Retrieves all users and their respective roles from the database."""
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return [{"username": user.username, "role": user.role.value} for user in users]
    finally:
        db.close()

if __name__ == "__main__":
    seed_default_users()
    test_users = ["alice", "sagar", "unknown_user"]
    print("\n--- Testing User Roles ---")
    for user in test_users:
        role = get_user_role(user)
        print(f"User: {user} -> Role: {role}")
