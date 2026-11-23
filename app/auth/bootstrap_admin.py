"""Create an initial administrator account without shipping default credentials."""
import getpass
import re

from app.auth.database import Base, engine, SessionLocal
from app.auth.models import User, RoleEnum
from app.auth.security import hash_password


def main():
    Base.metadata.create_all(bind=engine)

    username = input("Admin username: ").strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,50}", username):
        raise SystemExit("Username must be 3-50 characters: letters, numbers, dot, underscore or hyphen.")

    password = getpass.getpass("Admin password (minimum 12 characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if len(password) < 12:
        raise SystemExit("Password must contain at least 12 characters.")
    if password != confirmation:
        raise SystemExit("Passwords do not match.")

    db = SessionLocal()
    try:
        if db.query(User).filter(User.username == username).first():
            raise SystemExit("That username already exists.")
        db.add(User(
            username=username,
            hashed_password=hash_password(password),
            role=RoleEnum.ADMIN,
        ))
        db.commit()
        print(f"Administrator account '{username}' created.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
