import enum
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Enum, Text
from app.auth.database import Base

class RoleEnum(str, enum.Enum):
    ENGINEERING = "engineering"
    FINANCE = "finance"
    HR = "hr"
    MARKETING = "marketing"
    GENERAL = "general"
    ADMIN = "admin"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role = Column(Enum(RoleEnum), default=RoleEnum.GENERAL, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), nullable=False)
    role = Column(String(50), nullable=False)
    action = Column(String(50), nullable=False) # e.g., 'CHAT', 'LOGIN'
    query = Column(Text, nullable=True)
    status = Column(String(50), nullable=False) # 'PASSED', 'BLOCKED'
    timestamp = Column(DateTime, default=datetime.utcnow)
