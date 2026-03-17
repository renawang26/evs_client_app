from typing import Optional
from sqlalchemy import String, Integer, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class User(Base):
    __tablename__ = "users"
    # Required fields (no default) come first in dataclass order
    email:      Mapped[str]           = mapped_column(String, unique=True, nullable=False, index=True)
    password:   Mapped[str]           = mapped_column(String, nullable=False)  # bcrypt hash
    # Fields with defaults
    id:         Mapped[int]           = mapped_column(Integer, primary_key=True, init=False, default=None)
    is_admin:   Mapped[bool]          = mapped_column(Boolean, default=False)
    is_active:  Mapped[bool]          = mapped_column(Boolean, default=True)
    created_at: Mapped[Optional[str]] = mapped_column(DateTime, server_default=func.now(), init=False, default=None)
