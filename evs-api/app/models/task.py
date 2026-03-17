import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class Task(Base):
    __tablename__ = "tasks"
    # Required fields (no default) come first in dataclass order
    type:       Mapped[str]           = mapped_column(String(20), nullable=False)   # 'asr', 'nlp', 'si'
    # Fields with defaults
    user_id:    Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), default=None)
    status:     Mapped[str]           = mapped_column(String(20), default="pending")  # pending/running/done/failed
    progress:   Mapped[int]           = mapped_column(Integer, default=0)
    result_id:  Mapped[Optional[int]] = mapped_column(Integer, nullable=True, default=None)
    error:      Mapped[Optional[str]] = mapped_column(String, nullable=True, default=None)
    # Auto-generated / server fields (init=False)
    id:         Mapped[str]                    = mapped_column(String, primary_key=True, default_factory=lambda: str(uuid.uuid4()), init=False)
    created_at: Mapped[Optional[datetime]]     = mapped_column(DateTime, server_default=func.now(), init=False, default=None)
    updated_at: Mapped[Optional[datetime]]     = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), init=False, default=None)
