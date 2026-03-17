from datetime import datetime
from typing import Optional
from sqlalchemy import String, Integer, Float, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column
from app.core.database import Base


class AsrFile(Base):
    __tablename__ = "asr_files"
    # Required fields (no default) come first in dataclass order
    file_name:      Mapped[str]                = mapped_column(String, nullable=False)
    lang:           Mapped[str]                = mapped_column(String(5), nullable=False)
    asr_provider:   Mapped[str]                = mapped_column(String(30), nullable=False)
    model:          Mapped[str]                = mapped_column(String(50), nullable=False)
    # Fields with defaults
    slice_duration: Mapped[Optional[float]]    = mapped_column(Float, nullable=True, default=None)
    channel_num:    Mapped[Optional[int]]      = mapped_column(Integer, nullable=True, default=None)
    audio_file:     Mapped[Optional[str]]      = mapped_column(String, nullable=True, default=None)
    total_segments: Mapped[int]                = mapped_column(Integer, default=0)
    total_words:    Mapped[int]                = mapped_column(Integer, default=0)
    total_duration: Mapped[Optional[float]]    = mapped_column(Float, nullable=True, default=None)
    # Auto-generated / server fields (init=False)
    id:             Mapped[int]                = mapped_column(Integer, primary_key=True, init=False, default=None)
    created_at:     Mapped[Optional[datetime]] = mapped_column(DateTime, server_default=func.now(), init=False, default=None)
    updated_at:     Mapped[Optional[datetime]] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), init=False, default=None)
