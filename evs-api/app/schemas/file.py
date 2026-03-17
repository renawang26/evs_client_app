from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class FileResponse(BaseModel):
    id: int
    file_name: str
    lang: str
    asr_provider: str
    model: str
    total_segments: int
    total_words: int
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
