from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class TaskSubmitRequest(BaseModel):
    file_name: str
    lang: str           # 'en' or 'zh'
    provider: str       # 'crisperwhisper' or 'funasr'
    model: str


class TaskResponse(BaseModel):
    id: str
    type: str
    status: str
    progress: int
    result_id: Optional[int] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None

    model_config = {"from_attributes": True}
