import os
import shutil
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User
from app.models.asr_file import AsrFile
from app.schemas.file import FileResponse

router = APIRouter()


@router.get("", response_model=List[FileResponse])
def list_files(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user)
):
    return db.query(AsrFile).order_by(AsrFile.created_at.desc()).all()


@router.post("/upload", response_model=FileResponse)
def upload_file(
    file: UploadFile = File(...),
    lang: str = Form(...),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user)
):
    safe_name = os.path.basename(file.filename or "")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(settings.AUDIO_UPLOAD_DIR, safe_name)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    asr_file = AsrFile(
        file_name=safe_name,
        lang=lang,
        asr_provider="pending",
        model="pending",
        audio_file=dest,
    )
    db.add(asr_file)
    db.commit()
    db.refresh(asr_file)
    return asr_file
