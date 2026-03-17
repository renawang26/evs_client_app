import asyncio
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.task import Task
from app.schemas.task import TaskSubmitRequest, TaskResponse
from app.workers.asr_tasks import run_asr

router = APIRouter()


@router.post("/asr", response_model=TaskResponse)
def submit_asr_task(
    body: TaskSubmitRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user)
):
    from app.core.config import settings
    audio_path = os.path.join(settings.AUDIO_UPLOAD_DIR, body.file_name)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    task = Task(type="asr", user_id=user.id)
    db.add(task)
    db.commit()
    db.refresh(task)

    run_asr.delay(task.id, audio_path, body.lang,
                  body.provider, body.model, body.file_name)
    return task


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user)
):
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.get("/{task_id}/stream")
async def stream_task_progress(
    task_id: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user)
):
    async def event_generator():
        while True:
            task = db.query(Task).filter(Task.id == task_id).first()
            if not task:
                yield 'data: {"error": "not found"}\n\n'
                break
            yield f'data: {{"status": "{task.status}", "progress": {task.progress}}}\n\n'
            if task.status in ("done", "failed"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
