import sys
import os
# Allow importing existing Streamlit app code (asr_utils, db_utils etc.) from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from app.workers.celery_app import celery
from app.core.config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Module-level engine (shared pool across all tasks — avoids per-task pool exhaustion)
_engine = create_engine(settings.DATABASE_URL)
_SessionFactory = sessionmaker(bind=_engine)


def _get_db_session():
    return _SessionFactory()


@celery.task(bind=True)
def run_asr(self, task_id: str, audio_path: str, lang: str,
            provider: str, model: str, file_name: str):
    """
    Wrap existing ASRUtils.transcribe_audio() as a Celery task.
    Updates task.progress in DB during processing.
    """
    db = _get_db_session()
    try:
        from app.models.task import Task
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return
        task.status = "running"
        task.progress = 5
        db.commit()

        # Import torch-dependent code here so it only loads in worker process
        from utils.asr_utils import ASRUtils

        task.progress = 10
        db.commit()

        words_df, segments_df = ASRUtils.transcribe_audio(
            audio_path, provider, model, lang, file_name, None, None, 1
        )

        task.progress = 90
        db.commit()

        if words_df is None or words_df.empty:
            task.status = "failed"
            task.error = "No transcription results"
        else:
            # save_asr_results.py calls st.error() in error paths (Streamlit-dependent).
            # Wrap in try/except so a save failure does not discard the completed transcription.
            # TODO Phase 2: replace with worker-safe persistence via EVSDataUtils directly.
            try:
                from save_asr_results import save_asr_result_to_database
                save_asr_result_to_database(words_df, segments_df)
            except Exception as save_err:
                # Log but do not fail the task — transcription succeeded even if save failed
                task.error = f"Save warning: {str(save_err)[:200]}"
            task.status = "done"
            task.progress = 100

        db.commit()
    except Exception as e:
        db.rollback()
        from app.models.task import Task as TaskModel
        t = db.query(TaskModel).filter(TaskModel.id == task_id).first()
        if t:
            t.status = "failed"
            t.error = str(e)[:500]  # truncate long errors
            db.commit()
        raise
    finally:
        db.close()
