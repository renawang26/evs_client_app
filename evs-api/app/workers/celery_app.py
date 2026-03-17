from celery import Celery
from app.core.config import settings

celery = Celery(
    "evs",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)
celery.conf.task_serializer = "json"
celery.conf.result_serializer = "json"
