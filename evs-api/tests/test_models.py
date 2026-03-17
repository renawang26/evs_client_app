import pytest
from app.models.user import User
from app.models.task import Task
from app.models.asr_file import AsrFile


def test_task_has_uuid_id():
    t = Task(type="asr", user_id=1)
    assert t.id is not None
    assert len(t.id) == 36   # UUID format: 8-4-4-4-12


def test_task_default_status():
    t = Task(type="asr", user_id=1)
    assert t.status == "pending"
    assert t.progress == 0


def test_user_model_fields():
    u = User(email="test@evs.com", password="hashed")
    assert u.email == "test@evs.com"
    assert u.is_admin is False
    assert u.is_active is True


def test_asr_file_model_fields():
    f = AsrFile(file_name="test.wav", lang="en", asr_provider="crisperwhisper", model="default")
    assert f.file_name == "test.wav"
    assert f.total_segments == 0
    assert f.total_words == 0
