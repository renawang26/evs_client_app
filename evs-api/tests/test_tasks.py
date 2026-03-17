import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.core.database import Base, get_db
from app.core.security import hash_password, create_access_token
from app.models.user import User
from app.models.task import Task

TEST_DB = "sqlite:///./test_tasks.db"
engine = create_engine(TEST_DB, connect_args={"check_same_thread": False})
TestSession = sessionmaker(bind=engine)


def override_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup(tmp_path):
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestSession()
    db.add(User(email="u@evs.com", password=hash_password("pw")))
    db.commit()
    db.close()
    app.dependency_overrides[get_db] = override_db
    yield
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


def auth_header():
    return {"Authorization": f"Bearer {create_access_token({'sub': 'u@evs.com', 'is_admin': False})}"}


client = TestClient(app)


def test_submit_asr_returns_task_id(tmp_path):
    # Create fake audio file
    audio = tmp_path / "test.wav"
    audio.write_bytes(b"fake")

    with patch("app.core.config.settings.AUDIO_UPLOAD_DIR", str(tmp_path)), \
         patch("app.routers.tasks.run_asr.delay") as mock_delay:
        r = client.post("/tasks/asr", json={
            "file_name": "test.wav", "lang": "en",
            "provider": "crisperwhisper", "model": "default"
        }, headers=auth_header())

    assert r.status_code == 200
    data = r.json()
    assert "id" in data
    assert data["status"] == "pending"
    mock_delay.assert_called_once()


def test_get_task_not_found():
    r = client.get("/tasks/nonexistent-id", headers=auth_header())
    assert r.status_code == 404


def test_get_task_status():
    db = TestSession()
    task = Task(type="asr", status="running", progress=50, user_id=1)
    db.add(task)
    db.commit()
    task_id = task.id
    db.close()

    r = client.get(f"/tasks/{task_id}", headers=auth_header())
    assert r.status_code == 200
    assert r.json()["progress"] == 50
    assert r.json()["status"] == "running"
