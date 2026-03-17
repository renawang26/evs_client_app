import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.core.database import Base, get_db
from app.core.security import hash_password, create_access_token
from app.models.user import User

TEST_DB = "sqlite:///./test_files.db"
engine = create_engine(TEST_DB, connect_args={"check_same_thread": False})
TestSession = sessionmaker(bind=engine)


def override_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup():
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
    token = create_access_token({"sub": "u@evs.com", "is_admin": False})
    return {"Authorization": f"Bearer {token}"}


client = TestClient(app)


def test_get_files_empty():
    r = client.get("/files", headers=auth_header())
    assert r.status_code == 200
    assert r.json() == []


def test_upload_audio_file(tmp_path):
    audio = io.BytesIO(b"fake-audio-content")
    r = client.post(
        "/files/upload",
        files={"file": ("test.wav", audio, "audio/wav")},
        data={"lang": "en"},
        headers=auth_header()
    )
    assert r.status_code == 200
    body = r.json()
    assert body["file_name"] == "test.wav"
    assert body["lang"] == "en"
    assert body["asr_provider"] == "pending"
