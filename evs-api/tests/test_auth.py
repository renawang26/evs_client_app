import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.core.database import Base, get_db
from app.core.security import hash_password
from app.models.user import User

TEST_DB = "sqlite:///./test_auth.db"
engine = create_engine(TEST_DB, connect_args={"check_same_thread": False})
TestSession = sessionmaker(bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = TestSession()
    db.add(User(email="test@evs.com", password=hash_password("pass123")))
    db.commit()
    db.close()
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


client = TestClient(app)


def test_login_success():
    r = client.post("/auth/login", json={"email": "test@evs.com", "password": "pass123"})
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_login_wrong_password():
    r = client.post("/auth/login", json={"email": "test@evs.com", "password": "wrong"})
    assert r.status_code == 401


def test_login_unknown_user():
    r = client.post("/auth/login", json={"email": "nobody@evs.com", "password": "pass"})
    assert r.status_code == 401


def test_protected_route_without_token():
    r = client.get("/files")
    assert r.status_code == 403  # HTTPBearer returns 403 when no Authorization header


def test_protected_route_with_valid_token():
    login_r = client.post("/auth/login", json={"email": "test@evs.com", "password": "pass123"})
    token = login_r.json()["access_token"]
    r = client.get("/files", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
