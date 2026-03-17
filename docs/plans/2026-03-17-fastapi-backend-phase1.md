# FastAPI Backend — Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-ready FastAPI backend with JWT auth, file management, and async ASR task queue as the foundation for replacing Streamlit.

**Architecture:** FastAPI (stateless HTTP) + Celery (async ML tasks) + Redis (broker) + PostgreSQL (data). All ML code (torch/funasr) runs exclusively in Celery workers, never in the API layer. API returns `task_id` immediately; frontend polls `/tasks/{id}` or streams SSE for progress.

**Tech Stack:** FastAPI 0.111, SQLAlchemy 2.0, Alembic, Celery 5, Redis 7, PostgreSQL 16, python-jose (JWT), pytest + httpx (tests), Docker Compose

---

## Project Layout

All new code lives in `evs-api/` inside the existing repo:

```
evs-api/
├── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py       # Settings from env
│   │   ├── database.py     # SQLAlchemy engine + session
│   │   └── security.py     # JWT encode/decode, password hash
│   ├── models/             # SQLAlchemy ORM models
│   │   ├── user.py
│   │   ├── task.py
│   │   └── asr_file.py
│   ├── schemas/            # Pydantic request/response shapes
│   │   ├── auth.py
│   │   ├── task.py
│   │   └── file.py
│   ├── routers/
│   │   ├── auth.py
│   │   ├── tasks.py
│   │   └── files.py
│   └── workers/
│       ├── celery_app.py
│       └── asr_tasks.py
├── alembic/
│   ├── env.py
│   └── versions/
├── tests/
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_tasks.py
│   └── test_files.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `evs-api/requirements.txt`
- Create: `evs-api/app/__init__.py` (empty)
- Create: `evs-api/app/main.py`
- Create: `evs-api/app/core/config.py`

**Step 1: Create directory structure**

```bash
cd evs-api
mkdir -p app/core app/models app/schemas app/routers app/workers
mkdir -p alembic/versions tests
touch app/__init__.py app/core/__init__.py app/models/__init__.py
touch app/schemas/__init__.py app/routers/__init__.py app/workers/__init__.py
```

**Step 2: Create `evs-api/requirements.txt`**

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
sqlalchemy==2.0.30
alembic==1.13.1
psycopg2-binary==2.9.9
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.9
celery==5.3.6
redis==5.0.4
httpx==0.27.0
pytest==8.2.0
pytest-asyncio==0.23.6
```

**Step 3: Create `evs-api/app/core/config.py`**

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://evs:evs@localhost:5432/evs"
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str = "change-me-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 8  # 8 hours
    AUDIO_UPLOAD_DIR: str = "/tmp/evs_audio"

    class Config:
        env_file = ".env"

settings = Settings()
```

**Step 4: Create `evs-api/app/main.py`**

```python
from fastapi import FastAPI
from app.routers import auth, tasks, files

app = FastAPI(title="EVS Navigation API", version="1.0.0")

app.include_router(auth.router,  prefix="/auth",  tags=["auth"])
app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
app.include_router(files.router, prefix="/files", tags=["files"])

@app.get("/health")
def health():
    return {"status": "ok"}
```

**Step 5: Write smoke test**

`evs-api/tests/test_health.py`:
```python
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
```

**Step 6: Run test**

```bash
cd evs-api
pip install -r requirements.txt
pytest tests/test_health.py -v
```
Expected: PASS

**Step 7: Commit**

```bash
git add evs-api/
git commit -m "feat(api): scaffold FastAPI project structure"
```

---

## Task 2: Database Models

**Files:**
- Create: `evs-api/app/core/database.py`
- Create: `evs-api/app/models/user.py`
- Create: `evs-api/app/models/task.py`
- Create: `evs-api/app/models/asr_file.py`

**Step 1: Create `evs-api/app/core/database.py`**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Step 2: Create `evs-api/app/models/user.py`**

```python
from sqlalchemy import Column, Integer, String, Boolean, DateTime, func
from app.core.database import Base

class User(Base):
    __tablename__ = "users"
    id         = Column(Integer, primary_key=True)
    email      = Column(String, unique=True, nullable=False, index=True)
    password   = Column(String, nullable=False)   # bcrypt hash
    is_admin   = Column(Boolean, default=False)
    is_active  = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now())
```

**Step 3: Create `evs-api/app/models/task.py`**

```python
import uuid
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, func
from app.core.database import Base

class Task(Base):
    __tablename__ = "tasks"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    type       = Column(String(20), nullable=False)   # 'asr', 'nlp', 'si'
    status     = Column(String(20), default="pending") # pending/running/done/failed
    progress   = Column(Integer, default=0)
    user_id    = Column(Integer, ForeignKey("users.id"))
    result_id  = Column(Integer, nullable=True)
    error      = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
```

**Step 4: Create `evs-api/app/models/asr_file.py`**

```python
from sqlalchemy import Column, Integer, String, Float, DateTime, func
from app.core.database import Base

class AsrFile(Base):
    __tablename__ = "asr_files"
    id             = Column(Integer, primary_key=True)
    file_name      = Column(String, nullable=False)
    lang           = Column(String(5), nullable=False)
    asr_provider   = Column(String(30), nullable=False)
    model          = Column(String(50), nullable=False)
    slice_duration = Column(Float, nullable=True)
    channel_num    = Column(Integer, nullable=True)
    audio_file     = Column(String, nullable=True)
    total_segments = Column(Integer, default=0)
    total_words    = Column(Integer, default=0)
    total_duration = Column(Float, nullable=True)
    created_at     = Column(DateTime, server_default=func.now())
    updated_at     = Column(DateTime, server_default=func.now(), onupdate=func.now())
```

**Step 5: Write model test**

`evs-api/tests/test_models.py`:
```python
import pytest
from app.models.user import User
from app.models.task import Task

def test_task_has_uuid_id():
    t = Task(type="asr", user_id=1)
    assert t.id is not None
    assert len(t.id) == 36   # UUID format

def test_task_default_status():
    t = Task(type="asr", user_id=1)
    assert t.status == "pending"
    assert t.progress == 0
```

**Step 6: Run test**

```bash
pytest tests/test_models.py -v
```
Expected: PASS (no DB needed for model instantiation tests)

**Step 7: Commit**

```bash
git add evs-api/app/core/database.py evs-api/app/models/
git commit -m "feat(api): add SQLAlchemy models for user, task, asr_file"
```

---

## Task 3: Alembic Migration Setup

**Files:**
- Create: `evs-api/alembic.ini`
- Create: `evs-api/alembic/env.py`
- Create: `evs-api/alembic/versions/0001_initial.py`

**Step 1: Init Alembic**

```bash
cd evs-api
alembic init alembic
```

**Step 2: Edit `evs-api/alembic/env.py`** — replace `target_metadata = None` with:

```python
from app.models.user import User      # noqa: F401
from app.models.task import Task      # noqa: F401
from app.models.asr_file import AsrFile  # noqa: F401
from app.core.database import Base
from app.core.config import settings

config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)
target_metadata = Base.metadata
```

**Step 3: Generate initial migration**

```bash
alembic revision --autogenerate -m "initial schema"
```

Expected: creates `alembic/versions/xxxx_initial_schema.py` with `op.create_table` calls for `users`, `tasks`, `asr_files`.

**Step 4: Apply migration (requires PostgreSQL running)**

```bash
# Start postgres via docker for local dev:
docker run -d --name evs-pg -e POSTGRES_USER=evs -e POSTGRES_PASSWORD=evs \
  -e POSTGRES_DB=evs -p 5432:5432 postgres:16
alembic upgrade head
```

Expected: `Running upgrade -> xxxx, initial schema`

**Step 5: Commit**

```bash
git add evs-api/alembic/
git commit -m "feat(api): add Alembic migrations for initial schema"
```

---

## Task 4: JWT Auth — Core Security

**Files:**
- Create: `evs-api/app/core/security.py`
- Test: `evs-api/tests/test_security.py`

**Step 1: Write failing test**

`evs-api/tests/test_security.py`:
```python
from app.core.security import create_access_token, decode_token, hash_password, verify_password

def test_password_round_trip():
    hashed = hash_password("secret123")
    assert hashed != "secret123"
    assert verify_password("secret123", hashed) is True
    assert verify_password("wrong", hashed) is False

def test_jwt_round_trip():
    token = create_access_token({"sub": "user@test.com"})
    payload = decode_token(token)
    assert payload["sub"] == "user@test.com"

def test_jwt_invalid_token_returns_none():
    result = decode_token("not.a.valid.token")
    assert result is None
```

**Step 2: Run to verify failures**

```bash
pytest tests/test_security.py -v
```
Expected: FAIL — `ImportError: cannot import name 'create_access_token'`

**Step 3: Create `evs-api/app/core/security.py`**

```python
from datetime import datetime, timedelta, UTC
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(UTC) + timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
```

**Step 4: Run tests**

```bash
pytest tests/test_security.py -v
```
Expected: 3 PASS

**Step 5: Commit**

```bash
git add evs-api/app/core/security.py evs-api/tests/test_security.py
git commit -m "feat(api): add JWT + bcrypt security utilities"
```

---

## Task 5: Auth Router (POST /auth/login)

**Files:**
- Create: `evs-api/app/schemas/auth.py`
- Create: `evs-api/app/routers/auth.py`
- Test: `evs-api/tests/test_auth.py`

**Step 1: Create `evs-api/app/schemas/auth.py`**

```python
from pydantic import BaseModel, EmailStr

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
```

**Step 2: Write failing test**

`evs-api/tests/test_auth.py`:
```python
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.core.database import Base, get_db
from app.core.security import hash_password
from app.models.user import User

TEST_DB = "sqlite:///./test.db"
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
```

**Step 3: Run to verify failures**

```bash
pytest tests/test_auth.py -v
```
Expected: FAIL — router not implemented

**Step 4: Create `evs-api/app/routers/auth.py`**

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import verify_password, create_access_token
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenResponse

router = APIRouter()

@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials")
    token = create_access_token({"sub": user.email, "is_admin": user.is_admin})
    return TokenResponse(access_token=token)
```

**Step 5: Run tests**

```bash
pytest tests/test_auth.py -v
```
Expected: 3 PASS

**Step 6: Commit**

```bash
git add evs-api/app/schemas/auth.py evs-api/app/routers/auth.py evs-api/tests/test_auth.py
git commit -m "feat(api): add JWT login endpoint with tests"
```

---

## Task 6: Auth Dependency (Protected Routes)

**Files:**
- Modify: `evs-api/app/core/security.py`
- Test: add to `evs-api/tests/test_auth.py`

**Step 1: Add `get_current_user` dependency to `security.py`**

```python
# Add to existing security.py:
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from app.core.database import get_db

bearer = HTTPBearer()

def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(bearer),
    db: Session = Depends(get_db)
):
    from app.models.user import User
    payload = decode_token(creds.credentials)
    if not payload:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired token")
    user = db.query(User).filter(User.email == payload["sub"]).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return user
```

**Step 2: Add test to `tests/test_auth.py`**

```python
def test_protected_route_without_token():
    # Use /files as a protected route (added in Task 8)
    r = client.get("/files")
    assert r.status_code == 403  # HTTPBearer returns 403 when no token

def test_protected_route_with_valid_token():
    login_r = client.post("/auth/login", json={"email": "test@evs.com", "password": "pass123"})
    token = login_r.json()["access_token"]
    r = client.get("/files", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
```

**Step 3: Run tests (will fail until Task 8)**

```bash
pytest tests/test_auth.py -v
```
Note: `test_protected_route_with_valid_token` will fail until `/files` is implemented — that's OK, come back after Task 8.

**Step 4: Commit**

```bash
git add evs-api/app/core/security.py evs-api/tests/test_auth.py
git commit -m "feat(api): add get_current_user auth dependency"
```

---

## Task 7: File Upload Router

**Files:**
- Create: `evs-api/app/schemas/file.py`
- Create: `evs-api/app/routers/files.py`
- Test: `evs-api/tests/test_files.py`

**Step 1: Create `evs-api/app/schemas/file.py`**

```python
from pydantic import BaseModel
from datetime import datetime

class FileResponse(BaseModel):
    id: int
    file_name: str
    lang: str
    asr_provider: str
    model: str
    total_segments: int
    total_words: int
    created_at: datetime

    class Config:
        from_attributes = True
```

**Step 2: Write failing test**

`evs-api/tests/test_files.py`:
```python
import io, pytest
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

def test_upload_audio_file():
    audio = io.BytesIO(b"fake-audio-content")
    r = client.post(
        "/files/upload",
        files={"file": ("test.wav", audio, "audio/wav")},
        data={"lang": "en"},
        headers=auth_header()
    )
    assert r.status_code == 200
    assert r.json()["file_name"] == "test.wav"
```

**Step 3: Run to verify failures**

```bash
pytest tests/test_files.py -v
```
Expected: FAIL

**Step 4: Create `evs-api/app/routers/files.py`**

```python
import os, shutil
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.core.config import settings
from app.models.user import User
from app.models.asr_file import AsrFile
from app.schemas.file import FileResponse

router = APIRouter()

@router.get("", response_model=list[FileResponse])
def list_files(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return db.query(AsrFile).order_by(AsrFile.created_at.desc()).all()

@router.post("/upload", response_model=FileResponse)
def upload_file(
    file: UploadFile = File(...),
    lang: str = Form(...),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user)
):
    os.makedirs(settings.AUDIO_UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(settings.AUDIO_UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    asr_file = AsrFile(
        file_name=file.filename,
        lang=lang,
        asr_provider="pending",
        model="pending",
        audio_file=dest,
    )
    db.add(asr_file)
    db.commit()
    db.refresh(asr_file)
    return asr_file
```

**Step 5: Run tests**

```bash
pytest tests/test_files.py -v
```
Expected: 2 PASS

**Step 6: Commit**

```bash
git add evs-api/app/schemas/file.py evs-api/app/routers/files.py evs-api/tests/test_files.py
git commit -m "feat(api): add file upload and listing endpoints"
```

---

## Task 8: Celery App + ASR Task

**Files:**
- Create: `evs-api/app/workers/celery_app.py`
- Create: `evs-api/app/workers/asr_tasks.py`
- Create: `evs-api/app/schemas/task.py`
- Create: `evs-api/app/routers/tasks.py`
- Test: `evs-api/tests/test_tasks.py`

**Step 1: Create `evs-api/app/workers/celery_app.py`**

```python
from celery import Celery
from app.core.config import settings

celery = Celery(
    "evs",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)
celery.conf.task_serializer = "json"
celery.conf.result_serializer = "json"
```

**Step 2: Create `evs-api/app/workers/asr_tasks.py`**

```python
import sys, os
# Allow importing existing app code (asr_utils, db_utils etc.)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))

from app.workers.celery_app import celery
from app.core.config import settings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

def _get_db_session():
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()

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
        task = db.query(Task).get(task_id)
        task.status = "running"
        task.progress = 5
        db.commit()

        # Import here so torch only loads in worker process
        from utils.asr_utils import ASRUtils
        import pandas as pd

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
            from save_asr_results import save_asr_result_to_database
            save_asr_result_to_database(words_df, segments_df)
            task.status = "done"
            task.progress = 100

        db.commit()
    except Exception as e:
        task = db.query(Task).get(task_id) if task else None
        if task:
            task.status = "failed"
            task.error = str(e)
            db.commit()
        raise
    finally:
        db.close()
```

**Step 3: Create `evs-api/app/schemas/task.py`**

```python
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
    result_id: Optional[int]
    error: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
```

**Step 4: Create `evs-api/app/routers/tasks.py`**

```python
import asyncio
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
    task = Task(type="asr", user_id=user.id)
    db.add(task)
    db.commit()
    db.refresh(task)

    from app.core.config import settings
    import os
    audio_path = os.path.join(settings.AUDIO_UPLOAD_DIR, body.file_name)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    run_asr.delay(task.id, audio_path, body.lang,
                  body.provider, body.model, body.file_name)
    return task

@router.get("/{task_id}", response_model=TaskResponse)
def get_task(task_id: str, db: Session = Depends(get_db),
             _: User = Depends(get_current_user)):
    task = db.query(Task).get(task_id)
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
            task = db.query(Task).get(task_id)
            if not task:
                yield "data: {\"error\": \"not found\"}\n\n"
                break
            yield f"data: {{\"status\": \"{task.status}\", \"progress\": {task.progress}}}\n\n"
            if task.status in ("done", "failed"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Step 5: Write task tests**

`evs-api/tests/test_tasks.py`:
```python
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
         patch("app.workers.asr_tasks.run_asr.delay") as mock_delay:
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
```

**Step 6: Run tests**

```bash
pytest tests/test_tasks.py -v
```
Expected: 3 PASS

**Step 7: Commit**

```bash
git add evs-api/app/workers/ evs-api/app/schemas/task.py \
        evs-api/app/routers/tasks.py evs-api/tests/test_tasks.py
git commit -m "feat(api): add Celery ASR task queue with submit/poll/stream endpoints"
```

---

## Task 9: Docker Compose

**Files:**
- Create: `evs-api/Dockerfile`
- Create: `evs-api/docker-compose.yml`
- Create: `evs-api/.env.example`

**Step 1: Create `evs-api/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2
RUN apt-get update && apt-get install -y libpq-dev gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Mount the parent repo so workers can import asr_utils, db_utils etc.
COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 2: Create `evs-api/docker-compose.yml`**

```yaml
version: "3.9"

services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: evs
      POSTGRES_PASSWORD: evs
      POSTGRES_DB: evs
    volumes:
      - pg_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  api:
    build: .
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    volumes:
      - ..:/repo          # Mount parent repo for asr_utils imports
      - audio_files:/tmp/evs_audio
    environment:
      - DATABASE_URL=postgresql://evs:evs@postgres:5432/evs
      - REDIS_URL=redis://redis:6379/0
      - AUDIO_UPLOAD_DIR=/tmp/evs_audio
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis

  worker-en:
    build: .
    command: celery -A app.workers.celery_app.celery worker -Q asr_en -c 1 --loglevel=info
    volumes:
      - ..:/repo
      - audio_files:/tmp/evs_audio
    environment:
      - DATABASE_URL=postgresql://evs:evs@postgres:5432/evs
      - REDIS_URL=redis://redis:6379/0
      - AUDIO_UPLOAD_DIR=/tmp/evs_audio
    depends_on:
      - redis
      - postgres
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  worker-zh:
    build: .
    command: celery -A app.workers.celery_app.celery worker -Q asr_zh -c 2 --loglevel=info
    volumes:
      - ..:/repo
      - audio_files:/tmp/evs_audio
    environment:
      - DATABASE_URL=postgresql://evs:evs@postgres:5432/evs
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - postgres

  worker-cpu:
    build: .
    command: celery -A app.workers.celery_app.celery worker -Q nlp,default -c 4 --loglevel=info
    volumes:
      - ..:/repo
    environment:
      - DATABASE_URL=postgresql://evs:evs@postgres:5432/evs
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - postgres

volumes:
  pg_data:
  audio_files:
```

**Step 3: Create `evs-api/.env.example`**

```
DATABASE_URL=postgresql://evs:evs@localhost:5432/evs
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=change-me-to-a-random-64-char-string
ACCESS_TOKEN_EXPIRE_MINUTES=480
AUDIO_UPLOAD_DIR=/tmp/evs_audio
```

**Step 4: Smoke test with Docker Compose**

```bash
cd evs-api
cp .env.example .env
docker compose up -d postgres redis
sleep 3
alembic upgrade head
docker compose up api
```

Open `http://localhost:8000/docs` — FastAPI Swagger UI should appear.

**Step 5: Run full test suite**

```bash
pytest tests/ -v
```
Expected: All tests PASS

**Step 6: Commit**

```bash
git add evs-api/Dockerfile evs-api/docker-compose.yml evs-api/.env.example
git commit -m "feat(api): add Docker Compose for full local stack"
```

---

## Task 10: Final Integration Test

**Step 1: Full end-to-end smoke test (manual)**

```bash
# Start stack
docker compose up -d

# 1. Login
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@evs.com","password":"admin"}' | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. Upload audio
curl -s -X POST http://localhost:8000/files/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/test.wav" -F "lang=en"

# 3. Submit ASR task
TASK_ID=$(curl -s -X POST http://localhost:8000/tasks/asr \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"file_name":"test.wav","lang":"en","provider":"crisperwhisper","model":"default"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['id'])")

# 4. Poll progress
curl -s http://localhost:8000/tasks/$TASK_ID \
  -H "Authorization: Bearer $TOKEN"
```

Expected: `{"status": "running", "progress": ...}`

**Step 2: Run complete test suite one final time**

```bash
pytest tests/ -v --tb=short
```
Expected: All PASS

**Step 3: Final commit**

```bash
git add .
git commit -m "feat(api): Phase 1 complete — FastAPI backend with auth, files, ASR task queue"
```

---

## Phase 2 (Next Plan)

Once Phase 1 is merged and deployed:
- **Phase 2**: Vue 3 frontend (5 views, Pinia stores, SSE progress)
- **Phase 3**: NLP + SI analysis Celery tasks
- **Phase 4**: PostgreSQL data migration from existing SQLite
