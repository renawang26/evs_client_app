from fastapi import FastAPI
from app.routers import auth, tasks, files

app = FastAPI(title="EVS Navigation API", version="1.0.0")

app.include_router(auth.router,  prefix="/auth",  tags=["auth"])
app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
app.include_router(files.router, prefix="/files", tags=["files"])

@app.get("/health")
def health():
    return {"status": "ok"}
