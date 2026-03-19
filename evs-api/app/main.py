from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from app.routers import auth, tasks, files

app = FastAPI(title="EVS Navigation API", version="1.0.0")

app.include_router(auth.router,  prefix="/auth",  tags=["auth"])
app.include_router(tasks.router, prefix="/tasks", tags=["tasks"])
app.include_router(files.router, prefix="/files", tags=["files"])

# Serve evs_annotator/ web component at /static/annotator/
_annotator_path = Path(__file__).parent.parent.parent / "evs_annotator"
if _annotator_path.exists():
    app.mount("/static/annotator", StaticFiles(directory=str(_annotator_path)), name="annotator")

# Serve built Vue SPA — must come AFTER API routes
_frontend_path = Path(__file__).parent.parent / "static" / "frontend"
if _frontend_path.exists():
    app.mount("/app", StaticFiles(directory=str(_frontend_path), html=True), name="frontend")

@app.get("/health")
def health():
    return {"status": "ok"}
