from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.chat import router as chat_router
from api.routes.metrics import router as metrics_router

app = FastAPI(title="Ayush Persona API")

# CORS — allow all origins for now; tighten to Vercel domain before submission.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(metrics_router, prefix="/metrics")
app.include_router(chat_router, prefix="/chat")


@app.get("/health")
async def health():
    return {"status": "ok"}

_PUBLIC = Path(__file__).resolve().parent.parent.parent / "public"
if _PUBLIC.exists():
    app.mount("/", StaticFiles(directory=str(_PUBLIC), html=True), name="static")
