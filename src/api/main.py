from fastapi import FastAPI
from api.routes.metrics import router as metrics_router

app = FastAPI(title="Voice Agent API")

app.include_router(metrics_router, prefix="/metrics")


@app.get("/health")
async def health():
    return {"status": "ok"}
