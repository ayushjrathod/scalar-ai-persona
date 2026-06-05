from fastapi import FastAPI

app = FastAPI(title="Voice Agent API")

@app.get("/health")
async def health():
    return {"status": "ok"}
