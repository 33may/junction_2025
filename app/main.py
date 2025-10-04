from fastapi import FastAPI
from routers.classify import router as classify_router
from audio_pipeline import get_models
from routers.upload import router as upload_router

app = FastAPI(title="Junction 2025 - Audio Classifier API")

@app.on_event("startup")
def _load_models():
    # Preload ASR and diarization once
    asr, diar = get_models(log=False)
    app.state.asr_model = asr
    app.state.diar_pipeline = diar

# Mount routes
app.include_router(classify_router, prefix="/api")
app.include_router(upload_router, prefix="/api")

# Optional health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}
