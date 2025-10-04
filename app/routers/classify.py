from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio
import os
import glob
import json
from functools import partial

from routers.upload import _get_upload_dir
from audio_pipeline import transcribe

router = APIRouter()

def _find_audio_path(guid: str) -> str:
    upload_dir = _get_upload_dir()
    candidates = glob.glob(os.path.join(upload_dir, f"{guid}.*"))
    if not candidates:
        raise FileNotFoundError(f"No file found for guid: {guid}")
    # Pick the first match deterministically (sorted)
    candidates.sort()
    return candidates[0]

@router.get("/classify/{guid}")
async def classify(guid: str, request: Request):
    # Resolve file path
    try:
        audio_path = _find_audio_path(guid)
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Audio file not found")

    # Get preloaded ASR model
    asr_model = getattr(request.app.state, "asr_model", None)
    if asr_model is None:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="ASR model not loaded")

    async def event_stream():
        # Start transcription in a background thread
        fn = partial(transcribe, model=asr_model, audio_file=audio_path, save=False)
        task = asyncio.create_task(asyncio.to_thread(fn))

        # Periodic pings while transcribing
        try:
            while not task.done():
                yield b"event: ping\ndata: keep-alive\n\n"
                await asyncio.sleep(1.0)

            # Await result
            result = await task

            # Extract text from result
            text: Optional[str] = None
            if isinstance(result, dict):
                if isinstance(result.get("text"), str):
                    text = result["text"]
                elif isinstance(result.get("segments"), list):
                    try:
                        text = " ".join((seg.get("text") or "").strip() for seg in result["segments"] if isinstance(seg, dict))
                    except Exception:
                        text = None
            if not text:
                text = ""

            payload = json.dumps({"text": text})
            yield f"event: result\ndata: {payload}\n\n".encode("utf-8")
        except Exception as e:
            # Stream error to client, then end
            err = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {err}\n\n".encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")

