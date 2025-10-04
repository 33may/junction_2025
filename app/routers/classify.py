from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from typing import Optional
import asyncio
import os
import glob
import json
import re
from functools import partial

from routers.upload import _get_upload_dir
from audio_pipeline import transcribe
from app.jury import astream_judge

router = APIRouter()

def _find_audio_path(guid: str) -> str:
    upload_dir = _get_upload_dir()
    candidates = glob.glob(os.path.join(upload_dir, f"{guid}.*"))
    if not candidates:
        raise FileNotFoundError(f"No file found for guid: {guid}")
    # Pick the first match deterministically (sorted)
    candidates.sort()
    return candidates[0]

def _sentence_spans(text: str):
    # Yields (start, end, sentence_text) with trimmed whitespace, preserving original indices.
    spans = []
    for m in re.finditer(r'[^.?!]+[.?!]*', text):
        s, e = m.span()
        # Trim whitespace within the span while keeping global indices
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if s < e:
            spans.append((s, e, text[s:e]))
    return spans

def _get_jury_config():
    # Class labels
    raw_labels = os.getenv("JURY_CLASS_LABELS", "")
    class_labels = [c.strip() for c in raw_labels.split(",") if c.strip()] or ["extremist", "hate speech", "controversial", "normal"]

    # Personality files: env or default directory
    raw_files = os.getenv("JURY_PERSONALITY_FILES", "")
    if raw_files.strip():
        files = [p.strip() for p in raw_files.split(",") if p.strip()]
    else:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "agents"))
        files = sorted(glob.glob(os.path.join(base, "*/scratch.json")))
    if not files:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="No jury personalities configured")
    return class_labels, files

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

            # Proceed with jury only if there is content
            if text.strip():
                try:
                    class_labels, personality_files = _get_jury_config()
                except HTTPException as e:
                    err = json.dumps({"error": e.detail})
                    yield f"event: error\ndata: {err}\n\n".encode("utf-8")
                    return

                # Split into sentences with indices
                spans = _sentence_spans(text)
                for idx, (start, end, sentence) in enumerate(spans):
                    # Notify client which sentence is being processed
                    data = json.dumps({"index": idx, "start": start, "end": end, "text": sentence})
                    yield f"event: jury_sentence\ndata: {data}\n\n".encode("utf-8")

                    # Stream agent outputs and final verdict for this sentence
                    async for evt in astream_judge(
                        statement=sentence,
                        class_labels=class_labels,
                        personality_files=personality_files,
                        rounds=int(os.getenv("JURY_ROUNDS", "3") or "3"),
                        model=os.getenv("JURY_OPENAI_MODEL"),
                        temperature=float(os.getenv("JURY_TEMPERATURE", "0.2") or "0.2"),
                    ):
                        if evt.get("type") == "agent_output":
                            out = {
                                "index": idx,
                                "round": evt["round"],
                                "agent": evt["agent"],
                                "classification": evt["classification"],
                                "rationale": evt["rationale"],
                                "facts": evt.get("facts", []),
                            }
                            yield f"event: jury_agent\ndata: {json.dumps(out)}\n\n".encode("utf-8")
                        elif evt.get("type") == "final_decision":
                            ver = {
                                "index": idx,
                                "final_classification": evt["final_classification"],
                                "unanimous_round": evt.get("unanimous_round"),
                                "final_votes": evt.get("final_votes", {}),
                            }
                            yield f"event: jury_verdict\ndata: {json.dumps(ver)}\n\n".encode("utf-8")

        except Exception as e:
            # Stream error to client, then end
            err = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {err}\n\n".encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")
