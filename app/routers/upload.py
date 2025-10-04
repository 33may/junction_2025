from fastapi import APIRouter, UploadFile, File, HTTPException, status
from typing import Optional
import os
import uuid

# Try to load environment variables from a .env file if python-dotenv is available
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

router = APIRouter()

_CONTENT_TYPE_EXTENSION_MAP = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/flac": "flac",
    "audio/ogg": "ogg",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
    "audio/webm": "webm",
    "audio/opus": "opus",
    "audio/3gpp": "3gp",
}

def _get_upload_dir() -> str:
    # Prefer AUDIO_STORAGE_DIR; allow AUDIO_UPLOAD_DIR as a fallback; default "uploads"
    path = os.getenv("AUDIO_STORAGE_DIR") or os.getenv("AUDIO_UPLOAD_DIR") or "uploads"
    os.makedirs(path, exist_ok=True)
    return path

def _safe_ext(original_filename: Optional[str], content_type: Optional[str]) -> str:
    # Try from original filename
    if original_filename:
        ext = os.path.splitext(original_filename)[1].lower()
        if ext and len(ext) <= 10 and all(c.isalnum() or c == "." for c in ext):
            return ext
    # Fallback from content type
    if content_type and content_type in _CONTENT_TYPE_EXTENSION_MAP:
        return "." + _CONTENT_TYPE_EXTENSION_MAP[content_type]
    # As a last resort, no extension
    return ""

@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)) -> dict:
    # Validate content type
    if not file.content_type or not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only audio files are accepted.",
        )

    upload_dir = _get_upload_dir()

    guid = uuid.uuid4().hex  # simple GUID
    ext = _safe_ext(file.filename, file.content_type)
    target_path = os.path.join(upload_dir, f"{guid}{ext}")

    try:
        # Stream to disk in chunks
        chunk_size = 1024 * 1024  # 1MB
        with open(target_path, "wb") as out:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        # Clean up partial file if created
        with contextlib.suppress(Exception):
            if os.path.exists(target_path):
                os.remove(target_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        )
    finally:
        await file.close()

    # Return the GUID that matches the stored filename (without extension)
    return {"id": guid}

