from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os, json, torch

# --- load env ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_pipeline(token, device, log=True):
    # Create and move diarization pipeline to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if log:
        print("Start creating diar pipeline \n ================================")
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token
    ).to(device)

    if log:
        print("Diar pipeline created \n ================================")
    return pipe

def diarize(pipeline, audio_file, save=False, log=True):
    if log:
        print("Diar processing started \n ================================")
    out = pipeline(audio_file)
    ann = out.speaker_diarization
    segments = []
    speakers = set()
    for seg, _, spk in ann.itertracks(yield_label=True):
        segments.append({"start": float(seg.start), "end": float(seg.end), "speaker": spk})
        speakers.add(spk)

    if save:
        if log:
            print(f"Saving DIAR output \n ================================")
        with open("../data/outputs/diar.json", "w", encoding="utf-8") as f:
            json.dump({"segments": segments, "speakers": sorted(speakers)}, f, ensure_ascii=False, indent=2)

    if log:
        print("Diar processing completed \n ================================")

    return {"segments": segments, "speakers": sorted(speakers)}

