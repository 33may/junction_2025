# # pyannote 4.x: run diarization and dump JSON
# from pyannote.audio import Pipeline
# from pyannote.core import Annotation
# from dotenv import load_dotenv
# import os, json, time, torch
#
# # --- load env ---
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# # --- load pipeline ---
# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-3.1",
#     token=HF_TOKEN
# ).to(device)
#
#
# audio_file = "../data/out_dialogue.wav"
# t0 = time.time()
# out = pipeline(audio_file)
# dt = round(time.time() - t0, 3)
#
# ann: Annotation = out.speaker_diarization if hasattr(out, "speaker_diarization") else out
#
#
# segments = []
# speakers = set()
# for seg, _, spk in ann.itertracks(yield_label=True):
#     segments.append({
#         "start": round(float(seg.start), 3),
#         "end":   round(float(seg.end), 3),
#         "speaker": spk
#     })
#     speakers.add(spk)
#
# # --- optional embeddings ---
# emb = out.speaker_embeddings.tolist() if hasattr(out, "speaker_embeddings") else None
#
# # --- save JSON ---
# payload = {
#     "segments": segments,
#     "metadata": {
#         "model": "pyannote/speaker-diarization-3.1",
#         "version": "4.x",
#         "processing_time": dt,
#         "total_speakers": len(speakers),
#         "speakers": sorted(speakers)
#     },
#     "embeddings": emb
# }
# os.makedirs("../data/outputs", exist_ok=True)
# with open("../data/outputs/diar.json", "w", encoding="utf-8") as f:
#     json.dump(payload, f, ensure_ascii=False, indent=2)
#
# print("Saved ../data/outputs/diar.json")



from pyannote.audio import Pipeline
from dotenv import load_dotenv
import os, json, time, torch

# --- load env ---
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_pipeline(token, device):
    # Create and move diarization pipeline to device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token
    ).to(device)
    return pipe

def diarize(pipeline, audio_file, save=False):
    out = pipeline(audio_file)
    ann = out.speaker_diarization
    segments = []
    speakers = set()
    for seg, _, spk in ann.itertracks(yield_label=True):
        segments.append({"start": float(seg.start), "end": float(seg.end), "speaker": spk})
        speakers.add(spk)

    if save:
        with open("../data/outputs/diar.json", "w", encoding="utf-8") as f:
            json.dump({"segments": segments, "speakers": sorted(speakers)}, f, ensure_ascii=False, indent=2)

    return {"segments": segments, "speakers": sorted(speakers)}

audio_file = "../data/out_dialogue.wav"

pipeline = create_pipeline(HF_TOKEN, device)

out = diarize(pipeline, audio_file, save=True)

