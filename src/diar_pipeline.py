from pyannote.audio import Pipeline
import torch, json

from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HF_TOKEN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token
).to(device)

audio_file = "../data/out.wav"
diarization = pipeline(audio_file)


segments = []
speakers = set()

for turn, _, speaker in diarization.itertracks(yield_label=True):
    segments.append({
        "start": round(turn.start, 2),
        "end": round(turn.end, 2),
        "speaker": speaker
    })
    speakers.add(speaker)

result = {
    "segments": segments,
    "metadata": {
        "model": "pyannote/speaker-diarization-3.1",
        "total_speakers": len(speakers),
        "speakers": list(speakers)
    }
}

# save JSON
with open("../data/outputs/diar.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Saved diarization.json")
