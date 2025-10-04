import json
import time

import torch
import whisper

print(whisper.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model = whisper.load_model("medium", device=device)

audio_file = "../data/Want to become fluent in Eglish.mp3"

print("start processing")

time_start = time.time()

result = model.transcribe(audio_file)

print(f"processed time: {time.time() - time_start}")

with open("../data/outputs/transcription.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print(result["text"])

print(33)
