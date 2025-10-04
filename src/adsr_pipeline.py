import json
import time

import torch
import whisper

def create_model(device):
    model = whisper.load_model("medium", device=device)
    return model

def transcribe(audio_file, save=False):

    print("start processing")

    time_start = time.time()

    result = model.transcribe(audio_file)

    print(f"processed time: {time.time() - time_start}")

    if save:
        with open("../data/outputs/transcription.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

audio_file = "../data/Want to become fluent in Eglish.mp3"

model = create_model(device)

transcript = transcribe(audio_file, save=True)

print(transcript["text"])