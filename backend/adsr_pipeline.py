import json
import time

import torch
import whisper

def create_model(device, log=True):
    if log:
        print("Creating ASR model \n ================================")
    model = whisper.load_model("medium", device=device)
    if log:
        print("ASR model Created \n ================================")
    return model

def transcribe(model, audio_file, save=False, log=True):
    if log:
        print("Start ASR Transcription \n ================================")

    time_start = time.time()

    result = model.transcribe(audio_file)

    if log:
        print(f" ASR Transcription done in time: {time.time() - time_start} \n ================================")

    if save:
        if log:
            print(f"Saving ASR Transcription \n ================================")
        with open("../data/outputs/transcription.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# print(device)
#
# audio_file = "../data/Want to become fluent in Eglish.mp3"
#
# model = create_model(device)
#
# transcript = transcribe(audio_file, save=True)
#
# print(transcript["text"])