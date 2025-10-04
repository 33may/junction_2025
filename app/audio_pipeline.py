from adsr_pipeline import create_model, transcribe
from diar_pipeline import create_pipeline, diarize
import torch
from dotenv import load_dotenv
import os, json
from compose import merge_transcript_and_diarization

def log_output(output):
    for item in output["utterances"]:
        print(f"[{item['speaker']}]   ({item['start']} - {item['end']})    |     {item['text']}")

def run_full_pipeline(audio_file,save=False):
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asr_model = create_model(device)

    diar_pipeline = create_pipeline(HF_TOKEN, device)

    transcript = transcribe(model=asr_model, audio_file=audio_file, save=save)

    diar = diarize(pipeline=diar_pipeline, audio_file=audio_file, save=save)

    output = merge_transcript_and_diarization(transcript=transcript["segments"], diar=diar["segments"])

    if save:
        with open("../data/outputs/full_pipeline_out.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    log_output(output)

def get_models(log: bool = False):
    """
    Initialize and return (ASR model, diarization pipeline).
    Used by FastAPI startup to preload models.
    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    asr_model = create_model(device)
    diar_pipeline = create_pipeline(hf_token, device)

    if log:
        print(f"[audio_pipeline] Models loaded on device: {device}")

    return asr_model, diar_pipeline
