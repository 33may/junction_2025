from audio_pipeline import run_full_pipeline

def main():
    audio_file = "../data/out_dialogue.wav"
    run_full_pipeline(audio_file, save=True)


if __name__ == "__main__":
    main()




