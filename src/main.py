from audio_pipeline import run_full_pipeline
from jury import judge
import os, random

def main():
    audio_file = "../data/out_dialogue.wav"
    run_full_pipeline(audio_file, save=True)


def test_judge():
    N = 5
    agent_dirs = [os.path.join("../data/agents", d) for d in os.listdir("../data/agents") if os.path.isdir(os.path.join("../data/agents", d))]
    personality_files = [path + "/scratch.json" for path in random.sample(agent_dirs, N)]
    statement = "Euthanasia should be legal."
    class_labels = ["extremist", "hate speech", "controversial", "normal"]
    final_output = judge(statement, class_labels, personality_files, rounds=3)
    print(final_output)

if __name__ == "__main__":
    test_judge()




