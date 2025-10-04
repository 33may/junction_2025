import json

from audio_pipeline import run_full_pipeline
from jury import judge
import os, random
from generate_personas import get_set_of_personas

# def main():
#     audio_file = "../data/out_dialogue.wav"
#     run_full_pipeline(audio_file, save=True)


def test_judge():
    N = 5
    # agent_dirs = [os.path.join("../data/agents", d) for d in os.listdir("../data/agents") if os.path.isdir(os.path.join("../data/agents", d))]
    # personality_files = [path + "/scratch.json" for path in random.sample(agent_dirs, N)]

    personalities = get_set_of_personas(N, personas_path = "../data/personas_interesting.json")

    statement = "“Abortion must be legal. Period. This isn’t about cold statistics — it’s about bodies, pain, and freedom of choice. My friend sobbed in a hospital when the doctor said: ‘If we don’t terminate this pregnancy, you could die.’ Her husband held her hand, and in that moment there were no slogans, no outsiders’ opinions — only her life. We are talking about real people living with the consequences of rape, incest, and medical emergencies. We are talking about 14-year-old girls, about women who already have children and can’t even pay rent. Bans don’t make the world kinder — they make procedures deadlier, women poorer, and tragedies quieter. A body is not public property. Neither the church, nor the state, nor the neighbors have the right to decide what happens inside someone else’s body. Freedom without bodily autonomy is just an empty sign. I’m sick of being called “immoral” by the same people who refuse to support paid leave, affordable childcare, healthcare, or help for survivors. That’s hypocrisy. I am not asking everyone to like abortion. I am demanding the right not to die because of someone else’s dogma. I am demanding that doctors and patients make decisions without the fear of prison. I am demanding that poverty not turn women into incubators. Legal abortion is not about “irresponsibility.” It is about compassion and respect for the lives that already exist. No excuses. No stigma. No government hand on my body.”"
    final_output = judge(statement, personalities, rounds=3)
    print(final_output)

if __name__ == "__main__":
    test_judge()




