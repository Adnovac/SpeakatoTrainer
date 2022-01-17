import datetime
import argparse
from SpeakatoTrainer import SpeakatoTrainer

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", dest = "model", default = "", help="Model path")
parser.add_argument("-d", "--dataset", dest = "dataset", default = "", help="Dataset path")
parser.add_argument("-l", "--language", dest ="language", help="")
parser.add_argument("-md", "--mode", dest ="mode", help="")
args = parser.parse_args()

model_path = f"models/Speakato_model_{datetime.datetime.now().date()}"
dataset_path = f"examples\polish_commands_dataset"
language = "pl"
mode = "1"

if not args.mode:
    print("Welcome to speakato trainer! Before we start, let's configure some neccessary stuff!")
    mode_input = input(f"What do you want to do? Mode 1: Create model, Mode 2: add new data to previously created model. Default mode: {mode}\nType:")
    if mode_input:
        mode = mode_input
else:
    mode = args.mode

if not args.model:
    new_path = input(f"Where do you want to save your model/where did you save your model? Default path: {model_path}\nType:")
    if new_path:
        model_path = new_path
else:
    model_path = args.model

if not args.dataset:
    new_path = input(f"Where is dataset? Default path: {dataset_path}\nType:")
    if new_path:
        dataset_path = new_path
else:
    dataset_path = args.dataset

if not args.language:
    new_language = input(f"What language you want to use (pl/eng)/did you use in previously trained model? Default language: {language}\nType:")
    if new_language:
        language = new_language
else:
    new_language = args.language

print(f"""
    Config summary:
    Mode: {mode}
    Model path: {model_path}
    Dataset path: {dataset_path}
    Language: {language}
""")

speakato_trainer = SpeakatoTrainer(language, model_path, dataset_path, mode)

print("Starting training...")
speakato_trainer.train()
speakato_trainer.save()
print(speakato_trainer.predict("Hej"))