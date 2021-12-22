import datetime
import argparse
from SpeakatoTrainer import SpeakatoTrainer

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", dest = "model", default = "", help="Model path")
parser.add_argument("-d", "--dataset", dest = "dataset", default = "", help="Dataset path")
parser.add_argument("-l", "--language", dest ="language", help="")
args = parser.parse_args()

model_path = f"../models/Speakato_model_{datetime.datetime.now().date()}"
dataset_path = f"../dataset"
language = "pl"

if not args.model:
    print("Welcome to speakato trainer! Before we start, let's configure some neccessary stuff!")
    new_path = input(f"Where do you want to save your model? Default path: {model_path}")
    if new_path:
        model_path = new_path
else:
    model_path = args.model

if not args.model:
    new_path = input(f"Where is dataset? Default path: {dataset_path}")
    if new_path:
        dataset_path = new_path
else:
    dataset_path = args.dataset

if not args.language:
    new_language = input(f"What language you want to use (pl/eng)? Default language: {language}")
    if new_language:
        language = new_language
else:
    new_language = args.language

print(f"""
    Config summary:
    Model path: {model_path}
    Dataset path: {dataset_path}
    Language: {language}
""")

speakato_trainer = SpeakatoTrainer(language, model_path, dataset_path)

print("Starting training...")
speakato_trainer.train()
speakato_trainer.save()