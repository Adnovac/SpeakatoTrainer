import datetime
from SpeakatoTrainer import SpeakatoTrainer

model_path = f"../models/Speakato_model_{datetime.datetime.now().date()}"
dataset_path = f"../dataset"
language = "pl"

print("Welcome to speakato trainer! Before we start, let's configure some neccessary stuff!")
new_path = input(f"Where do you want to save your model? Default path: {model_path}")
if new_path:
    model_path = new_path

new_path = input(f"Where is dataset? Default path: {dataset_path}")
if new_path:
    dataset_path = new_path

new_language = input(f"What language you want to use (pl/eng)? Default language: {language}")
if new_language:
    language = new_language

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