import os.path as path
import spacy

class SpeakatoTrainer:
    global speakato_lemma
    global model_path
    global dataset_path
    global lemmatizer

    def __init__(language: str, model: str, dataset: str):
        """
        Constructs a new SpeakatoTrainer
        :param language: Dataset language
        :param model: Path in which trained model will be saved 
        :param dataset: Dataset path
        """
        global speakato_lemma
        global model_path
        global dataset_path
        global lemmatizer

        model_path = model
        if(path.exists(model_path)):
            raise Exception(f"Model: {model_path} already exists!")

        dataset_path = dataset
        if(not path.exists(dataset_path)):
            raise Exception(f"Dataset: {dataset_path} doesn't exists!")

        language = language.lower()

        if(language not in ["pl", "eng"]):
            raise Exception("Language not available. Supported languages: PL/ENG")

        if(language == "eng"):
            lemmatizer = spacy.load("en_core_web_sm")
        elif(language == "pl"):
            lemmatizer = spacy.load("pl_core_news_sm")

