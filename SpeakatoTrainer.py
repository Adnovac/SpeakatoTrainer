from SpeakatoLemmatizer import SpeakatoLemmatizer
import os.path as path

class SpeakatoTrainer:
    global speakato_lemma
    global model_path
    global dataset_path

    def __init__(language: str, model: str, dataset: str):
        """
        Constructs a new SpeakatoTrainer
        :param language: Language used in lemmatizer
        :param model: Path in which trained model will be saved 
        :param dataset: Dataset path
        """
        global speakato_lemma
        global model_path
        global dataset_path

        model_path = model
        dataset_path = dataset

        language = language.lower()
        if(path.exists(model_path)):
            raise Exception("There's al")
        if(language not in ["pl", "eng"]):
            raise Exception("Language not available. Supported languages: PL/ENG")
        
        speakato_lemma = SpeakatoLemmatizer(language)

    def train(self):
