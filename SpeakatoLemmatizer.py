import spacy

class SpeakatoLemmatizer:
    global lemmatizer

    def __init__(language: str):
        global lemmatizer
        language = language.lower()
        if(language not in ["pl", "eng"]):
            raise Exception("Language not available. Supported languages: PL/ENG")

        if(language == "eng"):
            lemmatier = spacy.load("en_core_web_sm")
        elif(language == "pl"):
            lemmatier = spacy.load("pl_core_news_sm")