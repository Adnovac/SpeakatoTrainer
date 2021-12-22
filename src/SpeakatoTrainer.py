from sklearn.model_selection import train_test_split
import os
import spacy
import pandas as pd
import numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense

class SpeakatoTrainer:
    global nlp
    global model_path
    global model
    global labels
    global X_train
    global X_test
    global y_train
    global y_test
    global model_info

    def __init__(self, language: str, model: str, dataset: str):
        """
        Constructs a new SpeakatoTrainer
        :param lang: Supported language
        :param model: Path in which trained model will be saved
        :param dataset: Dataset path
        """
        global nlp
        global model_path
        global dataset_path
        global lang
        global model_info

        self.check_initial_data(language,model,dataset)

        dataset_path = dataset
        model_path = model

        os.makedirs(model_path)

        if(language == "eng"):
            lemmatizer = "en_core_web_sm"
        elif(language == "pl"):
            lemmatizer = "pl_core_news_sm"

        model_info = {
            "language": language,
            "lemmatizer": lemmatizer,
        }
        nlp = spacy.load(lemmatizer)
        self.load_dataset(dataset_path)
        

    def check_initial_data(self, language: str, model_path: str, dataset_path: str):
        if(not os.path.exists(dataset_path)):
            raise Exception(f"Dataset: {dataset_path} doesn't exists!")

        if(os.path.exists(model_path)):
            raise Exception(f"Model: {model_path} already exists!")

        if(language not in ["pl", "eng"]):
            raise Exception("Language not available. Supported languages: pl/eng")

    def load_dataset(self, dataset_path: str):
        global labels
        global X_train
        global X_test
        global y_train
        global y_test

        with open(f"{dataset_path}/commands.txt", "r") as f:
            labels = sorted([x.rstrip("\n") for x in f.readlines()])
        
        data = pd.read_json(f"{dataset_path}/dataset.json")
        X = np.array([nlp(x).vector for x in data["text"]])
        y = pd.get_dummies(data[["command"]]).values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        model_info["token_len"] = len(X_train[0])


    
    def train(self):
        global model
        model = Sequential()
        model.add(Dense(500, activation='relu', input_dim=len(X_train[0])))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(len(labels), activation='softmax'))

        model.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

        model.fit(X_train, y_train, epochs=20)
        print("\nTrain data:")
        results  = model.evaluate(X_train, y_train)
        model_info["train_evaluation"] = results

        print("\nTest data:")
        results  = model.evaluate(X_test, y_test)
        model_info["test_evaluation"] = results


    def save(self):
        global model_info
        model.save(f"{model_path}/model")
        with open(f"{model_path}/commands.txt", "w+") as f:
            f.writelines([label + "\n" for label in labels])

        formatted_config = json.dumps(model_info, indent=4)
        with open(f"{model_path}/info.json", "w+") as f:
            f.write(formatted_config)