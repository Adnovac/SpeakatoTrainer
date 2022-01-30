from pyexpat import model
from sklearn.model_selection import train_test_split
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
import numpy as np
import json
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import re
import onnx
import tf2onnx

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

    def __init__(self, language: str, model: str, dataset: str, mode: str):
        """
        Constructs a new SpeakatoTrainer
        :param language: Supported language
        :param model: Path in which trained model will be saved
        :param dataset: Dataset path
        """
        global nlp
        global model_path
        global dataset_path
        global lang
        global selected_mode
        global model_info
        global language_selected

        self.check_initial_data(language,model,dataset,mode)

        dataset_path = dataset
        model_path = model
        language_selected = language
        selected_mode = mode

        if(selected_mode == "1"):
            os.makedirs(model_path)

        if(language == "eng"):
            spacy_model = "en_core_web_sm"
        elif(language == "pl"):
            spacy_model = "pl_core_news_sm"

        model_info = {
            "language": language,
            "spacy_model": spacy_model,
        }
        nlp = spacy.load(spacy_model)
        self.load_dataset(dataset_path)
        

    def check_initial_data(self, language: str, model_path: str, dataset_path: str, selected_mode : str):
        if(not os.path.exists(dataset_path)):
            raise Exception(f"Dataset: {dataset_path} doesn't exists!")

        if(os.path.exists(model_path) and selected_mode == "1"):
            raise Exception(f"Model: {model_path} already exists!")

        if(language not in ["pl", "eng"]):
            raise Exception("Language not available! Supported languages: pl/eng")

        if(selected_mode not in ["1", "2"]):
            raise Exception("Wrong mode selected! Supported modes: Mode 1: Create model, Mode 2: add new data to previously created model")


    def load_dataset(self, dataset_path: str):
        global labels
        global X_train
        global X_test
        global y_train
        global y_test
        global X
        global y

        with open(f"{dataset_path}/commands.txt", "r") as f:
            labels = sorted([x.rstrip("\n") for x in f.readlines()])
        data = pd.read_json(f"{dataset_path}/dataset.json")
        data["text"] = self.clean_data(data["text"])
        X = np.array([nlp(x).vector for x in data["text"]])
        y = pd.get_dummies(data[["command"]]).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
        model_info["token_len"] = len(X_train[0])


    def clean_data(self, texts:list):
        """
        Cleans text array by using lemmas, stopwords and lowering text
        :param texts: list of sentences in form of strings.
        """
        final_texts = list()

        for text in texts:
            text = text.lower()
            text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
            words = [w.lemma_ for w in nlp(text) if not w.lemma_ in nlp.Defaults.stop_words]
            words = ' '.join(words)

            final_texts.append(words)


        return final_texts
    

    def building_model(self):
        model = Sequential()
        model.add(Dense(500, activation='relu', input_dim=len(X_train[0])))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(len(labels), activation='softmax'))

        model.compile(optimizer='adam', 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])
        
        return model

    def load_model(self):
        model = load_model(f"{model_path}/model")

        return model

    def train(self):
        global model
        if(selected_mode == "1"):
            model = self.building_model()
        elif(selected_mode == "2"):
            model = self.load_model()

        model.fit(X_train,y_train, epochs=30, batch_size=4, validation_data=(X_test, y_test))
        model.fit(X, y, epochs=3, batch_size=4, validation_data=(X, y))

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

        onnx_model, _ = tf2onnx.convert.from_keras(model)
        onnx.save_model(onnx_model, f"{model_path}/model.onnx")

    def predict(self,X ):
        Predict = np.array([nlp(x).vector for x in [X]])
        y_predicted = model(Predict)
        recognized_index = y_predicted[0].numpy().tolist().index(max(y_predicted[0]))

        return recognized_index