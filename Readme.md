# Speakato Trainer
## Prerequisites
Before you start don't forget to install necessary packages:
```powershell
pip install -r requirements.txt
python -m spacy download pl_core_news_sm
python -m spacy download en_core_web_sm
```

## How to build training dataset?
Put the list of commands along with it's labels in the ```dataset.json```. Data should be saved in the following form:
```json
[
    {
        "text": "some text",
        "command": "example_command"
    }
]
```
Save command list in the ```commands.txt``` file:
```
greeting
open
close
alarm
```
An example dataset has been placed in ```examples\polish_commands_dataset```

## How to use?
Prepare dataset, run ```python src\main.py``` and pass necessary arguments. That's all!

## How to test trained model?
You can use predefined methods from ```src\Sandbox.ipynb``` to test your model. 