# Speakato Trainer
Speakato Trainer is a project used to create models conforming to the requirements of [Speakato](https://github.com/Adnovac/Speakato). 

Write down the desired commands, prepare dataset and Speakato will do the rest.

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
Sample datasets have been included in ```examples\```

## How to use?
Prepare dataset, run ```python src\main.py``` and pass necessary arguments. That's all!

Example usage with flags:
```python src/main.py --language pl --dataset .\examples\polish_commands_dataset\ --model .\models\test --mode 1```

Available modes:
- 1 - Create new model
- 2 - Add new data to previously created model

## How to test trained model?
You can use predefined methods from ```src\Sandbox.ipynb``` to test your model. 