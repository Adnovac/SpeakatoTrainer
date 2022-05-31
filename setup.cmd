python -m venv speakato
speakato\Scripts\pip.exe install --upgrade pip
speakato\Scripts\pip.exe install -r requirements.txt
speakato\Scripts\python.exe -m spacy download pl_core_news_md
speakato\Scripts\python.exe -m spacy download en_core_web_md
set /p DUMMY=Hit ENTER to continue...