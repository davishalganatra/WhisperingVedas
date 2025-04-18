@echo off
setlocal EnableDelayedExpansion

:: Set variables
set "PROJECT_DIR=D:\mcp server\mcp_server"
set "INPUT_DIR=%PROJECT_DIR%\data\inputs"
set "SACRED_TEXTS_DIR=%PROJECT_DIR%\sacred_texts\Dynamic_Inputs"
set "MODULES_DIR=%PROJECT_DIR%\modules"
set "VENV_DIR=%PROJECT_DIR%\venv"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"
set "MONGO_DIR=C:\Program Files\MongoDB\Server\7.0"
set "MONGO_DATA=%PROJECT_DIR%\mongodb\data"
set "MONGO_LOG=%PROJECT_DIR%\mongodb\log\mongod.log"
set "LOG_FILE=%PROJECT_DIR%\update_log.txt"
set "REPO_URL=https://github.com/davishalganatra/WhisperingVedas.git"
set "COMMIT_MESSAGE=Enhance merge_json.py and add dynamic template system for knowledge base"

:: Initialize log
echo [%DATE% %TIME%] Starting knowledge processing > "%LOG_FILE%"

:: Check project directory
if not exist "%PROJECT_DIR%" (
    echo [%DATE% %TIME%] ERROR: Directory %PROJECT_DIR% not found >> "%LOG_FILE%"
    echo ERROR: Project directory not found
    pause
    exit /b 1
)

:: Change to project directory
cd /d "%PROJECT_DIR%" || (
    echo [%DATE% %TIME%] ERROR: Failed to change to %PROJECT_DIR% >> "%LOG_FILE%"
    echo ERROR: Could not access project directory
    pause
    exit /b 1
)

:: Check Git installation
git --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Git is not installed >> "%LOG_FILE%"
    echo ERROR: Git is not installed. Install from https://git-scm.com
    pause
    exit /b 1
)

:: Check repository initialization
if not exist ".git" (
    echo [%DATE% %TIME%] ERROR: Git repository not initialized >> "%LOG_FILE%"
    echo ERROR: Git repository not found. Run 'git init' and set remote
    pause
    exit /b 1
)

:: Verify remote URL
git remote get-url origin >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] WARNING: No remote set. Setting to %REPO_URL% >> "%LOG_FILE%"
    git remote add origin "%REPO_URL%"
) else (
    for /f %%i in ('git remote get-url origin') do set "CURRENT_URL=%%i"
    if not "!CURRENT_URL!"=="%REPO_URL%" (
        echo [%DATE% %TIME%] WARNING: Remote URL is !CURRENT_URL!, expected %REPO_URL% >> "%LOG_FILE%"
        git remote set-url origin "%REPO_URL%"
    )
)

:: Install MongoDB
if not exist "%MONGO_DIR%" (
    echo [%DATE% %TIME%] Installing MongoDB >> "%LOG_FILE%"
    winget install MongoDB.Server --version 7.0.14 >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [%DATE% %TIME%] ERROR: Failed to install MongoDB >> "%LOG_FILE%"
        echo ERROR: MongoDB installation failed. Download from https://www.mongodb.com
        pause
        exit /b 1
    )
)

:: Create MongoDB directories and start server
mkdir "%MONGO_DATA%" "%PROJECT_DIR%\mongodb\log" 2>nul
echo [%DATE% %TIME%] Starting MongoDB server >> "%LOG_FILE%"
start /b "" "%MONGO_DIR%\bin\mongod.exe" --dbpath "%MONGO_DATA%" --logpath "%MONGO_LOG%" --logappend
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to start MongoDB >> "%LOG_FILE%"
    echo ERROR: MongoDB server failed to start. Check port 27017
    pause
    exit /b 1
)

:: Create virtual environment
if not exist "%VENV_DIR%" (
    echo [%DATE% %TIME%] Creating virtual environment >> "%LOG_FILE%"
    python -m venv "%VENV_DIR%" >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [%DATE% %TIME%] ERROR: Failed to create virtual environment >> "%LOG_FILE%"
        echo ERROR: Virtual environment creation failed
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call "%VENV_ACTIVATE%"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to activate virtual environment >> "%LOG_FILE%"
    echo ERROR: Could not activate virtual environment
    pause
    exit /b 1
)

:: Install dependencies
echo [%DATE% %TIME%] Installing dependencies >> "%LOG_FILE%"
(
echo PyPDF2
echo speechrecognition
echo spacy
echo scikit-learn
echo pymongo
echo pocketsphinx
) > requirements.txt
pip install -r requirements.txt >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to install dependencies >> "%LOG_FILE%"
    echo ERROR: Dependency installation failed
    pause
    exit /b 1
)

:: Install SpaCy model
python -m spacy download en_core_web_sm >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to install SpaCy model >> "%LOG_FILE%"
    echo ERROR: SpaCy model installation failed
    pause
    exit /b 1
)

:: Create modules directory
mkdir "%MODULES_DIR%" 2>nul

:: Create input_parser.py
echo [%DATE% %TIME%] Creating input_parser.py >> "%LOG_FILE%"
(
echo import os
echo import PyPDF2
echo import speech_recognition as sr
echo from pathlib import Path
echo.
echo def parse_text_file(file_path: str) -> str:
echo     try:
echo         with open(file_path, 'r', encoding='utf-8') as f:
echo             return f.read()
echo     except Exception as e:
echo         print(f"Error parsing text file {file_path}: {e}")
echo         return ""
echo.
echo def parse_pdf_file(file_path: str) -> str:
echo     try:
echo         with open(file_path, 'rb') as f:
echo             reader = PyPDF2.PdfReader(f)
echo             text = ""
echo             for page in reader.pages:
echo                 text += page.extract_text() or ""
echo             return text
echo     except Exception as e:
echo         print(f"Error parsing PDF {file_path}: {e}")
echo         return ""
echo.
echo def parse_audio_file(file_path: str) -> str:
echo     recognizer = sr.Recognizer()
echo     try:
echo         with sr.AudioFile(file_path) as source:
echo             audio = recognizer.record(source)
echo             text = recognizer.recognize_sphinx(audio)
echo             return text
echo     except Exception as e:
echo         print(f"Error transcribing audio {file_path}: {e}")
echo         return ""
echo.
echo def parse_input_files(input_dir: str) -> list:
echo     texts = []
echo     input_dir = Path(input_dir)
echo     for file_path in input_dir.glob("*"):
echo         if file_path.suffix.lower() in ('.txt', '.pdf', '.wav'):
echo             if file_path.suffix == '.txt':
echo                 text = parse_text_file(file_path)
echo             elif file_path.suffix == '.pdf':
echo                 text = parse_pdf_file(file_path)
echo             else:
echo                 text = parse_audio_file(file_path)
echo             if text:
echo                 texts.append({"file": file_path.name, "text": text, "source_type": file_path.suffix[1:]})
echo     return texts
) > "%MODULES_DIR%\input_parser.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create input_parser.py >> "%LOG_FILE%"
    echo ERROR: Could not create input_parser.py
    pause
    exit /b 1
)

:: Create nlp_processor.py
echo [%DATE% %TIME%] Creating nlp_processor.py >> "%LOG_FILE%"
(
echo import spacy
echo from sklearn.feature_extraction.text import TfidfVectorizer
echo from sklearn.naive_bayes import MultinomialNB
echo import numpy as np
echo from typing import Dict, List
echo.
echo class NLPProcessor:
echo     def __init__(self):
echo         self.nlp = spacy.load("en_core_web_sm")
echo         self.vectorizer = TfidfVectorizer()
echo         self.classifier = MultinomialNB()
echo         self.training_data = []
echo         self.training_labels = []
echo.
echo     def extract_entities(self, text: str) -> Dict[str, List[str]]:
echo         doc = self.nlp(text.lower())
echo         entities = {
echo             "chakra": [],
echo             "mantra": [],
echo             "emotion": [],
echo             "goal": []
echo         }
echo         chakra_keywords = {
echo             "root": ["root", "muladhara"],
echo             "sacral": ["sacral", "svadhisthana"],
echo             "solar_plexus": ["solar plexus", "manipura"],
echo             "heart": ["heart", "anahata"],
echo             "throat": ["throat", "vishuddha"],
echo             "third_eye": ["third eye", "ajna"],
echo             "crown": ["crown", "sahasrara"]
echo         }
echo         emotion_keywords = ["peace", "devotion", "courage", "detachment", "focus"]
echo         goal_keywords = ["clarity", "healing", "spiritual growth", "self-realization"]
echo.
echo         for token in doc:
echo             word = token.text
echo             for chakra, keywords in chakra_keywords.items():
echo                 if any(kw in word for kw in keywords):
echo                     entities["chakra"].append(chakra)
echo             if "om" in word or "namah" in word:
echo                 entities["mantra"].append(word)
echo             if word in emotion_keywords:
echo                 entities["emotion"].append(word)
echo             if word in goal_keywords:
echo                 entities["goal"].append(word)
echo.
echo         return entities
echo.
echo     def train_classifier(self, texts: List[str], labels: List[str]):
echo         if not texts or not labels:
echo             return
echo         X = self.vectorizer.fit_transform(texts)
echo         y = np.array(labels)
echo         self.classifier.fit(X, y)
echo         self.training_data.extend(texts)
echo         self.training_labels.extend(labels)
echo.
echo     def predict_field(self, text: str, field: str) -> str:
echo         if not self.training_data:
echo             return ""
echo         X = self.vectorizer.transform([text])
echo         return self.classifier.predict(X)[0]
) > "%MODULES_DIR%\nlp_processor.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create nlp_processor.py >> "%LOG_FILE%"
    echo ERROR: Could not create nlp_processor.py
    pause
    exit /b 1
)

:: Create template_generator.py
echo [%DATE% %TIME%] Creating template_generator.py >> "%LOG_FILE%"
(
echo import uuid
echo import json
echo import os
echo from typing import Dict, List
echo from .nlp_processor import NLPProcessor
echo.
echo class TemplateGenerator:
echo     def __init__(self, output_dir: str):
echo         self.nlp = NLPProcessor()
echo         self.output_dir = output_dir
echo         self.default_values = {
echo             "chakra": "heart",
echo             "mantra": "Om Shanti",
echo             "frequency": 528.0,
echo             "tone": "F",
echo             "repetitions": 54,
echo             "emotion": ["peace"],
echo             "goal": ["healing"]
echo         }
echo.
echo     def generate_template(self, text: str, source: str, file_name: str) -> Dict:
echo         entities = self.nlp.extract_entities(text)
echo         template = {
echo             "id": str(uuid.uuid4()),
echo             "source": source,
echo             "text": text[:500],
echo             "translation": text[:500],
echo             "emotion": entities.get("emotion") or self.default_values["emotion"],
echo             "goal": entities.get("goal") or self.default_values["goal"],
echo             "chakra": entities.get("chakra")[0] if entities.get("chakra") else self.default_values["chakra"],
echo             "mantra": entities.get("mantra")[0] if entities.get("mantra") else self.default_values["mantra"],
echo             "frequency": self.default_values["frequency"],
echo             "tone": self.default_values["tone"],
echo             "repetitions": self.default_values["repetitions"],
echo             "context": file_name
echo         }
echo         self.nlp.train_classifier([text], [template["chakra"]])
echo         return template
echo.
echo     def process_texts(self, texts: List[Dict]):
echo         os.makedirs(self.output_dir, exist_ok=True)
echo         for item in texts:
echo             template = self.generate_template(item["text"], f"{item['source_type'].upper()}_Source", item["file"])
echo             output_file = os.path.join(self.output_dir, f"{item['file'].split('.')[0]}.json")
echo             with open(output_file, 'w', encoding='utf-8') as f:
echo                 json.dump({"verses": [template]}, f, ensure_ascii=False, indent=2)
) > "%MODULES_DIR%\template_generator.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create template_generator.py >> "%LOG_FILE%"
    echo ERROR: Could not create template_generator.py
    pause
    exit /b 1
)

:: Create db_storage.py
echo [%DATE% %TIME%] Creating db_storage.py >> "%LOG_FILE%"
(
echo from pymongo import MongoClient
echo from typing import List, Dict
echo.
echo class KnowledgeDB:
echo     def __init__(self, db_name: str = "whispering_vedas", collection_name: str = "knowledge_base"):
echo         self.client = MongoClient("mongodb://localhost:27017/")
echo         self.db = self.client[db_name]
echo         self.collection = self.db[collection_name]
echo.
echo     def store_templates(self, templates: List[Dict]):
echo         try:
echo             if templates:
echo                 self.collection.insert_many(templates)
echo                 print(f"Stored {len(templates)} templates in MongoDB")
echo         except Exception as e:
echo             print(f"Error storing templates: {e}")
echo.
echo     def close(self):
echo         self.client.close()
) > "%MODULES_DIR%\db_storage.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create db_storage.py >> "%LOG_FILE%"
    echo ERROR: Could not create db_storage.py
    pause
    exit /b 1
)

:: Create merge_json.py
echo [%DATE% %TIME%] Creating merge_json.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo from pathlib import Path
echo from modules.nlp_processor import NLPProcessor
echo.
echo def load_json_files(directory: str) -> list:
echo     knowledge_base = []
echo     nlp = NLPProcessor()
echo     for root, _, files in os.walk(directory):
echo         for file in files:
echo             if file.endswith('.json'):
echo                 file_path = os.path.join(root, file)
echo                 try:
echo                     with open(file_path, 'r', encoding='utf-8') as f:
echo                         data = json.load(f)
echo                         source_name = os.path.basename(root).replace('_', ' ').title()
echo                         for key in ['verses', 'zodiac', 'numbers', 'records']:
echo                             if key in data:
echo                                 for entry in data[key]:
echo                                     entry['source'] = source_name if key == 'verses' else key.capitalize()
echo                                     if not entry.get('emotion') or not entry.get('goal'):
echo                                         entities = nlp.extract_entities(entry.get('text', ''))
echo                                         entry['emotion'] = entry.get('emotion') or entities.get('emotion', ['peace'])
echo                                         entry['goal'] = entry.get('goal') or entities.get('goal', ['healing'])
echo                                     entry['chakra'] = entry.get('chakra') or nlp.predict_field(entry.get('text', ''), 'chakra') or 'heart'
echo                                     entry['mantra'] = entry.get('mantra') or 'Om Shanti'
echo                                     entry['frequency'] = entry.get('frequency') or 528.0
echo                                     entry['tone'] = entry.get('tone') or 'F'
echo                                     entry['repetitions'] = entry.get('repetitions') or 54
echo                                     knowledge_base.append(entry)
echo                 except Exception as e:
echo                     print(f"Error reading {file_path}: {e}")
echo     return knowledge_base
echo.
echo def validate_entry(entry: dict) -> bool:
echo     required_fields = ['id', 'source', 'text', 'translation', 'emotion', 'goal', 'chakra', 'mantra', 'frequency', 'tone', 'repetitions']
echo     for field in required_fields:
echo         if field not in entry or entry[field] is None or (isinstance(entry[field], list) and not entry[field]):
echo             print(f"Validation error: Missing field '{field}' in {entry.get('id', 'unknown')}")
echo             return False
echo     return True
echo.
echo def main():
echo     sacred_texts_dir = 'D:\\sacred_texts'
echo     output_dir = 'data\\vedic_knowledge'
echo     output_file = os.path.join(output_dir, 'vedic_knowledge.json')
echo     os.makedirs(output_dir, exist_ok=True)
echo     knowledge_base = load_json_files(sacred_texts_dir)
echo     valid_entries = [entry for entry in knowledge_base if validate_entry(entry)]
echo     if not valid_entries:
echo         print("No valid entries found.")
echo         return
echo     with open(output_file, 'w', encoding='utf-8') as f:
echo         json.dump(valid_entries, f, ensure_ascii=False, indent=2)
echo     print(f"Saved {len(valid_entries)} entries to {output_file}")
echo.
echo if __name__ == '__main__':
echo     main()
) > merge_json.py
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create merge_json.py >> "%LOG_FILE%"
    echo ERROR: Could not create merge_json.py
    pause
    exit /b 1
)

:: Create input directory and sample files
mkdir "%INPUT_DIR%" "%SACRED_TEXTS_DIR%" 2>nul
echo This text discusses heart chakra healing and peace through chanting Om Shanti. > "%INPUT_DIR%\sample.txt"
:: Note: PDF and WAV are placeholders; replace with real files
echo Placeholder PDF content about spiritual growth. > "%INPUT_DIR%\sample.pdf"
echo Placeholder audio chanting Om Shanti. > "%INPUT_DIR%\sample.wav"

:: Create main_processor.py
echo [%DATE% %TIME%] Creating main_processor.py >> "%LOG_FILE%"
(
echo from modules.input_parser import parse_input_files
echo from modules.template_generator import TemplateGenerator
echo from modules.db_storage import KnowledgeDB
echo from merge_json import load_json_files
echo.
echo texts = parse_input_files("%INPUT_DIR%")
echo generator = TemplateGenerator("%SACRED_TEXTS_DIR%")
echo generator.process_texts(texts)
echo templates = load_json_files("D:\\sacred_texts")
echo db = KnowledgeDB()
echo db.store_templates(templates)
echo db.close()
) > main_processor.py
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create main_processor.py >> "%LOG_FILE%"
    echo ERROR: Could not create main_processor.py
    pause
    exit /b 1
)

:: Run main_processor.py
echo [%DATE% %TIME%] Running main_processor.py >> "%LOG_FILE%"
python main_processor.py >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to process knowledge >> "%LOG_FILE%"
    echo ERROR: Knowledge processing failed
    pause
    exit /b 1
)

:: Run merge_json.py
echo [%DATE% %TIME%] Running merge_json.py >> "%LOG_FILE%"
python merge_json.py >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to run merge_json.py >> "%LOG_FILE%"
    echo ERROR: Could not generate vedic_knowledge.json
    pause
    exit /b 1
)

:: Git operations
echo [%DATE% %TIME%] Starting Git operations >> "%LOG_FILE%"
git pull origin main >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] WARNING: Git pull failed, proceeding with local changes >> "%LOG_FILE%"
    echo WARNING: Could not pull from origin
)
git add modules\*.py merge_json.py requirements.txt main_processor.py data\inputs\ sacred_texts\Dynamic_Inputs\ data\vedic_knowledge\ >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to add files to Git >> "%LOG_FILE%"
    echo ERROR: Could not add files to Git
    pause
    exit /b 1
)
git commit -m "%COMMIT_MESSAGE%" >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] WARNING: Nothing to commit >> "%LOG_FILE%"
    echo WARNING: No changes to commit
)
git push origin main >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to push to %REPO_URL% >> "%LOG_FILE%"
    echo ERROR: Could not push to GitHub. Check credentials or network
    pause
    exit /b 1
)

:: Success message
echo [%DATE% %TIME%] Successfully processed knowledge and updated GitHub >> "%LOG_FILE%"
echo SUCCESS: Knowledge processed and pushed to %REPO_URL%
echo Check logs at %LOG_FILE% for details
pause
exit /b 0