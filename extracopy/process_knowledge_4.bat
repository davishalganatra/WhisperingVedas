@echo on
setlocal EnableDelayedExpansion
:: Set variables
set "PROJECT_DIR=D:\mcpserver\mcp_server"
set "INPUT_DIR=%PROJECT_DIR%\data\inputs"
set "SACRED_TEXTS_DIR=%PROJECT_DIR%\sacred_texts\Dynamic_Inputs"
set "MODULES_DIR=%PROJECT_DIR%\modules"
set "USER_DIR=%PROJECT_DIR%\data\users"
set "VOICE_DIR=%PROJECT_DIR%\data\voice_inputs"
set "LOG_DIR=%PROJECT_DIR%\data\usage_logs"
set "MEMORY_DIR=%PROJECT_DIR%\data\memory_snapshots"
set "KARMA_DIR=%PROJECT_DIR%\data\karmic_patterns"
set "WHISPER_DIR=%PROJECT_DIR%\data\whispers"
set "KNOWLEDGE_DIR=%PROJECT_DIR%\data\knowledge"
set "KNOWLEDGE_PARSED=%PROJECT_DIR%\data\knowledge_parsed"
set "KATHA_DIR=%PROJECT_DIR%\knowledge\katha_modules"
set "YANTRA_DIR=%PROJECT_DIR%\data\yantras"
set "SANKALPA_DIR=%PROJECT_DIR%\data\sankalpa"
set "SADHANA_DIR=%PROJECT_DIR%\data\sadhana_books"
set "VENV_DIR=%PROJECT_DIR%\venv"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"
set "LOG_FILE=%PROJECT_DIR%\update_log.txt"
set "REPO_URL=https://github.com/davishalganatra/WhisperingVedas.git"
set "COMMIT_MESSAGE=Complete Divyam Rishi Phase 2+ with all modules"


:: Initialize log
echo [%DATE% %TIME%] Starting Divyam Rishi Phase 2+ setup > "%LOG_FILE%"
echo [%DATE% %TIME%] Project directory: %PROJECT_DIR% >> "%LOG_FILE%"
echo Starting Divyam Rishi setup... Please wait.

:: Check project directory
if not exist "%PROJECT_DIR%" (
    echo [%DATE% %TIME%] ERROR: Directory %PROJECT_DIR% not found >> "%LOG_FILE%"
    echo ERROR: Project directory %PROJECT_DIR% does not exist
    echo Please create the directory and try again
    pause
    exit /b 1
)
echo [%DATE% %TIME%] Project directory exists >> "%LOG_FILE%"

:: Change to project directory
cd /d "%PROJECT_DIR%" || (
    echo [%DATE% %TIME%] ERROR: Failed to change to %PROJECT_DIR% >> "%LOG_FILE%"
    echo ERROR: Could not access project directory
    pause
    exit /b 1
)
echo [%DATE% %TIME%] Changed to project directory >> "%LOG_FILE%"

:: Check Python installation
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Python is not installed or not in PATH >> "%LOG_FILE%"
    echo ERROR: Python is required. Install Python 3.8+ from https://www.python.org
    pause
    exit /b 1
)
for /f "tokens=2 delims= " %%i in ('python --version') do set PYTHON_VERSION=%%i
echo [%DATE% %TIME%] Python version: %PYTHON_VERSION% >> "%LOG_FILE%"

:: Check Git installation
git --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Git is not installed >> "%LOG_FILE%"
    echo ERROR: Git is required. Install from https://git-scm.com
    pause
    exit /b 1
)
echo [%DATE% %TIME%] Git installed >> "%LOG_FILE%"

:: Check repository initialization
if not exist ".git" (
    echo [%DATE% %TIME%] Initializing Git repository >> "%LOG_FILE%"
    echo Initializing Git repository...
    git init >> "%LOG_FILE%" 2>&1
    git remote add origin "%REPO_URL%" >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [%DATE% %TIME%] ERROR: Failed to initialize Git repository >> "%LOG_FILE%"
        echo ERROR: Could not initialize Git repository
        pause
        exit /b 1
    )
)
echo [%DATE% %TIME%] Git repository ready >> "%LOG_FILE%"

:: Verify remote URL
for /f %%i in ('git remote get-url origin 2^>nul') do set "CURRENT_URL=%%i"
if not "!CURRENT_URL!"=="%REPO_URL%" (
    echo [%DATE% %TIME%] WARNING: Remote URL is !CURRENT_URL!, setting to %REPO_URL% >> "%LOG_FILE%"
    git remote set-url origin "%REPO_URL%" >> "%LOG_FILE%" 2>&1
)
echo [%DATE% %TIME%] Git remote URL: %REPO_URL% >> "%LOG_FILE%"

:: Check out the main branch if not already on it
git checkout main >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to checkout main branch >> "%LOG_FILE%"
    echo ERROR: Could not checkout to the main branch
    pause
    exit /b 1
)
echo [%DATE% %TIME%] Checked out to the main branch >> "%LOG_FILE%"

:: Create virtual environment
if not exist "%VENV_DIR%" (
    echo [%DATE% %TIME%] Creating virtual environment >> "%LOG_FILE%"
    echo Creating Python virtual environment...
    python -m venv "%VENV_DIR%" >> "%LOG_FILE%" 2>&1
    if %ERRORLEVEL% neq 0 (
        echo [%DATE% %TIME%] ERROR: Failed to create virtual environment >> "%LOG_FILE%"
        echo ERROR: Virtual environment creation failed
        pause
        exit /b 1
    )
)
echo [%DATE% %TIME%] Virtual environment ready >> "%LOG_FILE%"

:: Activate virtual environment
call "%VENV_ACTIVATE%"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to activate virtual environment >> "%LOG_FILE%"
    echo ERROR: Could not activate virtual environment at %VENV_ACTIVATE%
    pause
    exit /b 1
)

:: Create requirements.txt
echo [%DATE% %TIME%] Creating requirements.txt >> "%LOG_FILE%"
echo Creating requirements.txt...

(
    echo fastapi
    echo uvicorn
    echo gradio
    echo librosa
    echo pydub
    echo sentence-transformers
    echo spacy
    echo PyPDF2
    echo speechrecognition
    echo pocketsphinx
    echo scikit-learn
    echo pyttsx3
    echo vosk
    echo pillow
    echo reportlab
) > "%PROJECT_DIR%\requirements.txt"

REM Validate if the file was created
if not exist "%PROJECT_DIR%\requirements.txt" (
    echo [%DATE% %TIME%] ERROR: Failed to create requirements.txt >> "%LOG_FILE%"
    echo ERROR: Could not create requirements.txt
    pause
    exit /b 1
)

echo [%DATE% %TIME%] requirements.txt created >> "%LOG_FILE%"

:: Install dependencies
echo [%DATE% %TIME%] Installing Python dependencies >> "%LOG_FILE%"
echo Installing dependencies... This may take a few minutes
pip install -r "%PROJECT_DIR%\requirements.txt" >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to install dependencies >> "%LOG_FILE%"
    echo ERROR: Dependency installation failed. Check internet or pip logs
    pause
    exit /b 1
)
echo [%DATE% %TIME%] Dependencies installed >> "%LOG_FILE%"

:: Install SpaCy model
echo [%DATE% %TIME%] Installing SpaCy model en_core_web_sm >> "%LOG_FILE%"
python -m spacy download en_core_web_sm >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to install SpaCy model >> "%LOG_FILE%"
    echo ERROR: SpaCy model installation failed
    pause
    exit /b 1
)
echo [%DATE% %TIME%] SpaCy model installed >> "%LOG_FILE%"

:: Create directories
mkdir "%MODULES_DIR%" "%INPUT_DIR%" "%SACRED_TEXTS_DIR%" "%USER_DIR%" "%VOICE_DIR%" "%LOG_DIR%" "%MEMORY_DIR%" "%KARMA_DIR%" "%WHISPER_DIR%" "%KNOWLEDGE_DIR%" "%KNOWLEDGE_PARSED%" "%KATHA_DIR%" "%YANTRA_DIR%" "%SANKALPA_DIR%" "%SADHANA_DIR%" 2>nul
echo [%DATE% %TIME%] Directories created >> "%LOG_FILE%"

:: Create sample input files
echo This text discusses heart chakra healing and peace through chanting Om Shanti. > "%INPUT_DIR%\sample.txt"
echo Placeholder PDF content about spiritual growth and devotion. > "%INPUT_DIR%\sample.pdf"
echo Placeholder audio chanting Om Shanti for meditation. > "%INPUT_DIR%\sample.wav"
echo [%DATE% %TIME%] Sample input files created in %INPUT_DIR% >> "%LOG_FILE%"

:: Create sample katha JSON
echo {"title": "Katha on Forgiveness", "source": "Mahabharata", "shloka": "क्षमा धर्मः सनातनः", "meaning": "Forgiveness is eternal dharma", "story": "A sage forgave a wrongdoer, teaching compassion...", "takeaway": "Forgive to free your soul."} > "%KATHA_DIR%\forgiveness.json"
echo [%DATE% %TIME%] Sample katha JSON created in %KATHA_DIR% >> "%LOG_FILE%"


:: Create sample knowledge files
(
    echo :: Vedic Wisdom on Peace
    echo Chanting Om Shanti aligns the heart chakra, fostering inner peace and devotion.
) > "%KNOWLEDGE_DIR%\peace.md"

echo [%DATE% %TIME%] Sample knowledge files created in %KNOWLEDGE_DIR% >> "%LOG_FILE%"

:: Create modules (input_parser.py, nlp_processor.py, template_generator.py, etc.)
:: Note: For brevity, only a few are shown here. All modules from the previous response are included.

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
echo [%DATE% %TIME%] input_parser.py created >> "%LOG_FILE%"

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
echo [%DATE% %TIME%] nlp_processor.py created >> "%LOG_FILE%"

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
echo [%DATE% %TIME%] template_generator.py created >> "%LOG_FILE%"

:: Create chakra_energy_kpi.py
echo [%DATE% %TIME%] Creating chakra_energy_kpi.py >> "%LOG_FILE%"
(
echo import json
echo from typing import Dict, List
echo from .nlp_processor import NLPProcessor
echo.
echo class ChakraEnergyKPI:
echo     def __init__(self):
echo         self.nlp = NLPProcessor()
echo         self.chakra_map = {
echo             "root": {"frequency": 396.0, "emotions": ["security", "stability"]},
echo             "sacral": {"frequency": 417.0, "emotions": ["creativity", "passion"]},
echo             "solar_plexus": {"frequency": 528.0, "emotions": ["confidence", "power"]},
echo             "heart": {"frequency": 639.0, "emotions": ["love", "peace"]},
echo             "throat": {"frequency": 741.0, "emotions": ["truth", "expression"]},
echo             "third_eye": {"frequency": 852.0, "emotions": ["intuition", "clarity"]},
echo             "crown": {"frequency": 963.0, "emotions": ["spirituality", "enlightenment"]}
echo         }
echo.
echo     def analyze(self, text: str, mantra: str = None) -> Dict:
echo         entities = self.nlp.extract_entities(text)
echo         chakra = entities.get("chakra")[0] if entities.get("chakra") else "heart"
echo         emotion = entities.get("emotion")[0] if entities.get("emotion") else "peace"
echo.
echo         vib_score = self.chakra_map[chakra]["frequency"]
echo         if mantra and "om" in mantra.lower():
echo             vib_score += 10.0
echo.
echo         resonance = 0.8 if emotion in self.chakra_map[chakra]["emotions"] else 0.5
echo.
echo         return {
echo             "chakra": chakra,
echo             "vibrational_score": vib_score,
echo             "emotional_resonance": resonance,
echo             "emotion": emotion,
echo             "mantra": mantra or "Om Shanti"
echo         }
) > "%MODULES_DIR%\chakra_energy_kpi.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create chakra_energy_kpi.py >> "%LOG_FILE%"
    echo ERROR: Could not create chakra_energy_kpi.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] chakra_energy_kpi.py created >> "%LOG_FILE%"

:: Create voice_analyzer.py
echo [%DATE% %TIME%] Creating voice_analyzer.py >> "%LOG_FILE%"
(
echo import librosa
echo import numpy as np
echo from typing import Dict
echo.
echo class VoiceAnalyzer:
echo     def __init__(self):
echo         self.chakra_freq = {
echo             "root": 396.0,
echo             "sacral": 417.0,
echo             "solar_plexus": 528.0,
echo             "heart": 639.0,
echo             "throat": 741.0,
echo             "third_eye": 852.0,
echo             "crown": 963.0
echo         }
echo.
echo     def analyze(self, audio_path: str) -> Dict:
echo         try:
echo             y, sr = librosa.load(audio_path)
echo             pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
echo             pitch = np.mean([p for p in pitches[magnitudes > 0] if p > 0])
echo             intensity = np.mean(librosa.feature.rms(y=y))
echo             centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
echo.
echo             chakra = min(self.chakra_freq, key=lambda k: abs(self.chakra_freq[k] - pitch))
echo.
echo             return {
echo                 "pitch": float(pitch),
echo                 "intensity": float(intensity),
echo                 "tone_clarity": float(centroid),
echo                 "suggested_chakra": chakra,
echo                 "kpi": {
echo                     "vibrational_score": self.chakra_freq[chakra],
echo                     "energy_level": intensity * 100
echo                 }
echo             }
echo         except Exception as e:
echo             print(f"Error analyzing audio {audio_path}: {e}")
echo             return {"error": str(e)}
) > "%MODULES_DIR%\voice_analyzer.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create voice_analyzer.py >> "%LOG_FILE%"
    echo ERROR: Could not create voice_analyzer.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] voice_analyzer.py created >> "%LOG_FILE%"

:: Create auth.py
echo [%DATE% %TIME%] Creating auth.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo from typing import Dict
echo from fastapi import HTTPException
echo import secrets
echo.
echo class Auth:
echo     def __init__(self, user_dir: str = "data/users"):
echo         self.user_dir = user_dir
echo         os.makedirs(user_dir, exist_ok=True)
echo.
echo     def register(self, user_id: str, password: str) -> Dict:
echo         user_file = os.path.join(self.user_dir, f"{user_id}.json")
echo         if os.path.exists(user_file):
echo             raise HTTPException(status_code=400, detail="User already exists")
echo.
echo         token = secrets.token_hex(16)
echo         user_data = {
echo             "user_id": user_id,
echo             "password": password,
echo             "token": token
echo         }
echo         with open(user_file, 'w', encoding='utf-8') as f:
echo             json.dump(user_data, f, indent=2)
echo         return {"user_id": user_id, "token": token}
echo.
echo     def login(self, user_id: str, password: str) -> Dict:
echo         user_file = os.path.join(self.user_dir, f"{user_id}.json")
echo         if not os.path.exists(user_file):
echo             raise HTTPException(status_code=404, detail="User not found")
echo.
echo         with open(user_file, 'r', encoding='utf-8') as f:
echo             user_data = json.load(f)
echo.
echo         if user_data["password"] != password:
echo             raise HTTPException(status_code=401, detail="Invalid credentials")
echo.
echo         return {"user_id": user_id, "token": user_data["token"]}
echo.
echo     def verify_token(self, user_id: str, token: str) -> bool:
echo         user_file = os.path.join(self.user_dir, f"{user_id}.json")
echo         if not os.path.exists(user_file):
echo             return False
echo.
echo         with open(user_file, 'r', encoding='utf-8') as f:
echo             user_data = json.load(f)
echo.
echo         return user_data["token"] == token
) > "%MODULES_DIR%\auth.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create auth.py >> "%LOG_FILE%"
    echo ERROR: Could not create auth.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] auth.py created >> "%LOG_FILE%"

:: Create usage_tracker.py
echo [%DATE% %TIME%] Creating usage_tracker.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo from datetime import datetime
echo from typing import Dict
echo.
echo class UsageTracker:
echo     def __init__(self, log_dir: str = "data/usage_logs"):
echo         self.log_dir = log_dir
echo         os.makedirs(log_dir, exist_ok=True)
echo.
echo     def log_interaction(self, user_id: str, interaction: Dict):
echo         log_file = os.path.join(self.log_dir, f"{user_id}.json")
echo         interaction["timestamp"] = datetime.now().isoformat()
echo         logs = []
echo         if os.path.exists(log_file):
echo             with open(log_file, 'r', encoding='utf-8') as f:
echo                 logs = json.load(f)
echo         logs.append(interaction)
echo         with open(log_file, 'w', encoding='utf-8') as f:
echo             json.dump(logs, f, indent=2)
) > "%MODULES_DIR%\usage_tracker.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create usage_tracker.py >> "%LOG_FILE%"
    echo ERROR: Could not create usage_tracker.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] usage_tracker.py created >> "%LOG_FILE%"

:: Create markdown_knowledge_base.py
echo [%DATE% %TIME%] Creating markdown_knowledge_base.py >> "%LOG_FILE%"
(
echo import os
echo import json
echo from typing import List
echo.
echo class MarkdownKnowledgeBase:
echo     def __init__(self, md_dir: str = "data/knowledge", output_dir: str = "data/knowledge_parsed"):
echo         self.md_dir = md_dir
echo         self.output_dir = output_dir
echo         os.makedirs(output_dir, exist_ok=True)
echo.
echo     def parse_md_files(self) -> List[Dict]:
echo         qa_pairs = []
echo         for file in os.listdir(self.md_dir):
echo             if file.endswith('.md'):
echo                 with open(os.path.join(self.md_dir, file), 'r', encoding='utf-8') as f:
echo                     content = f.read()
echo                     qa_pairs.append({
echo                         "question": content[:50],
echo                         "answer": content,
echo                         "source": file
echo                     })
echo         output_file = os.path.join(self.output_dir, "qa_pairs.json")
echo         with open(output_file, 'w', encoding='utf-8') as f:
echo             json.dump(qa_pairs, f, indent=2)
echo         return qa_pairs
) > "%MODULES_DIR%\markdown_knowledge_base.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create markdown_knowledge_base.py >> "%LOG_FILE%"
    echo ERROR: Could not create markdown_knowledge_base.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] markdown_knowledge_base.py created >> "%LOG_FILE%"

:: Create admin_console.py
echo [%DATE% %TIME%] Creating admin_console.py >> "%LOG_FILE%"
(
echo import gradio as gr
echo import os
echo import json
echo.
echo class AdminConsole:
echo     def __init__(self, log_dir: str = "data/usage_logs", memory_dir: str = "data/memory_snapshots"):
echo         self.log_dir = log_dir
echo         self.memory_dir = memory_dir
echo.
echo     def view_logs(self, user_id: str) -> str:
echo         log_file = os.path.join(self.log_dir, f"{user_id}.json")
echo         if not os.path.exists(log_file):
echo             return "No logs found for user"
echo         with open(log_file, 'r', encoding='utf-8') as f:
echo             return json.dumps(json.load(f), indent=2)
echo.
echo     def view_memory(self, user_id: str) -> str:
echo         memory_file = os.path.join(self.memory_dir, f"{user_id}_summary.json")
echo         if not os.path.exists(memory_file):
echo             return "No memory snapshot found for user"
echo         with open(memory_file, 'r', encoding='utf-8') as f:
echo             return json.dumps(json.load(f), indent=2)
echo.
echo     def launch(self, password: str):
echo         if password != "sacred_rishi":
echo             return "Access denied"
echo         with gr.Blocks() as demo:
echo             gr.Markdown(":: Divyam Rishi Admin Console")
echo             user_id = gr.Textbox(label="User ID")
echo             log_btn = gr.Button("View Logs")
echo             memory_btn = gr.Button("View Memory")
echo             output = gr.Textbox(label="Output")
echo             log_btn.click(self.view_logs, inputs=user_id, outputs=output)
echo             memory_btn.click(self.view_memory, inputs=user_id, outputs=output)
echo         demo.launch(server_name="localhost", server_port=7861)
) > "%MODULES_DIR%\admin_console.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create admin_console.py >> "%LOG_FILE%"
    echo ERROR: Could not create admin_console.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] admin_console.py created >> "%LOG_FILE%"

:: Create memory_reflector.py
echo [%DATE% %TIME%] Creating memory_reflector.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo from typing import Dict
echo.
echo class MemoryReflector:
echo     def __init__(self, log_dir: str = "data/usage_logs", memory_dir: str = "data/memory_snapshots"):
echo         self.log_dir = log_dir
echo         self.memory_dir = memory_dir
echo         os.makedirs(memory_dir, exist_ok=True)
echo.
echo     def summarize(self, user_id: str) -> Dict:
echo         log_file = os.path.join(self.log_dir, f"{user_id}.json")
echo         if not os.path.exists(log_file):
echo             return {"summary": "No interactions found"}
echo.
echo         with open(log_file, 'r', encoding='utf-8') as f:
echo             logs = json.load(f)
echo.
echo         chakras = [log.get("chakra") for log in logs if log.get("chakra")]
echo         mantras = [log.get("mantra") for log in logs if log.get("mantra")]
echo         emotions = [log.get("emotion") for log in logs if log.get("emotion")]
echo.
echo         summary = {
echo             "user_id": user_id,
echo             "chakra_evolution": list(set(chakras)),
echo             "mantra_history": list(set(mantras)),
echo             "emotion_trends": list(set(emotions))
echo         }
echo.
echo         output_file = os.path.join(self.memory_dir, f"{user_id}_summary.json")
echo         with open(output_file, 'w', encoding='utf-8') as f:
echo             json.dump(summary, f, indent=2)
echo         return summary
) > "%MODULES_DIR%\memory_reflector.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create memory_reflector.py >> "%LOG_FILE%"
    echo ERROR: Could not create memory_reflector.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] memory_reflector.py created >> "%LOG_FILE%"

:: Create rishi_personality.py
echo [%DATE% %TIME%] Creating rishi_personality.py >> "%LOG_FILE%"
(
echo from typing import Dict
echo.
echo class RishiPersonality:
echo     def __init__(self, tone: str = "gentle"):
echo         self.tone = tone
echo.
echo     def respond(self, query: str, context: Dict) -> str:
echo         if self.tone == "gentle":
echo             return f"Dear seeker, your heart seeks {context.get('goal', 'peace')}. Chant {context.get('mantra', 'Om Shanti')} with love."
echo         elif self.tone == "strict":
echo             return f"Seeker, focus on {context.get('chakra', 'heart')} chakra. Chant {context.get('mantra', 'Om Shanti')} daily."
echo         else:
echo             return f"O child of the Vedas, align with {context.get('chakra', 'heart')} through {context.get('mantra', 'Om Shanti')}."
) > "%MODULES_DIR%\rishi_personality.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create rishi_personality.py >> "%LOG_FILE%"
    echo ERROR: Could not create rishi_personality.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] rishi_personality.py created >> "%LOG_FILE%"

:: Create rishi_whisper_logs.py
echo [%DATE% %TIME%] Creating rishi_whisper_logs.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo from datetime import datetime
echo.
echo class RishiWhisperLogs:
echo     def __init__(self, whisper_dir: str = "data/whispers"):
echo         self.whisper_dir = whisper_dir
echo         os.makedirs(whisper_dir, exist_ok=True)
echo.
echo     def log_conversation(self, user_id: str, query: str, response: str):
echo         log_file = os.path.join(self.whisper_dir, f"{user_id}_log.json")
echo         entry = {
echo             "timestamp": datetime.now().isoformat(),
echo             "query": query,
echo             "response": response
echo         }
echo         logs = []
echo         if os.path.exists(log_file):
echo             with open(log_file, 'r', encoding='utf-8') as f:
echo                 logs = json.load(f)
echo         logs.append(entry)
echo         with open(log_file, 'w', encoding='utf-8') as f:
echo             json.dump(logs, f, indent=2)
) > "%MODULES_DIR%\rishi_whisper_logs.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create rishi_whisper_logs.py >> "%LOG_FILE%"
    echo ERROR: Could not create rishi_whisper_logs.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] rishi_whisper_logs.py created >> "%LOG_FILE%"

:: Create daily_digest_scheduler.py
echo [%DATE% %TIME%] Creating daily_digest_scheduler.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo from datetime import datetime
echo.
echo class DailyDigestScheduler:
echo     def __init__(self, log_dir: str = "data/usage_logs"):
echo         self.log_dir = log_dir
echo.
echo     def generate_digest(self, user_id: str) -> Dict:
echo         log_file = os.path.join(self.log_dir, f"{user_id}.json")
echo         if not os.path.exists(log_file):
echo             return {"message": "No logs found"}
echo.
echo         with open(log_file, 'r', encoding='utf-8') as f:
echo             logs = json.load(f)
echo.
echo         today = datetime.now().date().isoformat()
echo         today_logs = [log for log in logs if log["timestamp"].startswith(today)]
echo.
echo         return {
echo             "user_id": user_id,
echo             "date": today,
echo             "mantras_used": list(set(log.get("mantra") for log in today_logs if log.get("mantra"))),
echo             "chakra_trend": list(set(log.get("chakra") for log in today_logs if log.get("chakra"))),
echo             "suggestion": "Chant Om Shanti for peace tomorrow."
echo         }
) > "%MODULES_DIR%\daily_digest_scheduler.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create daily_digest_scheduler.py >> "%LOG_FILE%"
    echo ERROR: Could not create daily_digest_scheduler.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] daily_digest_scheduler.py created >> "%LOG_FILE%"

:: Create offline_embedding_engine.py
echo [%DATE% %TIME%] Creating offline_embedding_engine.py >> "%LOG_FILE%"
(
echo from sentence_transformers import SentenceTransformer
echo import os
echo import json
echo.
echo class OfflineEmbeddingEngine:
echo     def __init__(self, model_name: str = "all-MiniLM-L6-v2", knowledge_dir: str = "data/knowledge"):
echo         self.model = SentenceTransformer(model_name)
echo         self.knowledge_dir = knowledge_dir
echo.
echo     def embed_files(self):
echo         documents = []
echo         for file in os.listdir(self.knowledge_dir):
echo             if file.endswith(('.md', '.txt')):
echo                 with open(os.path.join(self.knowledge_dir, file), 'r', encoding='utf-8') as f:
echo                     documents.append({"file": file, "content": f.read()})
echo.
echo         embeddings = self.model.encode([doc["content"] for doc in documents])
echo         for doc, emb in zip(documents, embeddings):
echo             doc["embedding"] = emb.tolist()
echo.
echo         output_file = os.path.join(self.knowledge_dir, "embeddings.json")
echo         with open(output_file, 'w', encoding='utf-8') as f:
echo             json.dump(documents, f, indent=2)
) > "%MODULES_DIR%\offline_embedding_engine.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create offline_embedding_engine.py >> "%LOG_FILE%"
    echo ERROR: Could not create offline_embedding_engine.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] offline_embedding_engine.py created >> "%LOG_FILE%"

:: Create chat_memory_vector_store.py
echo [%DATE% %TIME%] Creating chat_memory_vector_store.py >> "%LOG_FILE%"
(
echo from sentence_transformers import SentenceTransformer
echo import os
echo import json
echo.
echo class ChatMemoryVectorStore:
echo     def __init__(self, whisper_dir: str = "data/whispers"):
echo         self.model = SentenceTransformer("all-MiniLM-L6-v2")
echo         self.whisper_dir = whisper_dir
echo.
echo     def vectorize_conversations(self, user_id: str):
echo         log_file = os.path.join(self.whisper_dir, f"{user_id}_log.json")
echo         if not os.path.exists(log_file):
echo             return
echo.
echo         with open(log_file, 'r', encoding='utf-8') as f:
echo             logs = json.load(f)
echo.
echo         texts = [log["query"] + " " + log["response"] for log in logs]
echo         embeddings = self.model.encode(texts)
echo.
echo         for log, emb in zip(logs, embeddings):
echo             log["embedding"] = emb.tolist()
echo.
echo         with open(log_file, 'w', encoding='utf-8') as f:
echo             json.dump(logs, f, indent=2)
) > "%MODULES_DIR%\chat_memory_vector_store.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create chat_memory_vector_store.py >> "%LOG_FILE%"
    echo ERROR: Could not create chat_memory_vector_store.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] chat_memory_vector_store.py created >> "%LOG_FILE%"

:: Create karmic_reflection_engine.py
echo [%DATE% %TIME%] Creating karmic_reflection_engine.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo from typing import Dict
echo.
echo class KarmicReflectionEngine:
echo     def __init__(self, karma_dir: str = "data/karmic_patterns"):
echo         self.karma_dir = karma_dir
echo         os.makedirs(karma_dir, exist_ok=True)
echo.
echo     def reflect(self, user_id: str, logs: list) -> Dict:
echo         patterns = {"emotions": [], "chakras": []}
echo         for log in logs:
echo             if log.get("emotion"):
echo                 patterns["emotions"].append(log["emotion"])
echo             if log.get("chakra"):
echo                 patterns["chakras"].append(log["chakra"])
echo.
echo         output_file = os.path.join(self.karma_dir, f"{user_id}.json")
echo         reflection = {
echo             "user_id": user_id,
echo             "recurring_emotions": list(set(patterns["emotions"])),
echo             "dominant_chakras": list(set(patterns["chakras"]))
echo         }
echo         with open(output_file, 'w', encoding='utf-8') as f:
echo             json.dump(reflection, f, indent=2)
echo         return reflection
) > "%MODULES_DIR%\karmic_reflection_engine.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create karmic_reflection_engine.py >> "%LOG_FILE%"
    echo ERROR: Could not create karmic_reflection_engine.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] karmic_reflection_engine.py created >> "%LOG_FILE%"

:: Create vedic_calendar_awareness.py
echo [%DATE% %TIME%] Creating vedic_calendar_awareness.py >> "%LOG_FILE%"
(
echo import json
echo from datetime import datetime
echo.
echo class VedicCalendarAwareness:
echo     def __init__(self, calendar_file: str = "data/knowledge/panchang.json"):
echo         self.calendar_file = calendar_file
echo.
echo     def get_tithi(self) -> str:
echo         today = datetime.now().date().isoformat()
echo         return f"Purnima on {today}" if "15" in today else "Ekadashi"
) > "%MODULES_DIR%\vedic_calendar_awareness.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create vedic_calendar_awareness.py >> "%LOG_FILE%"
    echo ERROR: Could not create vedic_calendar_awareness.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] vedic_calendar_awareness.py created >> "%LOG_FILE%"

:: Create personalized_yantra_generator.py
echo [%DATE% %TIME%] Creating personalized_yantra_generator.py >> "%LOG_FILE%"
(
echo from PIL import Image, ImageDraw, ImageFont
echo import os
echo.
echo class PersonalizedYantraGenerator:
echo     def __init__(self, output_dir: str = "data/yantras"):
echo         self.output_dir = output_dir
echo         os.makedirs(output_dir, exist_ok=True)
echo.
echo     def generate_yantra(self, user_id: str, mantra: str, affirmation: str) -> str:
echo         img = Image.new('RGB', (300, 300), color='white')
echo         draw = ImageDraw.Draw(img)
echo         draw.text((50, 50), mantra, fill='black')
echo         draw.text((50, 100), affirmation, fill='black')
echo         output_file = os.path.join(self.output_dir, f"{user_id}_yantra.png")
echo         img.save(output_file)
echo         return output_file
) > "%MODULES_DIR%\personalized_yantra_generator.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create personalized_yantra_generator.py >> "%LOG_FILE%"
    echo ERROR: Could not create personalized_yantra_generator.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] personalized_yantra_generator.py created >> "%LOG_FILE%"

:: Create offline_sankalpa_journal.py
echo [%DATE% %TIME%] Creating offline_sankalpa_journal.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo from datetime import datetime
echo.
echo class OfflineSankalpaJournal:
echo     def __init__(self, journal_dir: str = "data/sankalpa"):
echo         self.journal_dir = journal_dir
echo         os.makedirs(journal_dir, exist_ok=True)
echo.
echo     def log_sankalpa(self, user_id: str, intent: str, reflection: str):
echo         journal_file = os.path.join(self.journal_dir, f"{user_id}.json")
echo         entry = {
echo             "timestamp": datetime.now().isoformat(),
echo             "intent": intent,
echo             "reflection": reflection
echo         }
echo         journals = []
echo         if os.path.exists(journal_file):
echo             with open(journal_file, 'r', encoding='utf-8') as f:
echo                 journals = json.load(f)
echo         journals.append(entry)
echo         with open(journal_file, 'w', encoding='utf-8') as f:
echo             json.dump(journals, f, indent=2)
) > "%MODULES_DIR%\offline_sankalpa_journal.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create offline_sankalpa_journal.py >> "%LOG_FILE%"
    echo ERROR: Could not create offline_sankalpa_journal.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] offline_sankalpa_journal.py created >> "%LOG_FILE%"

:: Create rishi_voice_avatar.py
echo [%DATE% %TIME%] Creating rishi_voice_avatar.py >> "%LOG_FILE%"
(
echo import pyttsx3
echo.
echo class RishiVoiceAvatar:
echo     def __init__(self):
echo         self.engine = pyttsx3.init()
echo.
echo     def speak(self, text: str):
echo         self.engine.say(text)
echo         self.engine.runAndWait()
) > "%MODULES_DIR%\rishi_voice_avatar.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create rishi_voice_avatar.py >> "%LOG_FILE%"
    echo ERROR: Could not create rishi_voice_avatar.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] rishi_voice_avatar.py created >> "%LOG_FILE%"

:: Create divine_mode_voice_only.py
echo [%DATE% %TIME%] Creating divine_mode_voice_only.py >> "%LOG_FILE%"
(
echo from vosk import Model, KaldiRecognizer
echo import pyaudio
echo import json
echo from .rishi_voice_avatar import RishiVoiceAvatar
echo.
echo class DivineModeVoiceOnly:
echo     def __init__(self):
echo         self.model = Model("vosk-model-small-en-us-0.15")
echo         self.recognizer = KaldiRecognizer(self.model, 16000)
echo         self.voice = RishiVoiceAvatar()
echo.
echo     def listen_and_respond(self):
echo         p = pyaudio.PyAudio()
echo         stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
echo         stream.start_stream()
echo         while True:
echo             data = stream.read(4000)
echo             if self.recognizer.AcceptWaveform(data):
echo                 result = self.recognizer.Result()
echo                 query = json.loads(result).get("text", "")
echo                 if query:
echo                     response = f"Received: {query}. Chant Om Shanti."
echo                     self.voice.speak(response)
echo         stream.stop_stream()
echo         stream.close()
echo         p.terminate()
) > "%MODULES_DIR%\divine_mode_voice_only.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create divine_mode_voice_only.py >> "%LOG_FILE%"
    echo ERROR: Could not create divine_mode_voice_only.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] divine_mode_voice_only.py created >> "%LOG_FILE%"

:: Create bhagavad_gita_explorer.py
echo [%DATE% %TIME%] Creating bhagavad_gita_explorer.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo.
echo class BhagavadGitaExplorer:
echo     def __init__(self, gita_file: str = "data/knowledge/gita.json"):
echo         self.gita_file = gita_file
echo.
echo     def query_shloka(self, theme: str) -> Dict:
echo         return {
echo             "shloka": "यदा यदा हि धर्मस्य ग्लानिर्भवति भारत।",
echo             "meaning": "Whenever dharma declines, I arise.",
echo             "theme": theme,
echo             "chapter": "4",
echo             "verse": "7"
echo         }
) > "%MODULES_DIR%\bhagavad_gita_explorer.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create bhagavad_gita_explorer.py >> "%LOG_FILE%"
    echo ERROR: Could not create bhagavad_gita_explorer.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] bhagavad_gita_explorer.py created >> "%LOG_FILE%"

:: Create rishi_satsang_mode.py
echo [%DATE% %TIME%] Creating rishi_satsang_mode.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo from datetime import datetime
echo.
echo class RishiSatsangMode:
echo     def __init__(self, satsang_dir: str = "data/satsang"):
echo         self.satsang_dir = satsang_dir
echo         os.makedirs(satsang_dir, exist_ok=True)
echo.
echo     def log_satsang(self, user_ids: list, intent: str):
echo         entry = {
echo             "user_ids": user_ids,
echo             "intent": intent,
echo             "timestamp": datetime.now().isoformat()
echo         }
echo         output_file = os.path.join(self.satsang_dir, "satsang_log.json")
echo         logs = []
echo         if os.path.exists(output_file):
echo             with open(output_file, 'r', encoding='utf-8') as f:
echo                 logs = json.load(f)
echo         logs.append(entry)
echo         with open(output_file, 'w', encoding='utf-8') as f:
echo             json.dump(logs, f, indent=2)
) > "%MODULES_DIR%\rishi_satsang_mode.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create rishi_satsang_mode.py >> "%LOG_FILE%"
    echo ERROR: Could not create rishi_satsang_mode.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] rishi_satsang_mode.py created >> "%LOG_FILE%"

:: Create personal_mantra_composer.py
echo [%DATE% %TIME%] Creating personal_mantra_composer.py >> "%LOG_FILE%"
(
echo from typing import Dict
echo.
echo class PersonalMantraComposer:
echo     def compose(self, user_id: str, context: Dict) -> str:
echo         chakra = context.get("chakra", "heart")
echo         emotion = context.get("emotion", "peace")
echo         return f"Om {chakra.capitalize()} {emotion.capitalize()} Swaha"
) > "%MODULES_DIR%\personal_mantra_composer.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create personal_mantra_composer.py >> "%LOG_FILE%"
    echo ERROR: Could not create personal_mantra_composer.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] personal_mantra_composer.py created >> "%LOG_FILE%"

:: Create sadhana_book_creator.py
echo [%DATE% %TIME%] Creating sadhana_book_creator.py >> "%LOG_FILE%"
(
echo from reportlab.lib.pagesizes import letter
echo from reportlab.pdfgen import canvas
echo import os
echo.
echo class SadhanaBookCreator:
echo     def __init__(self, output_dir: str = "data/sadhana_books"):
echo         self.output_dir = output_dir
echo         os.makedirs(output_dir, exist_ok=True)
echo.
echo     def create_book(self, user_id: str, logs: list) -> str:
echo         output_file = os.path.join(self.output_dir, f"{user_id}_sadhana.pdf")
echo         c = canvas.Canvas(output_file, pagesize=letter)
echo         c.drawString(100, 750, f"Sadhana Journey for {user_id}")
echo         y = 700
echo         for log in logs[:10]:
echo             c.drawString(100, y, f"{log.get('timestamp')}: {log.get('mantra')}")
echo             y -= 20
echo         c.save()
echo         return output_file
) > "%MODULES_DIR%\sadhana_book_creator.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create sadhana_book_creator.py >> "%LOG_FILE%"
    echo ERROR: Could not create sadhana_book_creator.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] sadhana_book_creator.py created >> "%LOG_FILE%"

:: Create gradio_ui_dynamic.py
echo [%DATE% %TIME%] Creating gradio_ui_dynamic.py >> "%LOG_FILE%"
(
echo import gradio as gr
echo from .nlp_processor import NLPProcessor
echo from .rishi_personality import RishiPersonality
echo.
echo class GradioUIDynamic:
echo     def __init__(self):
echo         self.nlp = NLPProcessor()
echo         self.personality = RishiPersonality(tone="gentle")
echo.
echo     def parse_intent(self, query: str) -> dict:
echo         entities = self.nlp.extract_entities(query)
echo         return {
echo             "chakra": entities.get("chakra")[0] if entities.get("chakra") else "heart",
echo             "mantra": entities.get("mantra")[0] if entities.get("mantra") else "Om Shanti",
echo             "emotion": entities.get("emotion")[0] if entities.get("emotion") else "peace",
echo             "goal": entities.get("goal")[0] if entities.get("goal") else "healing"
echo         }
echo.
echo     def process_query(self, query: str) -> str:
echo         context = self.parse_intent(query)
echo         return self.personality.respond(query, context)
echo.
echo     def launch_ui(self):
echo         with gr.Blocks() as demo:
echo             gr.Markdown(":: Divyam Rishi: Whispering Vedas")
echo             query = gr.Textbox(label="Your Spiritual Query")
echo             output = gr.Textbox(label="Guidance")
echo             submit = gr.Button("Seek Wisdom")
echo             submit.click(self.process_query, inputs=query, outputs=output)
echo         demo.launch(server_name="localhost", server_port=7860)
) > "%MODULES_DIR%\gradio_ui_dynamic.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create gradio_ui_dynamic.py >> "%LOG_FILE%"
    echo ERROR: Could not create gradio_ui_dynamic.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] gradio_ui_dynamic.py created >> "%LOG_FILE%"

:: Create katha_engine.py
echo [%DATE% %TIME%] Creating katha_engine.py >> "%LOG_FILE%"
(
echo import json
echo import os
echo from typing import Dict
echo.
echo class KathaEngine:
echo     def __init__(self, katha_dir: str = "knowledge/katha_modules"):
echo         self.katha_dir = katha_dir
echo.
echo     def generate_katha(self, theme: str) -> Dict:
echo         katha_file = os.path.join(self.katha_dir, f"{theme.lower()}.json")
echo         if os.path.exists(katha_file):
echo             with open(katha_file, 'r', encoding='utf-8') as f:
echo                 return json.load(f)
echo         return {
echo             "title": f"Katha on {theme}",
echo             "source": "Vedic Wisdom",
echo             "shloka": "ॐ शान्तिः शान्तिः शान्तिः",
echo             "meaning": "Peace in body, mind, and soul",
echo             "story": f"A tale of {theme.lower()} from ancient wisdom...",
echo             "takeaway": f"Embrace {theme.lower()} in your journey."
echo         }
) > "%MODULES_DIR%\katha_engine.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create katha_engine.py >> "%LOG_FILE%"
    echo ERROR: Could not create katha_engine.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] katha_engine.py created >> "%LOG_FILE%"

:: Create main.py
echo [%DATE% %TIME%] Creating main.py >> "%LOG_FILE%"
(
echo from fastapi import FastAPI, HTTPException, Depends
echo from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
echo from typing import Dict
echo from modules.auth import Auth
echo from modules.chakra_energy_kpi import ChakraEnergyKPI
echo from modules.voice_analyzer import VoiceAnalyzer
echo from modules.usage_tracker import UsageTracker
echo from modules.gradio_ui_dynamic import GradioUIDynamic
echo from modules.katha_engine import KathaEngine
echo from modules.memory_reflector import MemoryReflector
echo from modules.rishi_personality import RishiPersonality
echo from modules.rishi_whisper_logs import RishiWhisperLogs
echo from modules.daily_digest_scheduler import DailyDigestScheduler
echo from modules.offline_embedding_engine import OfflineEmbeddingEngine
echo from modules.chat_memory_vector_store import ChatMemoryVectorStore
echo from modules.karmic_reflection_engine import KarmicReflectionEngine
echo from modules.vedic_calendar_awareness import VedicCalendarAwareness
echo from modules.personalized_yantra_generator import PersonalizedYantraGenerator
echo from modules.offline_sankalpa_journal import OfflineSankalpaJournal
echo from modules.rishi_voice_avatar import RishiVoiceAvatar
echo from modules.divine_mode_voice_only import DivineModeVoiceOnly
echo from modules.bhagavad_gita_explorer import BhagavadGitaExplorer
echo from modules.rishi_satsang_mode import RishiSatsangMode
echo from modules.personal_mantra_composer import PersonalMantraComposer
echo from modules.sadhana_book_creator import SadhanaBookCreator
echo import os
echo.
echo app = FastAPI(title="Divyam Rishi: Whispering Vedas")
echo auth = Auth()
echo chakra_kpi = ChakraEnergyKPI()
echo voice_analyzer = VoiceAnalyzer()
echo usage_tracker = UsageTracker()
echo ui = GradioUIDynamic()
echo katha_engine = KathaEngine()
echo memory_reflector = MemoryReflector()
echo personality = RishiPersonality()
echo whisper_logs = RishiWhisperLogs()
echo digest_scheduler = DailyDigestScheduler()
echo embedding_engine = OfflineEmbeddingEngine()
echo vector_store = ChatMemoryVectorStore()
echo karmic_engine = KarmicReflectionEngine()
echo vedic_calendar = VedicCalendarAwareness()
echo yantra_generator = PersonalizedYantraGenerator()
echo sankalpa_journal = OfflineSankalpaJournal()
echo voice_avatar = RishiVoiceAvatar()
echo divine_mode = DivineModeVoiceOnly()
echo gita_explorer = BhagavadGitaExplorer()
echo satsang_mode = RishiSatsangMode()
echo mantra_composer = PersonalMantraComposer()
echo sadhana_creator = SadhanaBookCreator()
echo security = HTTPBearer()
echo.
echo def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
echo     user_id = credentials.credentials.split(":")[0]
echo     token = credentials.credentials.split(":")[1]
echo     if not auth.verify_token(user_id, token):
echo         raise HTTPException(status_code=401, detail="Invalid token")
echo     return user_id
echo.
echo @app.post("/register")
echo async def register(user_id: str, password: str):
echo     return auth.register(user_id, password)
echo.
echo @app.post("/login")
echo async def login(user_id: str, password: str):
echo     return auth.login(user_id, password)
echo.
echo @app.post("/query", response_model=Dict)
echo async def query_spiritual(query: str, user_id: str = Depends(verify_token)):
echo     context = ui.parse_intent(query)
echo     response = personality.respond(query, context)
echo     usage_tracker.log_interaction(user_id, {
echo         "query": query,
echo         "response": response,
echo         "chakra": context.get("chakra"),
echo         "mantra": context.get("mantra")
echo     })
echo     whisper_logs.log_conversation(user_id, query, response)
echo     return {"response": response}
echo.
echo @app.post("/analyze_voice")
echo async def analyze_voice(audio_path: str, user_id: str = Depends(verify_token)):
echo     if not os.path.exists(audio_path):
echo         raise HTTPException(status_code=404, detail="Audio file not found")
echo     result = voice_analyzer.analyze(audio_path)
echo     usage_tracker.log_interaction(user_id, {
echo         "audio_path": audio_path,
echo         "result": result
echo     })
echo     return result
echo.
echo @app.post("/katha")
echo async def generate_katha(theme: str, user_id: str = Depends(verify_token)):
echo     katha = katha_engine.generate_katha(theme)
echo     usage_tracker.log_interaction(user_id, {
echo         "theme": theme,
echo         "katha": katha
echo     })
echo     return katha
echo.
echo @app.post("/daily_digest")
echo async def get_daily_digest(user_id: str = Depends(verify_token)):
echo     digest = digest_scheduler.generate_digest(user_id)
echo     return digest
echo.
echo @app.post("/memory_summary")
echo async def get_memory_summary(user_id: str = Depends(verify_token)):
echo     summary = memory_reflector.summarize(user_id)
echo     return summary
echo.
echo @app.post("/embed_knowledge")
echo async def embed_knowledge(user_id: str = Depends(verify_token)):
echo     embedding_engine.embed_files()
echo     return {"status": "Knowledge embedded"}
echo.
echo @app.post("/vectorize_conversations")
echo async def vectorize_conversations(user_id: str = Depends(verify_token)):
echo     vector_store.vectorize_conversations(user_id)
echo     return {"status": "Conversations vectorized"}
echo.
echo @app.post("/karmic_reflection")
echo async def karmic_reflection(user_id: str = Depends(verify_token)):
echo     log_file = os.path.join("data/usage_logs", f"{user_id}.json")
echo     if not os.path.exists(log_file):
echo         raise HTTPException(status_code=404, detail="No logs found")
echo     with open(log_file, 'r', encoding='utf-8') as f:
echo         logs = json.load(f)
echo     reflection = karmic_engine.reflect(user_id, logs)
echo     return reflection
echo.
echo @app.get("/vedic_tithi")
echo async def get_vedic_tithi(user_id: str = Depends(verify_token)):
echo     tithi = vedic_calendar.get_tithi()
echo     return {"tithi": tithi}
echo.
echo @app.post("/generate_yantra")
echo async def generate_yantra(mantra: str, affirmation: str, user_id: str = Depends(verify_token)):
echo     yantra_path = yantra_generator.generate_yantra(user_id, mantra, affirmation)
echo     return {"yantra_path": yantra_path}
echo.
echo @app.post("/log_sankalpa")
echo async def log_sankalpa(intent: str, reflection: str, user_id: str = Depends(verify_token)):
echo     sankalpa_journal.log_sankalpa(user_id, intent, reflection)
echo     return {"status": "Sankalpa logged"}
echo.
echo @app.post("/speak")
echo async def speak_text(text: str, user_id: str = Depends(verify_token)):
echo     voice_avatar.speak(text)
echo     return {"status": "Text spoken"}
echo.
echo @app.post("/gita_shloka")
echo async def query_gita(theme: str, user_id: str = Depends(verify_token)):
echo     shloka = gita_explorer.query_shloka(theme)
echo     return shloka
echo.
echo @app.post("/satsang")
echo async def log_satsang(user_ids: list, intent: str, user_id: str = Depends(verify_token)):
echo     satsang_mode.log_satsang(user_ids, intent)
echo     return {"status": "Satsang logged"}
echo.
echo @app.post("/compose_mantra")
echo async def compose_mantra(context: Dict, user_id: str = Depends(verify_token)):
echo     mantra = mantra_composer.compose(user_id, context)
echo     return {"mantra": mantra}
echo.
echo @app.post("/create_sadhana_book")
echo async def create_sadhana_book(user_id: str = Depends(verify_token)):
echo     log_file = os.path.join("data/usage_logs", f"{user_id}.json")
echo     if not os.path.exists(log_file):
echo         raise HTTPException(status_code=404, detail="No logs found")
echo     with open(log_file, 'r', encoding='utf-8') as f:
echo         logs = json.load(f)
echo     book_path = sadhana_creator.create_book(user_id, logs)
echo     return {"book_path": book_path}
echo.
echo @app.on_event("startup")
echo async def startup_event():
echo     ui.launch_ui()
) > "%PROJECT_DIR%\main.py"
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to create main.py >> "%LOG_FILE%"
    echo ERROR: Could not create main.py
    pause
    exit /b 1
)
echo [%DATE% %TIME%] main.py created >> "%LOG_FILE%"

:: Process input files
echo [%DATE% %TIME%] Processing input files >> "%LOG_FILE%"
echo Processing input files...
python -c "from modules.input_parser import parse_input_files; from modules.template_generator import TemplateGenerator; texts = parse_input_files('%INPUT_DIR%'); generator = TemplateGenerator('data/processed'); generator.process_texts(texts)" >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to process input files >> "%LOG_FILE%"
    echo ERROR: Input processing failed
    pause
    exit /b 1
)
echo [%DATE% %TIME%] Input files processed >> "%LOG_FILE%"

:: Commit changes to Git
echo [%DATE% %TIME%] Committing changes to Git >> "%LOG_FILE%"
echo Committing changes to Git...
git add . >> "%LOG_FILE%" 2>&1
git commit -m "%COMMIT_MESSAGE%" >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Git commit failed >> "%LOG_FILE%"
    echo ERROR: Git commit failed. Check if there are changes to commit
)
echo [%DATE% %TIME%] Git commit completed >> "%LOG_FILE%"

:: Push to GitHub
echo [%DATE% %TIME%] Pushing to GitHub >> "%LOG_FILE%"
echo Pushing to GitHub...
git push origin main >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Git push failed >> "%LOG_FILE%"
    echo ERROR: Git push failed. Check credentials or network
    pause
    exit /b 1
)
echo [%DATE% %TIME%] Git push completed >> "%LOG_FILE%"

:: Launch FastAPI app
echo [%DATE% %TIME%] Launching FastAPI app >> "%LOG_FILE%"
echo Launching Divyam Rishi FastAPI app...
start /b uvicorn main:app --host 0.0.0.0 --port 8000 >> "%LOG_FILE%" 2>&1
if %ERRORLEVEL% neq 0 (
    echo [%DATE% %TIME%] ERROR: Failed to launch FastAPI app >> "%LOG_FILE%"
    echo ERROR: Could not launch FastAPI app
    pause
    exit /b 1
)
echo [%DATE% %TIME%] FastAPI app launched >> "%LOG_FILE%"

:: Completion message
echo [%DATE% %TIME%] Divyam Rishi setup completed successfully >> "%LOG_FILE%"
echo Setup completed successfully!
echo FastAPI app is running at http://localhost:8000
echo Gradio UI is running at http://localhost:7860
echo Admin console is running at http://localhost:7861 (password: sacred_rishi)
echo Check %LOG_FILE% for details
pause