```batch
@echo off
ECHO Setting up MCP Server for Whispering Vedas...

:: Set paths
SET "PROJECT_DIR=D:\mcp server\mcp_server"
SET "TEXTS_DIR=D:\sacred_texts"
SET "PYTHON=python"

:: Navigate to project directory
cd /d "%PROJECT_DIR%"
IF ERRORLEVEL 1 (
    ECHO Failed to navigate to %PROJECT_DIR%. Ensure it exists.
    pause
    exit /b 1
)

:: Check if virtual environment exists
IF NOT EXIST "venv" (
    ECHO Creating virtual environment...
    %PYTHON% -m venv venv
    IF ERRORLEVEL 1 (
        ECHO Failed to create virtual environment. Ensure Python 3.10+ is installed.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call venv\Scripts\activate
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Install dependencies
ECHO Installing dependencies...
pip install fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.2 httpx==0.27.2 apscheduler==3.10.4 SpeechRecognition==3.10.4 gradio==4.44.0
IF ERRORLEVEL 1 (
    ECHO Failed to install dependencies. Retrying...
    pip install --force-reinstall fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.2 httpx==0.27.2 apscheduler==3.10.4 SpeechRecognition==3.10.4 gradio==4.44.0
    IF ERRORLEVEL 1 (
        ECHO Failed to install dependencies after retry.
        pause
        exit /b 1
    )
)

:: Install PyAudio for voice feedback
ECHO Installing PyAudio...
pip install PyAudio
IF ERRORLEVEL 1 (
    ECHO Failed to install PyAudio. Voice feedback may not work.
)

:: Create directories if they don't exist
ECHO Creating directories...
IF NOT EXIST "data\vedic_knowledge" mkdir data\vedic_knowledge
IF NOT EXIST "data\user_logs" mkdir data\user_logs
IF NOT EXIST "data\chants" mkdir data\chants
IF NOT EXIST "data\user_profiles" mkdir data\user_profiles
IF NOT EXIST "modules" mkdir modules

:: Create merge_json.py
ECHO Creating merge_json.py...
(
echo import json
echo import os
echo from pathlib import Path
echo.
echo def merge_json_files(texts_dir, output_file):
echo     texts = [
echo         {
echo             "name": "Ramcharitmanas",
echo             "files": [
echo                 {"file": "Ramcharitmanas\\1_बाल_काण्ड_data.json", "section": "Bal Kānd"},
echo                 {"file": "Ramcharitmanas\\2_अयोध्या_काण्ड_data.json", "section": "Ayodhya Kānd"},
echo                 {"file": "Ramcharitmanas\\3_अरण्य_काण्ड_data.json", "section": "Aranya Kānd"},
echo                 {"file": "Ramcharitmanas\\4_किष्किंधा_काण्ड_data.json", "section": "Kishkindha Kānd"},
echo                 {"file": "Ramcharitmanas\\5_सुंदर_काण्ड_data.json", "section": "Sundar Kānd"},
echo                 {"file": "Ramcharitmanas\\6_लंका_काण्ड_data.json", "section": "Lanka Kānd"},
echo                 {"file": "Ramcharitmanas\\7_उत्तर_काण्ड_data.json", "section": "Uttar Kānd"}
echo             ]
echo         },
echo         {
echo             "name": "Srimad Bhagavad Gita",
echo             "files": [
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_1.json", "section": "Chapter 1"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_2.json", "section": "Chapter 2"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_3.json", "section": "Chapter 3"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_4.json", "section": "Chapter 4"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_5.json", "section": "Chapter 5"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_6.json", "section": "Chapter 6"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_7.json", "section": "Chapter 7"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_8.json", "section": "Chapter 8"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_9.json", "section": "Chapter 9"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_10.json", "section": "Chapter 10"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_11.json", "section": "Chapter 11"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_12.json", "section": "Chapter 12"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_13.json", "section": "Chapter 13"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_14.json", "section": "Chapter 14"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_15.json", "section": "Chapter 15"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_16.json", "section": "Chapter 16"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_17.json", "section": "Chapter 17"},
echo                 {"file": "SrimadBhagvadGita\\bhagavad_gita_chapter_18.json", "section": "Chapter 18"}
echo             ]
echo         },
echo         {
echo             "name": "Mahabharata",
echo             "files": []
echo         },
echo         {
echo             "name": "Valmiki Ramayana",
echo             "files": [
echo                 {"file": "ValmikiRamayana\\1_balakanda.json", "section": "Balakanda"},
echo                 {"file": "ValmikiRamayana\\2_ayodhyakanda.json", "section": "Ayodhyakanda"},
echo                 {"file": "ValmikiRamayana\\3_aranyakanda.json", "section": "Aranyakanda"},
echo                 {"file": "ValmikiRamayana\\4_kishkindhakanda.json", "section": "Kishkindhakanda"},
echo                 {"file": "ValmikiRamayana\\5_sundarakanda.json", "section": "Sundarakanda"},
echo                 {"file": "ValmikiRamayana\\6_yudhhakanda.json", "section": "Yudhhakanda"},
echo                 {"file": "ValmikiRamayana\\7_uttarakanda.json", "section": "Uttarakanda"}
echo             ]
echo         },
echo         {
echo             "name": "Rigveda",
echo             "files": [
echo                 {"file": "Rigveda\\rigveda_mandala_1.json", "section": "Mandala 1"},
echo                 {"file": "Rigveda\\rigveda_mandala_2.json", "section": "Mandala 2"},
echo                 {"file": "Rigveda\\rigveda_mandala_3.json", "section": "Mandala 3"},
echo                 {"file": "Rigveda\\rigveda_mandala_4.json", "section": "Mandala 4"},
echo                 {"file": "Rigveda\\rigveda_mandala_5.json", "section": "Mandala 5"},
echo                 {"file": "Rigveda\\rigveda_mandala_6.json", "section": "Mandala 6"},
echo                 {"file": "Rigveda\\rigveda_mandala_7.json", "section": "Mandala 7"},
echo                 {"file": "Rigveda\\rigveda_mandala_8.json", "section": "Mandala 8"},
echo                 {"file": "Rigveda\\rigveda_mandala_9.json", "section": "Mandala 9"},
echo                 {"file": "Rigveda\\rigveda_mandala_10.json", "section": "Mandala 10"}
echo             ]
echo         },
echo         {
echo             "name": "Yajurveda Shukla",
echo             "files": [
echo                 {"file": "Yajurveda\\vajasneyi_madhyadina_samhita.json", "section": "Vajasaneyi Madhyandina Samhita"},
echo                 {"file": "Yajurveda\\vajasneyi_kanva_samhita_chapters.json", "section": "Vajasaneyi Kanva Samhita"}
echo             ]
echo         },
echo         {
echo             "name": "Atharvaveda",
echo             "files": []
echo         }
echo     ]
echo     knowledge_base = {"texts": []}
echo     total_verses = 0
echo     print(f"Checking files in {texts_dir}...")
echo     for text in texts:
echo         text_entry = {"name": text["name"], "sections": []}
echo         print(f"\nProcessing {text['name']}...")
echo         for file_info in text["files"]:
echo             file_path = os.path.join(texts_dir, file_info["file"])
echo             print(f"Checking {file_path}...")
echo             if not os.path.exists(file_path):
echo                 print(f"  Skipped: File not found")
echo                 continue
echo             try:
echo                 with open(file_path, "r", encoding="utf-8") as f:
echo                     data = json.load(f)
echo                 verses = []
echo                 if isinstance(data, list):
echo                     print(f"  Found list of {len(data)} items")
echo                     for i, item in enumerate(data):
echo                         verse = {
echo                             "id": f"{text['name']}_{file_info['section']}_{i}",
echo                             "text": item.get("text") or item.get("verse") or item.get("content") or "",
echo                             "translation": item.get("translation") or item.get("meaning") or item.get("english") or "",
echo                             "emotion": item.get("emotion", []),
echo                             "goal": item.get("goal", []),
echo                             "chakra": item.get("chakra", None),
echo                             "mantra": item.get("mantra", None),
echo                             "frequency": item.get("frequency", 432.0),
echo                             "tone": item.get("tone", "G"),
echo                             "repetitions": item.get("repetitions", 108)
echo                         }
echo                         verses.append(verse)
echo                 elif isinstance(data, dict):
echo                     print(f"  Found dictionary")
echo                     if "verses" in data or "shlokas" in data or "hymns" in data:
echo                         verse_list = data.get("verses") or data.get("shlokas") or data.get("hymns")
echo                         print(f"    Contains {len(verse_list)} verses/shlokas/hymns")
echo                         for i, item in enumerate(verse_list):
echo                             verse = {
echo                                 "id": f"{text['name']}_{file_info['section']}_{i}",
echo                                 "text": item.get("text") or item.get("verse") or item.get("content") or "",
echo                                 "translation": item.get("translation") or item.get("meaning") or item.get("english") or "",
echo                                 "emotion": item.get("emotion", []),
echo                                 "goal": item.get("goal", []),
echo                                 "chakra": item.get("chakra", None),
echo                                 "mantra": item.get("mantra", None),
echo                                 "frequency": item.get("frequency", 432.0),
echo                                 "tone": item.get("tone", "G"),
echo                                 "repetitions": item.get("repetitions", 108)
echo                             }
echo                             verses.append(verse)
echo                     else:
echo                         print(f"    Scanning dictionary keys")
echo                         for key, value in data.items():
echo                             if isinstance(value, dict):
echo                                 verse = {
echo                                     "id": f"{text['name']}_{file_info['section']}_{key}",
echo                                     "text": value.get("text") or value.get("verse") or value.get("content") or "",
echo                                     "translation": value.get("translation") or value.get("meaning") or value.get("english") or "",
echo                                     "emotion": value.get("emotion", []),
echo                                     "goal": value.get("goal", []),
echo                                     "chakra": value.get("chakra", None),
echo                                     "mantra": value.get("mantra", None),
echo                                     "frequency": value.get("frequency", 432.0),
echo                                     "tone": value.get("tone", "G"),
echo                                     "repetitions": value.get("repetitions", 108)
echo                                 }
echo                                 verses.append(verse)
echo                 if verses:
echo                     print(f"  Added {len(verses)} verses")
echo                     text_entry["sections"].append({"name": file_info["section"], "verses": verses})
echo                     total_verses += len(verses)
echo                 else:
echo                     print(f"  No verses found in {file_path}")
echo             except Exception as e:
echo                 print(f"  Error processing {file_path}: {str(e)}")
echo                 continue
echo         if text_entry["sections"]:
echo             knowledge_base["texts"].append(text_entry)
echo             print(f"Added {text['name']} with {len(text_entry['sections'])} sections")
echo     if not knowledge_base["texts"]:
echo         print("No texts processed. Creating minimal output.")
echo         knowledge_base = {
echo             "texts": [
echo                 {
echo                     "name": "Default",
echo                     "sections": [
echo                         {
echo                             "name": "Default",
echo                             "verses": [
echo                                 {
echo                                     "id": "default_0",
echo                                     "text": "Om Namah Shivaya",
echo                                     "translation": "Universal chant",
echo                                     "emotion": [],
echo                                     "goal": [],
echo                                     "chakra": null,
echo                                     "mantra": "Om Namah Shivaya",
echo                                     "frequency": 432.0,
echo                                     "tone": "G",
echo                                     "repetitions": 108
echo                                 }
echo                             ]
echo                         }
echo                     ]
echo                 }
echo             ]
echo         }
echo         total_verses = 1
echo     os.makedirs(os.path.dirname(output_file), exist_ok=True)
echo     try:
echo         with open(output_file, "w", encoding="utf-8") as f:
echo             json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
echo         print(f"\nSuccess: Created {output_file} with {total_verses} verses")
echo     except Exception as e:
echo         print(f"\nFailed to write {output_file}: {str(e)}")
echo.
echo if __name__ == "__main__":
echo     texts_dir = r"%TEXTS_DIR%"
echo     output_file = r"%PROJECT_DIR%\data\vedic_knowledge\vedic_knowledge.json"
echo     merge_json_files(texts_dir, output_file)
) > merge_json.py
IF ERRORLEVEL 1 (
    ECHO Failed to create merge_json.py.
    pause
    exit /b 1
)

:: Run merge script
ECHO Combining JSON files into vedic_knowledge.json...
%PYTHON% merge_json.py
IF ERRORLEVEL 1 (
    ECHO Failed to combine JSON files. Check merge_json.py output above.
    pause
    exit /b 1
)

:: Pause to review merge output
ECHO Press any key to continue to server setup...
pause

:: Create default user profile
ECHO Creating default user profile...
(
echo import json
echo from modules.user_profiles import create_user_profile
echo.
echo create_user_profile(
echo     user_id="user123",
echo     name="Default User",
echo     goals=["clarity", "peace"],
echo     chakra_focus="heart",
echo     healing_preferences=["mantra", "meditation"],
echo     preferred_tone="G"
echo )
) > create_default_profile.py
%PYTHON% create_default_profile.py
IF ERRORLEVEL 1 (
    ECHO Failed to create default user profile.
    pause
    exit /b 1
)
del create_default_profile.py

:: Start the server
ECHO Starting MCP Server...
%PYTHON% main.py
IF ERRORLEVEL 1 (
    ECHO Failed to start server. Check main.py or dependencies.
    pause
    exit /b 1
)

pause