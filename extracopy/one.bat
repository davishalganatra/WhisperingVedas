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
pip install fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.2 httpx==0.27.2 apscheduler==3.10.4 SpeechRecognition==3.10.4
IF ERRORLEVEL 1 (
    ECHO Failed to install dependencies.
    pause
    exit /b 1
)

:: Create directories
mkdir data\vedic_knowledge data\user_logs data\chants modules
IF ERRORLEVEL 1 (
    ECHO Failed to create directories.
    pause
    exit /b 1
)

:: Create Python script to combine JSON files
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
echo     for text in texts:
echo         text_entry = {"name": text["name"], "sections": []}
echo         for file_info in text["files"]:
echo             file_path = os.path.join(texts_dir, file_info["file"])
echo             if os.path.exists(file_path):
echo                 try:
echo                     with open(file_path, "r", encoding="utf-8") as f:
echo                         data = json.load(f)
echo                     verses = []
echo                     # Standardize verses (assuming varied JSON structures)
echo                     if isinstance(data, list):
echo                         verses = [{"id": f"{text['name']}_{file_info['section']}_{i}", "text": item.get("text", ""), "translation": item.get("translation", ""), "emotion": [], "goal": [], "chakra": null, "mantra": null, "frequency": 432.0, "tone": "G", "repetitions": 108} for i, item in enumerate(data)]
echo                     elif isinstance(data, dict):
echo                         for key, value in data.items():
echo                             if isinstance(value, list):
echo                                 verses.extend([{"id": f"{text['name']}_{file_info['section']}_{i}", "text": v.get("text", ""), "translation": v.get("translation", ""), "emotion": [], "goal": [], "chakra": null, "mantra": null, "frequency": 432.0, "tone": "G", "repetitions": 108} for i, v in enumerate(value)])
echo                             elif isinstance(value, dict):
echo                                 verses.append({"id": f"{text['name']}_{file_info['section']}_{key}", "text": value.get("text", ""), "translation": value.get("translation", ""), "emotion": [], "goal": [], "chakra": null, "mantra": null, "frequency": 432.0, "tone": "G", "repetitions": 108})
echo                     text_entry["sections"].append({"name": file_info["section"], "verses": verses})
echo                 except Exception as e:
echo                     print(f"Error processing {file_path}: {e}")
echo         if text_entry["sections"]:
echo             knowledge_base["texts"].append(text_entry)
echo     os.makedirs(os.path.dirname(output_file), exist_ok=True)
echo     with open(output_file, "w", encoding="utf-8") as f:
echo         json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
echo     print(f"Created {output_file} with {sum(len(s['verses']) for t in knowledge_base['texts'] for s in t['sections'])} verses")
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
    ECHO Failed to combine JSON files. Check merge_json.py or source files in %TEXTS_DIR%.
    pause
    exit /b 1
)

:: Create main.py
ECHO Creating main.py...
(
echo from fastapi import FastAPI, HTTPException, Security, Depends, Request
echo from fastapi.security import APIKeyHeader
echo from fastapi.middleware.cors import CORSMiddleware
echo from fastapi.responses import StreamingResponse
echo from pydantic import BaseModel
echo from typing import Optional, List
echo from starlette.status import HTTP_403_FORBIDDEN
echo import os
echo import time
echo import json
echo import re
echo from modules.mantra_suggestion import suggest_mantra
echo from modules.chant_generation import generate_chant
echo from modules.event_logging import log_event
echo from modules.tap_sadhana import start_tap_sadhana
echo from modules.voice_feedback import process_voice_feedback
echo.
echo app = FastAPI(title="MCP Server - Whispering Vedas", version="1.0.0")
echo.
echo # CORS middleware
echo app.add_middleware(
echo     CORSMiddleware,
echo     allow_origins=["http://localhost:3001", "http://localhost:8000"],
echo     allow_credentials=True,
echo     allow_methods=["*"],
echo     allow_headers=["*"],
echo )
echo.
echo # API key authentication
echo api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
echo.
echo async def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
echo     auth_header = request.headers.get("Authorization")
echo     bearer_token = None
echo     if auth_header and auth_header.startswith("Bearer "):
echo         bearer_token = auth_header.replace("Bearer ", "")
echo     if api_key == "mcp-secret-key" or bearer_token == "mcp-secret-key":
echo         return True
echo     raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")
echo.
echo # User management
echo def load_users():
echo     try:
echo         with open("data/users.json", "r") as f:
echo             return json.load(f)
echo     except:
echo         return {}
echo.
echo def save_user(user_id, data):
echo     users = load_users()
echo     users[user_id] = data
echo     os.makedirs("data", exist_ok=True)
echo     with open("data/users.json", "w") as f:
echo         json.dump(users, f)
echo.
echo # Vedic knowledge base
echo def load_vedic_knowledge():
echo     try:
echo         with open("data/vedic_knowledge/vedic_knowledge.json", "r") as f:
echo             return json.load(f)
echo     except:
echo         return {"texts": []}
echo.
echo def query_shloka(emotion: str = None, goal: str = None, chakra: str = None, keyword: str = None):
echo     vedas = load_vedic_knowledge()
echo     matches = []
echo     for text in vedas["texts"]:
echo         for section in text["sections"]:
echo             for verse in section["verses"]:
echo                 if (
echo                     (not emotion or emotion.lower() in [e.lower() for e in verse.get("emotion", [])]) and
echo                     (not goal or goal.lower() in [g.lower() for g in verse.get("goal", [])]) and
echo                     (not chakra or chakra.lower() == verse.get("chakra", "").lower()) and
echo                     (not keyword or keyword.lower() in verse.get("text", "").lower() or keyword.lower() in verse.get("translation", "").lower())
echo                 ):
echo                     matches.append(verse)
echo     return matches[0] if matches else {
echo         "mantra": "Om Namah Shivaya",
echo         "frequency": 432.0,
echo         "tone": "G",
echo         "repetitions": 108,
echo         "text": "Default mantra",
echo         "translation": "Universal chant"
echo     }
echo.
echo # Pydantic models
echo class MantraRequest(BaseModel):
echo     emotion: str
echo     goal: str
echo     chakra: Optional[str] = None
echo.
echo class MantraResponse(BaseModel):
echo     mantra: str
echo     frequency: float
echo     tone: str
echo     repetitions: int
echo     text: Optional[str]
echo     translation: Optional[str]
echo.
echo class ChantRequest(BaseModel):
echo     mantra: str
echo     frequency: float
echo     tone: str
echo.
echo class ChantResponse(BaseModel):
echo     audio_path: str
echo.
echo class EventLogRequest(BaseModel):
echo     user_id: str
echo     mantra: str
echo     vibration_level: float
echo     emotional_state: str
echo.
echo class TapSadhanaRequest(BaseModel):
echo     user_id: str
echo     schedule_name: str
echo.
echo class VoiceFeedbackRequest(BaseModel):
echo     user_id: str
echo     audio_path: str
echo.
echo class VoiceFeedbackResponse(BaseModel):
echo     energy_kpis: dict
echo.
echo class ChatMessage(BaseModel):
echo     role: str
echo     content: str
echo.
echo class ChatRequest(BaseModel):
echo     model: str
echo     messages: List[ChatMessage]
echo     max_tokens: Optional[int] = 100
echo     temperature: Optional[float] = 0.7
echo     stream: Optional[bool] = False
echo.
echo class ChatResponse(BaseModel):
echo     choices: List[dict]
echo     created: int
echo     model: str
echo     usage: dict
echo.
echo # Dynamic command parser
echo def parse_command(message: str):
echo     message = message.lower().strip()
echo     commands = {
echo         "start tapasya": r"start tapasya\s+(\w+)\s+for\s+user(\w+)",
echo         "log event": r"log event\s+for\s+user(\w+):\s*mantra\s+(.+?),\s*vibration\s+([\d.]+),\s*state\s+(.+)",
echo         "generate chant": r"generate chant\s+for\s+(.+?)(?:,\s*frequency\s+([\d.]+))?(?:,\s*tone\s+(\w+))?",
echo         "analyze chant": r"analyze my chant\s+from\s+(.+?)\s+for\s+user(\w+)",
echo         "suggest mantra": r"(?:suggest a mantra for\s+)?(?:emotion:\s*(\w+),\s*goal:\s*(\w+)(?:,\s*chakra:\s*(\w+))?)|(\w+)\s+(\w+)(?:\s+(\w+))?",
echo         "find shloka": r"(?:find a shloka for|shloka about)\s+(.+)"
echo     }
echo     for cmd, pattern in commands.items():
echo         match = re.match(pattern, message)
echo         if match:
echo             return cmd, match.groups()
echo     return None, None
echo.
echo # Root endpoint
echo @app.get("/v1")
echo async def v1_root():
echo     return {
echo         "message": "Welcome to MCP Server - Whispering Vedas API.",
echo         "usage": "Use POST /v1/chat/completions with X-API-Key or Bearer token. Examples: 'Suggest a mantra for emotion: calm, goal: clarity, chakra: third_eye', 'calm clarity third_eye', 'Find a shloka for peace', 'Start tapasya daily_vedic for user123', 'Log event for user123: mantra Om Namah Shivaya, vibration 7.5, state peaceful', 'Generate chant for Om Namah Shivaya', 'Analyze my chant from sample.wav for user123'."
echo     }
echo.
echo # Debug endpoint
echo @app.get("/v1/debug")
echo async def debug(api_key_verified: bool = Depends(verify_api_key)):
echo     return {"status": "ok", "message": "Authorization working!"}
echo.
echo # GET /v1/chat/completions
echo @app.get("/v1/chat/completions")
echo async def chat_completions_get():
echo     return {
echo         "message": "This endpoint requires a POST request. Use /docs for testing or send a JSON payload with model and messages."
echo     }
echo.
echo # OpenAI-compatible endpoint
echo @app.post("/v1/chat/completions", response_model=ChatResponse)
echo async def chat_completions(request: ChatRequest, api_key_verified: bool = Depends(verify_api_key)):
echo     try:
echo         user_message = next((msg.content for msg in request.messages if msg.role == "user"), "")
echo         if not user_message:
echo             raise HTTPException(status_code=400, detail="No user message provided")
echo.
echo         response_content = ""
echo         cmd, params = parse_command(user_message.lower())
echo.
echo         if cmd == "start tapasya":
echo             schedule_name, user_id = params
echo             start_tap_sadhana(user_id, schedule_name)
echo             save_user(user_id, {"schedule": schedule_name, "last_tapasya": int(time.time())})
echo             response_content = f"Tapasya schedule '{schedule_name}' started for user {user_id}"
echo.
echo         elif cmd == "log event":
echo             user_id, mantra, vibration_level, emotional_state = params
echo             vibration_level = float(vibration_level)
echo             log_event(user_id, mantra, vibration_level, emotional_state)
echo             save_user(user_id, {"last_event": {"mantra": mantra, "vibration": vibration_level, "state": emotional_state}})
echo             response_content = f"Event logged for user {user_id}: {mantra}"
echo.
echo         elif cmd == "generate chant":
echo             mantra, frequency, tone = params
echo             frequency = float(frequency) if frequency else 432.0
echo             tone = tone if tone else "G"
echo             audio_path = await generate_chant(mantra, frequency, tone)
echo             response_content = f"Chant generated at {audio_path}"
echo.
echo         elif cmd == "analyze chant":
echo             audio_path, user_id = params
echo             kpis = process_voice_feedback(user_id, audio_path)
echo             save_user(user_id, {"last_feedback": kpis})
echo             response_content = f"Voice feedback: {kpis}"
echo.
echo         elif cmd == "suggest mantra" or cmd == "find shloka":
echo             emotion, goal, chakra, simple_emotion, simple_goal, simple_chakra = params + (None,) * (6 - len(params)) if params else (None,) * 6
echo             keyword = params[0] if cmd == "find shloka" else None
echo             if simple_emotion and simple_goal:
echo                 emotion, goal, chakra = simple_emotion, simple_goal, simple_chakra
echo             elif keyword:
echo                 emotion_words = ["calm", "peace", "happy", "focus"]
echo                 goal_words = ["clarity", "balance", "strength", "spiritual growth"]
echo                 chakra_words = ["root", "heart", "third_eye", "solar_plexus"]
echo                 emotion = goal = chakra = None
echo                 for word in keyword.lower().split():
echo                     if word in emotion_words:
echo                         emotion = word
echo                     if word in goal_words:
echo                         goal = word
echo                     if word in chakra_words:
echo                         chakra = word
echo                 keyword = keyword.lower()
echo             shloka = query_shloka(emotion, goal, chakra, keyword)
echo             response_content = (
echo                 f"Mantra: {shloka['mantra']}\n"
echo                 f"Frequency: {shloka['frequency']} Hz\n"
echo                 f"Tone: {shloka['tone']}\n"
echo                 f"Repetitions: {shloka['repetitions']}\n"
echo                 f"Shloka: {shloka['text']}\n"
echo                 f"Translation: {shloka['translation']}"
echo             )
echo.
echo         else:
echo             emotion_words = ["calm", "peace", "happy", "focus"]
echo             goal_words = ["clarity", "balance", "strength", "spiritual growth"]
echo             chakra_words = ["root", "heart", "third_eye", "solar_plexus"]
echo             emotion = goal = chakra = keyword = None
echo             for word in user_message.lower().split():
echo                 if word in emotion_words:
echo                     emotion = word
echo                 if word in goal_words:
echo                     goal = word
echo                 if word in chakra_words:
echo                     chakra = word
echo             shloka = query_shloka(emotion, goal, chakra, keyword)
echo             response_content = (
echo                 f"Mantra: {shloka['mantra']}\n"
echo                 f"Frequency: {shloka['frequency']} Hz\n"
echo                 f"Tone: {shloka['tone']}\n"
echo                 f"Repetitions: {shloka['repetitions']}\n"
echo                 f"Shloka: {shloka['text']}\n"
echo                 f"Translation: {shloka['translation']}"
echo             )
echo.
echo         if request.stream:
echo             def generate_stream():
echo                 for word in response_content.split():
echo                     payload = {
echo                         "choices": [
echo                             {
echo                                 "delta": {"content": word + " "},
echo                                 "finish_reason": None
echo                             }
echo                         ]
echo                     }
echo                     yield f"data: {json.dumps(payload)}\n\n"
echo                     time.sleep(0.1)
echo                 yield "data: {\"choices\":[{\"delta\": {}, \"finish_reason\": \"stop\"}]}\n\n"
echo                 yield "data: [DONE]\n\n"
echo             return StreamingResponse(generate_stream(), media_type="text/event-stream")
echo.
echo         return {
echo             "choices": [
echo                 {
echo                     "message": {
echo                         "role": "assistant",
echo                         "content": response_content
echo                     },
echo                     "finish_reason": "stop"
echo                 }
echo             ],
echo             "created": int(time.time()),
echo             "model": request.model,
echo             "usage": {
echo                 "prompt_tokens": len(user_message.split()),
echo                 "completion_tokens": len(response_content.split()),
echo                 "total_tokens": len(user_message.split()) + len(response_content.split())
echo             }
echo         }
echo     except Exception as e:
echo         print("🔥 Exception occurred:", str(e))
echo         raise HTTPException(status_code=500, detail=str(e))
echo.
echo # Existing endpoints
echo @app.post("/suggest_mantra", response_model=MantraResponse)
echo async def suggest_mantra_endpoint(request: MantraRequest):
echo     try:
echo         shloka = query_shloka(request.emotion, request.goal, request.chakra)
echo         return {
echo             "mantra": shloka["mantra"],
echo             "frequency": shloka["frequency"],
echo             "tone": shloka["tone"],
echo             "repetitions": shloka["repetitions"],
echo             "text": shloka.get("text"),
echo             "translation": shloka.get("translation")
echo         }
echo     except Exception as e:
echo         raise HTTPException(status_code=500, detail=str(e))
echo.
echo @app.post("/generate_chant", response_model=ChantResponse)
echo async def generate_chant_endpoint(request: ChantRequest):
echo     try:
echo         audio_path = await generate_chant(request.mantra, request.frequency, request.tone)
echo         return {"audio_path": audio_path}
echo     except Exception as e:
echo         raise HTTPException(status_code=500, detail=str(e))
echo.
echo @app.post("/log_event")
echo async def log_event_endpoint(request: EventLogRequest):
echo     try:
echo         log_event(request.user_id, request.mantra, request.vibration_level, request.emotional_state)
echo         return {"status": "success"}
echo     except Exception as e:
echo         raise HTTPException(status_code=500, detail=str(e))
echo.
echo @app.post("/start_tap_sadhana")
echo async def start_tap_sadhana_endpoint(request: TapSadhanaRequest):
echo     try:
echo         start_tap_sadhana(request.user_id, request.schedule_name)
echo         return {"status": "Tapasya schedule started"}
echo     except Exception as e:
echo         raise HTTPException(status_code=500, detail=str(e))
echo.
echo @app.post("/voice_feedback", response_model=VoiceFeedbackResponse)
echo async def voice_feedback_endpoint(request: VoiceFeedbackRequest):
echo     try:
echo         kpis = process_voice_feedback(request.user_id, request.audio_path)
echo         return {"energy_kpis": kpis}
echo     except Exception as e:
echo         raise HTTPException(status_code=500, detail=str(e))
echo.
echo if __name__ == "__main__":
echo     import uvicorn
echo     os.makedirs("data/user_logs", exist_ok=True)
echo     os.makedirs("data/chants", exist_ok=True)
echo     os.makedirs("data/vedic_knowledge", exist_ok=True)
echo     uvicorn.run(app, host="0.0.0.0", port=8000)
) > main.py
IF ERRORLEVEL 1 (
    ECHO Failed to create main.py.
    pause
    exit /b 1
)

:: Create chant_generation.py
ECHO Creating chant_generation.py...
(
echo import httpx
echo import os
echo import time
echo.
echo async def generate_chant(mantra: str, frequency: float, tone: str) -> str:
echo     async with httpx.AsyncClient() as client:
echo         try:
echo             response = await client.post(
echo                 "http://localhost:8081/generate",
echo                 json={"mantra": mantra, "frequency": frequency, "tone": tone}
echo             )
echo             response.raise_for_status()
echo             audio_path = response.json().get("audio_path", "data/chant_output.wav")
echo             return audio_path
echo         except httpx.HTTPError:
echo             # Fallback: Save mantra as text file
echo             audio_path = f"data/chants/{mantra.replace(' ', '_')}_{int(time.time())}.txt"
echo             os.makedirs("data/chants", exist_ok=True)
echo             with open(audio_path, "w") as f:
echo                 f.write(f"Mantra: {mantra}, Frequency: {frequency} Hz, Tone: {tone}")
echo             return audio_path
) > modules\chant_generation.py
IF ERRORLEVEL 1 (
    ECHO Failed to create chant_generation.py.
    pause
    exit /b 1
)

:: Create mantra_suggestion.py
ECHO Creating mantra_suggestion.py...
(
echo import json
echo import os
echo.
echo def suggest_mantra(emotion: str, goal: str, chakra: str = None) -> dict:
echo     try:
echo         with open("data/vedic_knowledge/vedic_knowledge.json", "r") as f:
echo             vedas = json.load(f)
echo     except:
echo         return {
echo             "mantra": "Om Namah Shivaya",
echo             "frequency": 432.0,
echo             "tone": "G",
echo             "repetitions": 108
echo         }
echo.
echo     for text in vedas["texts"]:
echo         for section in text["sections"]:
echo             for verse in section["verses"]:
echo                 if (
echo                     emotion.lower() in [e.lower() for e in verse.get("emotion", [])] and
echo                     goal.lower() in [g.lower() for g in verse.get("goal", [])] and
echo                     (not chakra or chakra.lower() == verse.get("chakra", "").lower())
echo                 ):
echo                     return {
echo                         "mantra": verse["mantra"],
echo                         "frequency": verse["frequency"],
echo                         "tone": verse["tone"],
echo                         "repetitions": verse["repetitions"],
echo                         "text": verse["text"],
echo                         "translation": verse["translation"]
echo                     }
echo     return {
echo         "mantra": "Om Namah Shivaya",
echo         "frequency": 432.0,
echo         "tone": "G",
echo         "repetitions": 108
echo     }
) > modules\mantra_suggestion.py
IF ERRORLEVEL 1 (
    ECHO Failed to create mantra_suggestion.py.
    pause
    exit /b 1
)

:: Create event_logging.py
ECHO Creating event_logging.py...
(
echo import os
echo import time
echo import json
echo.
echo def log_event(user_id: str, mantra: str, vibration_level: float, emotional_state: str):
echo     log_entry = {
echo         "user_id": user_id,
echo         "mantra": mantra,
echo         "vibration_level": vibration_level,
echo         "emotional_state": emotional_state,
echo         "timestamp": int(time.time())
echo     }
echo     os.makedirs("data/user_logs", exist_ok=True)
echo     log_file = f"data/user_logs/{user_id}_events.json"
echo     logs = []
echo     try:
echo         with open(log_file, "r") as f:
echo             logs = json.load(f)
echo     except:
echo         pass
echo     logs.append(log_entry)
echo     with open(log_file, "w") as f:
echo         json.dump(logs, f)
) > modules\event_logging.py
IF ERRORLEVEL 1 (
    ECHO Failed to create event_logging.py.
    pause
    exit /b 1
)

:: Create tap_sadhana.py
ECHO Creating tap_sadhana.py...
(
echo from apscheduler.schedulers.background import BackgroundScheduler
echo import time
echo.
echo scheduler = BackgroundScheduler()
echo scheduler.start()
echo.
echo def start_tap_sadhana(user_id: str, schedule_name: str):
echo     def log_sadhana():
echo         print(f"Running tapasya for {user_id}: {schedule_name} at {time.time()}")
echo.
echo     if schedule_name == "daily_vedic":
echo         scheduler.add_job(log_sadhana, 'interval', days=1)
echo     else:
echo         scheduler.add_job(log_sadhana, 'interval', days=1)
) > modules\tap_sadhana.py
IF ERRORLEVEL 1 (
    ECHO Failed to create tap_sadhana.py.
    pause
    exit /b 1
)

:: Create voice_feedback.py
ECHO Creating voice_feedback.py...
(
echo import speech_recognition as sr
echo import time
echo.
echo def process_voice_feedback(user_id: str, audio_path: str) -> dict:
echo     try:
echo         recognizer = sr.Recognizer()
echo         with sr.AudioFile(audio_path) as source:
echo             audio = recognizer.record(source)
echo         text = recognizer.recognize_google(audio)
echo         return {
echo             "user_id": user_id,
echo             "recognized_text": text,
echo             "confidence": 0.9,
echo             "timestamp": int(time.time())
echo         }
echo     except Exception as e:
echo         return {
echo             "user_id": user_id,
echo             "error": f"Could not recognize audio: {str(e)}",
echo             "timestamp": int(time.time())
echo         }
) > modules\voice_feedback.py
IF ERRORLEVEL 1 (
    ECHO Failed to create voice_feedback.py.
    pause
    exit /b 1
)

:: Start the server
ECHO Starting MCP Server...
%PYTHON% main.py
IF ERRORLEVEL 1 (
    ECHO Failed to start server. Check main.py or dependencies.
    pause
    exit /b 1
)

pause