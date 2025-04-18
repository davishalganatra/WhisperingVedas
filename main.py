from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from starlette.status import HTTP_403_FORBIDDEN
import os
import time
import json
import re
from modules.mantra_suggestion import suggest_mantra
from modules.chant_generation import generate_chant, play_chant
from modules.akashic_logger import log_chant_action, get_user_logs
from modules.tap_sadhana_scheduler import schedule_tapasya, remove_tapasya
from modules.user_profiles import create_user_profile, get_user_profile, UserProfile
from modules.event_logging import log_event
from modules.tap_sadhana import start_tap_sadhana
from modules.voice_feedback import process_voice_feedback
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Server - Whispering Vedas", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:8000", "http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    auth_header = request.headers.get("Authorization")
    bearer_token = None
    if auth_header and auth_header.startswith("Bearer "):
        bearer_token = auth_header.replace("Bearer ", "")
    if api_key == "mcp-secret-key" or bearer_token == "mcp-secret-key":
        return True
    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")

def load_vedic_knowledge():
    try:
        with open("data/vedic_knowledge/vedic_knowledge.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load vedic_knowledge.json: {str(e)}")
        return {
            "texts": [
                {
                    "name": "Default",
                    "sections": [
                        {
                            "name": "Default",
                            "verses": [
                                {
                                    "id": "default_0",
                                    "text": "Om Namah Shivaya",
                                    "translation": "Universal chant",
                                    "emotion": [],
                                    "goal": [],
                                    "chakra": None,
                                    "mantra": "Om Namah Shivaya",
                                    "frequency": 432.0,
                                    "tone": "G",
                                    "repetitions": 108
                                }
                            ]
                        }
                    ]
                }
            ]
        }

class MantraRequest(BaseModel):
    emotion: str
    goal: str
    chakra: Optional[str] = None
    user_id: Optional[str] = None
    astrological_context: Optional[str] = None  # Placeholder for future use

class TapSadhanaRequest(BaseModel):
    user_id: str
    schedule_name: str
    mantra: str
    repetitions: int
    time_of_day: str = "06:00"

class UserProfileRequest(BaseModel):
    user_id: str
    name: str
    goals: Optional[List[str]] = None
    chakra_focus: Optional[str] = None
    healing_preferences: Optional[List[str]] = None
    preferred_tone: Optional[str] = "G"

class EventLogRequest(BaseModel):
    user_id: str
    mantra: str
    vibration_level: float
    emotional_state: str

class VoiceFeedbackRequest(BaseModel):
    user_id: str
    audio_path: str

@app.get("/v1")
async def root():
    return {
        "message": "Welcome to MCP Server - Whispering Vedas API.",
        "usage": "Use POST /suggest_mantra, /start_tap_sadhana, /log_event, /process_voice_feedback, etc."
    }

@app.post("/suggest_mantra")
async def suggest_mantra_endpoint(request: MantraRequest, authorized: bool = Depends(verify_api_key)):
    mantra_response = suggest_mantra(
        emotion=request.emotion,
        goal=request.goal,
        chakra=request.chakra,
        user_id=request.user_id
    )
    # Log the suggestion as an event
    log_event(
        user_id=request.user_id or "user123",
        mantra=mantra_response["mantra"],
        vibration_level=mantra_response["frequency"] / 60.0,  # Simplified conversion
        emotional_state=request.emotion or "calm"
    )
    logger.info(f"Suggested mantra for user {request.user_id}: {mantra_response['mantra']}")
    return mantra_response

@app.post("/generate_chant")
async def generate_chant_endpoint(mantra: str, frequency: float, tone: str, authorized: bool = Depends(verify_api_key)):
    audio_path = await generate_chant(mantra, frequency, tone)
    return {"audio_path": audio_path}

@app.post("/play_chant")
async def play_chant_endpoint(audio_path: str, authorized: bool = Depends(verify_api_key)):
    result = await play_chant(audio_path)
    return {"message": result}

@app.post("/log_event")
async def log_event_endpoint(request: EventLogRequest, authorized: bool = Depends(verify_api_key)):
    result = log_event(
        user_id=request.user_id,
        mantra=request.mantra,
        vibration_level=request.vibration_level,
        emotional_state=request.emotional_state
    )
    logger.info(f"Logged event for user {request.user_id}: {request.mantra}")
    return result

@app.post("/start_tap_sadhana")
async def start_tap_sadhana_endpoint(request: TapSadhanaRequest, authorized: bool = Depends(verify_api_key)):
    result = schedule_tapasya(
        user_id=request.user_id,
        schedule_name=request.schedule_name,
        mantra=request.mantra,
        repetitions=request.repetitions,
        time_of_day=request.time_of_day
    )
    logger.info(f"Scheduled tapasya for user {request.user_id}: {request.schedule_name}")
    return result

@app.post("/create_user_profile")
async def create_user_profile_endpoint(request: UserProfileRequest, authorized: bool = Depends(verify_api_key)):
    result = create_user_profile(
        user_id=request.user_id,
        name=request.name,
        goals=request.goals,
        chakra_focus=request.chakra_focus,
        healing_preferences=request.healing_preferences,
        preferred_tone=request.preferred_tone
    )
    logger.info(f"Created profile for user {request.user_id}")
    return result

@app.get("/get_user_logs/{user_id}")
async def get_user_logs_endpoint(user_id: str, authorized: bool = Depends(verify_api_key)):
    logs = get_user_logs(user_id)
    logger.info(f"Fetched logs for user {user_id}")
    return logs

@app.post("/process_voice_feedback")
async def process_voice_feedback_endpoint(request: VoiceFeedbackRequest, authorized: bool = Depends(verify_api_key)):
    kpis = process_voice_feedback(
        user_id=request.user_id,
        audio_path=request.audio_path
    )
    logger.info(f"Processed voice feedback for user {request.user_id}")
    return kpis
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)