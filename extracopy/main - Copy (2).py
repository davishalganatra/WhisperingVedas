from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from modules.mantra_suggestion import suggest_mantra
from modules.chant_generation import generate_chant
from modules.event_logging import log_event
from modules.tap_sadhana import start_tap_sadhana
from modules.voice_feedback import process_voice_feedback
from fastapi import Request
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN
from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
import time
import json
import traceback

app = FastAPI(title="MCP Server - Whispering Vedas", version="1.0.0")

# Add CORS middleware for AnythingLLM compatibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key authentication disabled
#api_key_header = APIKeyHeader(name="X-API-Key")

#async def verify_api_key(api_key: str = Security(api_key_header)):
#    if api_key != "mcp-secret-key":
#        raise HTTPException(status_code=403, detail="Invalid API key")
#    return api_key
    
# ChatGPT Suggested Method starts
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(request: Request, api_key: str = Security(api_key_header)):
    auth_header = request.headers.get("Authorization")
    bearer_token = None
    if auth_header and auth_header.startswith("Bearer "):
        bearer_token = auth_header.replace("Bearer ", "")

    if api_key == "mcp-secret-key" or bearer_token == "mcp-secret-key":
        return True

    raise HTTPException(status_code=HTTP_403_FORBIDDEN, detail="Invalid API key")
#
#@app.get("/v1/doc")
#async def get_doc(api_key_verified: bool = Depends(verify_api_key)):
#    return {"message": "This is the API documentation"}
#
@app.get("/v1/debug")
async def debug(api_key_verified: bool = Depends(verify_api_key)):
    return {"status": "ok", "message": "Authorization working!"}
# ChatGPT Suggested Method ends 


# Pydantic models
class MantraRequest(BaseModel):
    emotion: str
    goal: str
    chakra: Optional[str] = None

class MantraResponse(BaseModel):
    mantra: str
    frequency: float
    tone: str
    repetitions: int

class ChantRequest(BaseModel):
    mantra: str
    frequency: float
    tone: str

class ChantResponse(BaseModel):
    audio_path: str

class EventLogRequest(BaseModel):
    user_id: str
    mantra: str
    vibration_level: float
    emotional_state: str

class TapSadhanaRequest(BaseModel):
    user_id: str
    schedule_name: str

class VoiceFeedbackRequest(BaseModel):
    user_id: str
    audio_path: str

class VoiceFeedbackResponse(BaseModel):
    energy_kpis: dict

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False  # Add this line to include the stream attribute

class ChatResponse(BaseModel):
    choices: List[dict]
    created: int
    model: str
    usage: dict

# Root endpoint for /v1
@app.get("/v1")
async def v1_root():
    return {"message": "Welcome to MCP Server - Whispering Vedas API. Use POST /v1/chat/completions for mantra suggestions, event logging, or tapasya scheduling."}

# GET endpoint for /v1/chat/completions
@app.get("/v1/chat/completions")
async def chat_completions_get():
    return {
        "message": "This endpoint requires a POST request. Use /docs for testing or send a JSON payload with model and messages."
    }

# OpenAI-compatible endpoint for AnythingLLM
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest, api_key: str = Security(verify_api_key)):
    try:
        # Extract the latest user message
        user_message = next((msg.content for msg in request.messages if msg.role == "user"), "")
        if not user_message:
            raise HTTPException(status_code=400, detail="No user message provided")

        response_content = ""
        # Handle different commands
        if "start tapasya" in user_message.lower():
            user_id = "user123"
            schedule_name = "daily_vedic"
            if "user" in user_message.lower():
                try:
                    user_id = user_message.lower().split("user")[1].split()[0].strip()
                except:
                    pass
            if "daily_vedic" in user_message.lower():
                schedule_name = "daily_vedic"
            elif "other_schedule" in user_message.lower():
                schedule_name = "other_schedule"
            start_tap_sadhana(user_id, schedule_name)
            response_content = f"Tapasya schedule '{schedule_name}' started for user {user_id}"

        elif "log event" in user_message.lower():
            user_id = "user123"
            mantra = "Om Namah Shivaya"
            vibration_level = 7.5
            emotional_state = "peaceful"
            try:
                parts = user_message.lower().split(",")
                for part in parts:
                    if "user" in part:
                        user_id = part.split("user")[1].strip()
                    if "mantra" in part:
                        mantra = part.split("mantra")[1].strip()
                    if "vibration" in part:
                        vibration_level = float(part.split("vibration")[1].strip())
                    if "state" in part:
                        emotional_state = part.split("state")[1].strip()
            except:
                pass
            log_event(user_id, mantra, vibration_level, emotional_state)
            response_content = f"Event logged for user {user_id}: {mantra}"

        else:
            emotion, goal, chakra = "calm", "clarity", None
            if "emotion:" in user_message.lower() and "goal:" in user_message.lower():
                try:
                    parts = user_message.lower().split(",")
                    for part in parts:
                        if "emotion:" in part:
                            emotion = part.split("emotion:")[1].strip()
                        if "goal:" in part:
                            goal = part.split("goal:")[1].strip()
                        if "chakra:" in part:
                            chakra = part.split("chakra:")[1].strip()
                except:
                    pass
            else:
                parts = user_message.lower().split()
                if len(parts) >= 2:
                    emotion = parts[0]
                    goal = parts[1]

            mantra_data = suggest_mantra(emotion, goal, chakra)
            response_content = (
                f"Mantra: {mantra_data['mantra']}\n"
                f"Frequency: {mantra_data['frequency']} Hz\n"
                f"Tone: {mantra_data['tone']}\n"
                f"Repetitions: {mantra_data['repetitions']}"
            )

        # Check if streaming is requested
        if request.stream:
            def generate_stream():
                for word in response_content.split():
                    payload = {
                        "choices": [
                            {
                                "delta": {"content": word + " "},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    time.sleep(0.1)
                # Finish the stream
                yield "data: {\"choices\":[{\"delta\": {}, \"finish_reason\": \"stop\"}]}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_stream(), media_type="text/event-stream")

        # Fallback non-streamed response
                # Fallback non-streamed response
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "created": int(time.time()),
            "model": request.model,
            "usage": {
                "prompt_tokens": len(user_message.split()),
                "completion_tokens": len(response_content.split()),
                "total_tokens": len(user_message.split()) + len(response_content.split())
            }
        }

    except Exception as e:
        import traceback
        print("🔥 Exception occurred:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Existing endpoints
@app.post("/suggest_mantra", response_model=MantraResponse)
async def suggest_mantra_endpoint(request: MantraRequest):
    try:
        result = suggest_mantra(request.emotion, request.goal, request.chakra)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_chant", response_model=ChantResponse)
async def generate_chant_endpoint(request: ChantRequest):
    try:
        audio_path = await generate_chant(request.mantra, request.frequency, request.tone)
        return {"audio_path": audio_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_event")
async def log_event_endpoint(request: EventLogRequest):
    try:
        log_event(request.user_id, request.mantra, request.vibration_level, request.emotional_state)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_tap_sadhana")
async def start_tap_sadhana_endpoint(request: TapSadhanaRequest):
    try:
        start_tap_sadhana(request.user_id, request.schedule_name)
        return {"status": "Tapasya schedule started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/voice_feedback", response_model=VoiceFeedbackResponse)
async def voice_feedback_endpoint(request: VoiceFeedbackRequest):
    try:
        kpis = process_voice_feedback(request.user_id, request.audio_path)
        return {"energy_kpis": kpis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    os.makedirs("data/user_logs", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)