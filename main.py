import uvicorn
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import logging
from datetime import datetime
import tempfile
import os
from modules.voice_analyzer import VoiceAnalyzer
import gradio as gr
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="MCP Server - Whispering Vedas")

API_KEY = "mcp-secret-key"

def verify_api_key(x_api_key: str):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

class MantraRequest(BaseModel):
    user_id: str
    emotion: str
    goal: str
    chakra: Optional[str] = None
    astrological_context: Optional[str] = None

class TapSadhanaRequest(BaseModel):
    user_id: str
    schedule_name: str
    mantra: str
    repetitions: int
    time_of_day: str

class EventLog(BaseModel):
    user_id: str
    mantra: str
    vibration_level: float
    emotional_state: str
    timestamp: Optional[float] = None
    context: Optional[str] = "manual"
    repetitions: Optional[int] = 108

class UserLog(BaseModel):
    timestamp: float
    mantra: str
    repetitions: int
    context: str
    emotional_state: str

# Simulated storage
user_logs = []

@app.post("/suggest_mantra")
async def suggest_mantra(request: MantraRequest, api_key: str = Depends(verify_api_key)):
    logger.info(f"Received /suggest_mantra request: {request.dict()}")
    try:
        response = {
            "mantra": "Om Shanti",
            "frequency": 432.0,
            "tone": "calm",
            "repetitions": 108,
            "text": "Om Shanti",
            "translation": "Peace"
        }
        return response
    except Exception as e:
        logger.error(f"Error in /suggest_mantra: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_tap_sadhana")
async def start_tap_sadhana(request: TapSadhanaRequest, api_key: str = Depends(verify_api_key)):
    logger.info(f"Received /start_tap_sadhana request: {request.dict()}")
    try:
        response = {
            "message": f"Tapasya '{request.schedule_name}' started with mantra '{request.mantra}'"
        }
        return response
    except Exception as e:
        logger.error(f"Error in /start_tap_sadhana: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/log_event")
async def log_event(event: EventLog, api_key: str = Depends(verify_api_key)):
    logger.info(f"Received /log_event request: {event.dict()}")
    try:
        event_dict = event.dict()
        event_dict["timestamp"] = datetime.now().timestamp()
        user_logs.append(event_dict)
        return {"message": "Event logged successfully"}
    except Exception as e:
        logger.error(f"Error in /log_event: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_user_logs/{user_id}")
async def get_user_logs(user_id: str, api_key: str = Depends(verify_api_key)):
    logger.info(f"Received /get_user_logs request for user_id: {user_id}")
    try:
        logs = [log for log in user_logs if log["user_id"] == user_id]
        return logs
    except Exception as e:
        logger.error(f"Error in /get_user_logs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process_voice_feedback")
async def process_voice_feedback(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    api_key: str = Depends(verify_api_key)
):
    logger.info(f"Received /process_voice_feedback request for user_id: {user_id}")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        voice_analyzer = VoiceAnalyzer(use_transcription=False)
        result = voice_analyzer.process_voice(temp_file_path)

        # ✅ Windows-safe file deletion
        import time
        attempts = 3
        for attempt in range(attempts):
            try:
                os.unlink(temp_file_path)
                break
            except PermissionError:
                if attempt < attempts - 1:
                    time.sleep(1)
                else:
                    logger.warning(f"Could not delete temp file after {attempts} attempts: {temp_file_path}")
            except Exception as e:
                logger.warning(f"Unexpected error deleting file: {e}")
                break

        # ✅ Handle analysis errors properly
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        logger.error(f"Error in /process_voice_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/launch_ui")
async def launch_ui():
    logger.info("Received /launch_ui request")
    try:
        from modules.gradio_ui_dynamic import main as launch_gradio
        launch_gradio()
        return {"message": "Gradio UI launched"}
    except Exception as e:
        logger.error(f"Error launching UI: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)