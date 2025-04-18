import gradio as gr
import httpx
import json
import os
import re
from datetime import datetime
import logging
import tempfile
from pathlib import Path
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
API_KEY = "mcp-secret-key"

async def query_mcp_server(endpoint: str, payload: dict = None, files: dict = None):
    """Make a POST request to MCP Server."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            logger.info(f"Querying {BASE_URL}/{endpoint} with payload: {payload}, files: {files}")
            headers = {"X-API-Key": API_KEY}
            if files:
                response = await client.post(
                    f"{BASE_URL}/{endpoint}",
                    data=payload,
                    files=files,
                    headers=headers
                )
            else:
                response = await client.post(
                    f"{BASE_URL}/{endpoint}",
                    json=payload,
                    headers=headers
                )
            response.raise_for_status()
            logger.info(f"Response from {endpoint}: {response.json()}")
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to query {endpoint}: {str(e)}")
            return {"error": f"Failed to query {endpoint}: {str(e)}"}

def parse_user_query(query: str, audio_file: str = None):
    """Parse user query to extract intent and parameters using rule-based NLP."""
    query = query.lower().strip() if query else ""

    intents = {
        "suggest_mantra": ["suggest", "mantra", "help", "recommend"],
        "start_tap_sadhana": ["start", "tapasya", "schedule", "daily"],
        "log_event": ["log", "record", "chant"],
        "voice_feedback": ["voice", "feedback", "audio"]
    }
    emotions = ["heavy", "calm", "peace", "focus", "sad", "happy", "anxious", "angry"]
    goals = ["clarity", "balance", "strength", "spiritual growth", "peace", "healing"]
    chakras = ["root", "heart", "third_eye", "solar_plexus", "throat", "crown"]

    result = {
        "intent": None,
        "emotion": None,
        "goal": None,
        "chakra": None,
        "user_id": "user123",
        "context": "manual",
        "audio_path": audio_file,
        "astrological_context": "neutral",
        "schedule_name": "daily_vedic",
        "mantra": "Om Namah Shivaya",
        "repetitions": 108,
        "time_of_day": "06:00",
        "vibration_level": 7.5,
        "emotional_state": "peaceful"
    }

    # Prioritize voice_feedback if audio file is provided
    if audio_file:
        result["intent"] = "voice_feedback"
        result["emotional_state"] = "neutral"
    else:
        for intent, keywords in intents.items():
            if any(keyword in query for keyword in keywords):
                result["intent"] = intent
                break

        for word in query.split():
            if word in emotions:
                result["emotion"] = word
            if word in goals:
                result["goal"] = word
            if word in chakras:
                result["chakra"] = word

        if result["intent"] == "start_tap_sadhana":
            # Extract mantra
            match = re.search(r"with\s+(.+?)(?:\s+\d+|$)", query)
            if match:
                result["mantra"] = match.group(1).strip()
            # Extract repetitions
            match = re.search(r"(\d+)\s+repetitions", query)
            if match:
                try:
                    result["repetitions"] = int(match.group(1))
                except ValueError:
                    result["repetitions"] = 108
            # Extract time
            match = re.search(r"at\s+(\d{2}:\d{2})", query)
            if match:
                result["time_of_day"] = match.group(1)
            result["schedule_name"] = "daily_vedic"

        if result["intent"] == "log_event":
            match = re.search(r"chant\s+(.+?)(?:\s|$)", query)
            if match:
                result["mantra"] = match.group(1).strip()
            result["emotional_state"] = result["emotion"] or "peaceful"

        if not result["intent"]:
            result["intent"] = "suggest_mantra"

    logger.info(f"Parsed query '{query}' with audio '{audio_file}' to: {result}")
    return result

async def process_query(query: str, audio_file: str = None):
    """Process user query and fetch response from MCP Server."""
    if audio_file:
        file_path = Path(audio_file)
        temp_wav = None
        if file_path.suffix.lower() == ".mp3":
            try:
                temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
                audio = AudioSegment.from_mp3(audio_file)
                audio.export(temp_wav, format="wav")
                audio_file = temp_wav
                logger.info(f"Converted MP3 to WAV: {audio_file}")
            except Exception as e:
                return f"Error converting MP3 to WAV: {str(e)}", []
        elif file_path.suffix.lower() != ".wav":
            return "Error: Please upload a WAV or MP3 file.", []
    
    parsed = parse_user_query(query, audio_file)
    intent = parsed["intent"]
    user_id = parsed["user_id"]

    if intent == "suggest_mantra":
        payload = {
            "user_id": user_id,
            "emotion": parsed["emotion"] or "calm",
            "goal": parsed["goal"] or "clarity",
            "chakra": parsed["chakra"],
            "astrological_context": parsed["astrological_context"]
        }
        logger.debug(f"Sending payload to /suggest_mantra: {payload}")
        response = await query_mcp_server("suggest_mantra", payload)
        if "error" in response:
            return response["error"], []
        mantra_response = response
        event_payload = {
            "user_id": user_id,
            "mantra": mantra_response["mantra"],
            "vibration_level": mantra_response["frequency"] / 60.0,
            "emotional_state": parsed["emotion"] or "calm"
        }
        await query_mcp_server("log_event", event_payload)
        logs = await query_mcp_server(f"get_user_logs/{user_id}", {})
        logs = logs if isinstance(logs, list) else []
        logs_display = [
            f"{datetime.fromtimestamp(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} - "
            f"Mantra: {log['mantra']}, Reps: {log['repetitions']}, Context: {log['context']}, Emotion: {log.get('emotional_state', 'neutral')}"
            for log in logs[-5:]
        ]
        return (
            f"Mantra: {mantra_response['mantra']}\n"
            f"Frequency: {mantra_response['frequency']} Hz\n"
            f"Tone: {mantra_response['tone']}\n"
            f"Repetitions: {mantra_response['repetitions']}\n"
            f"Shloka: {mantra_response['text']}\n"
            f"Translation: {mantra_response['translation']}",
            logs_display
        )

    elif intent == "start_tap_sadhana":
        payload = {
            "user_id": user_id,
            "schedule_name": parsed["schedule_name"],
            "mantra": parsed["mantra"],
            "repetitions": int(parsed["repetitions"]),
            "time_of_day": parsed["time_of_day"]
        }
        logger.debug(f"Sending payload to /start_tap_sadhana: {payload}")
        response = await query_mcp_server("start_tap_sadhana", payload)
        if "error" in response:
            return response["error"], []
        logs = await query_mcp_server(f"get_user_logs/{user_id}", {})
        logs = logs if isinstance(logs, list) else []
        logs_display = [
            f"{datetime.fromtimestamp(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} - "
            f"Mantra: {log['mantra']}, Reps: {log['repetitions']}, Context: {log['context']}, Emotion: {log.get('emotional_state', 'neutral')}"
            for log in logs[-5:]
        ]
        return response["message"], logs_display

    elif intent == "log_event":
        payload = {
            "user_id": user_id,
            "mantra": parsed["mantra"],
            "vibration_level": parsed["vibration_level"],
            "emotional_state": parsed["emotional_state"]
        }
        logger.debug(f"Sending payload to /log_event: {payload}")
        response = await query_mcp_server("log_event", payload)
        if "error" in response:
            return response["error"], []
        logs = await query_mcp_server(f"get_user_logs/{user_id}", {})
        logs = logs if isinstance(logs, list) else []
        logs_display = [
            f"{datetime.fromtimestamp(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} - "
            f"Mantra: {log['mantra']}, Reps: {log['repetitions']}, Context: {log['context']}, Emotion: {log.get('emotional_state', 'neutral')}"
            for log in logs[-5:]
        ]
        return "Event logged successfully", logs_display

    elif intent == "voice_feedback" and audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            with open(audio_file, "rb") as f:
                temp_file.write(f.read())
            temp_file_path = temp_file.name
        
        files = {"file": (os.path.basename(temp_file_path), open(temp_file_path, "rb"), "audio/wav")}
        payload = {"user_id": user_id}
        logger.debug(f"Sending payload to /process_voice_feedback: {payload}, files: {files}")
        response = await query_mcp_server("process_voice_feedback", payload, files)
        
        os.unlink(temp_file_path)
        if temp_wav:
            os.unlink(temp_wav)
        
        if "error" in response:
            return response["error"], []
        
        event_payload = {
            "user_id": user_id,
            "mantra": "Voice Feedback",
            "vibration_level": response.get("energy_level", 0.0),
            "emotional_state": parsed["emotional_state"] or "neutral"
        }
        await query_mcp_server("log_event", event_payload)
        logs = await query_mcp_server(f"get_user_logs/{user_id}", {})
        logs = logs if isinstance(logs, list) else []
        logs_display = [
            f"{datetime.fromtimestamp(log['timestamp']).strftime('%Y-%m-%d %H:%M:%S')} - "
            f"Mantra: {log['mantra']}, Reps: {log['repetitions']}, Context: {log['context']}, Emotion: {log.get('emotional_state', 'neutral')}"
            for log in logs[-5:]
        ]
        return (
            f"Voice Feedback Processed:\n"
            f"Energy Level: {response.get('energy_level', 0.0)}\n"
            f"Clarity Score: {response.get('clarity_score', 0.0)}\n"
            f"Message: {response.get('message', 'Processed')}",
            logs_display
        )

    return "Please provide a valid query or audio input", []

def main():
    """Launch Gradio interface."""
    with gr.Blocks(title="Divyam Rishi - Whispering Vedas") as demo:
        gr.Markdown("""
        # Divyam Rishi - Your Spiritual Assistant
        Welcome to **Whispering Vedas**, an AI-powered platform to guide your Vedic chanting and spiritual journey. 
        Divyam Rishi helps you:
        - **Discover Mantras**: Get personalized mantra suggestions based on your emotions and goals (e.g., "I feel heavy, suggest a mantra for my heart chakra").
        - **Schedule Tapasya**: Plan daily chanting practices (e.g., "Start tapasya with Om Namah Shivaya 108 repetitions at 06:00").
        - **Log Chanting**: Record your chanting sessions (e.g., "Log chant Om Namah Shivaya").
        - **Analyze Voice**: Upload a WAV or MP3 file of your chant to analyze its energy and clarity (e.g., "Process voice feedback" with an audio upload).

        **How to Use**:
        - Enter a query in the textbox below, such as:
          - "Suggest a mantra for peace"
          - "Start tapasya with Om Shanti 54 repetitions at 07:00"
          - "Process voice feedback" (with a WAV or MP3 file)
        - Upload a WAV or MP3 file for voice analysis.
        - Click **Submit** to see the response and recent logs.
        """)
        with gr.Row():
            query_input = gr.Textbox(label="Enter your query (e.g., 'Start tapasya with Om Namah Shivaya 108 repetitions at 06:00')")
            audio_input = gr.Audio(sources=["upload"], type="filepath", label="Upload Voice Input (WAV or MP3)")
        submit_button = gr.Button("Submit")
        response_output = gr.Textbox(label="Response", lines=10)
        logs_output = gr.Textbox(label="Recent Logs", lines=5)

        submit_button.click(
            fn=process_query,
            inputs=[query_input, audio_input],
            outputs=[response_output, logs_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()