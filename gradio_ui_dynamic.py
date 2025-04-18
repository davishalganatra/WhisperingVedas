import gradio as gr
import httpx
import json
import os
import re
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"
API_KEY = "mcp-secret-key"

async def query_mcp_server(endpoint: str, payload: dict):
    """Make a POST request to MCP Server."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            logger.info(f"Querying {BASE_URL}/{endpoint} with payload: {payload}")
            response = await client.post(
                f"{BASE_URL}/{endpoint}",
                json=payload,
                headers={"X-API-Key": API_KEY}
            )
            response.raise_for_status()
            logger.info(f"Response from {endpoint}: {response.json()}")
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Failed to query {endpoint}: {str(e)}")
            return {"error": f"Failed to query {endpoint}: {str(e)}"}

def parse_user_query(query: str):
    """Parse user query to extract intent and parameters using rule-based NLP."""
    query = query.lower().strip()

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
        "audio_path": None,
        "astrological_context": "neutral"  # Placeholder for future use
    }

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

    if "start" in query and "tapasya" in query:
        result["intent"] = "start_tap_sadhana"
        result["schedule_name"] = "daily_vedic"
        result["mantra"] = "Om Namah Shivaya"
        result["repetitions"] = 108
        result["time_of_day"] = "06:00"
        if "mantra" in query:
            match = re.search(r"mantra\s+(.+?)(?:\s|$)", query)
            if match:
                result["mantra"] = match.group(1)
        if "repetitions" in query:
            match = re.search(r"repetitions\s+(\d+)", query)
            if match:
                result["repetitions"] = int(match.group(1))
        if "at" in query:
            match = re.search(r"at\s+(\d{2}:\d{2})", query)
            if match:
                result["time_of_day"] = match.group(1)

    if "log" in query or "chant" in query:
        result["intent"] = "log_event"
        result["mantra"] = "Om Namah Shivaya"
        result["vibration_level"] = 7.5
        result["emotional_state"] = result["emotion"] or "peaceful"
        if "mantra" in query:
            match = re.search(r"mantra\s+(.+?)(?:\s|$)", query)
            if match:
                result["mantra"] = match.group(1)

    if "voice" in query or "feedback" in query:
        result["intent"] = "voice_feedback"
        result["audio_path"] = "data/chants/user_feedback.wav"  # Placeholder
        result["emotional_state"] = result["emotion"] or "neutral"

    if not result["intent"]:
        result["intent"] = "suggest_mantra"

    logger.info(f"Parsed query '{query}' to: {result}")
    return result

async def process_query(query: str):
    """Process user query and fetch response from MCP Server."""
    parsed = parse_user_query(query)
    intent = parsed["intent"]
    user_id = parsed["user_id"]

    if intent == "suggest_mantra":
        payload = {
            "emotion": parsed["emotion"] or "calm",
            "goal": parsed["goal"] or "clarity",
            "chakra": parsed["chakra"],
            "user_id": user_id,
            "astrological_context": parsed["astrological_context"]
        }
        response = await query_mcp_server("suggest_mantra", payload)
        if "error" in response:
            return response["error"], []
        mantra_response = response
        # Log the suggestion as an event
        event_payload = {
            "user_id": user_id,
            "mantra": mantra_response["mantra"],
            "vibration_level": mantra_response["frequency"] / 60.0,  # Simplified conversion
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
            "repetitions": parsed["repetitions"],
            "time_of_day": parsed["time_of_day"]
        }
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

    elif intent == "voice_feedback":
        payload = {
            "user_id": user_id,
            "audio_path": parsed["audio_path"]
        }
        response = await query_mcp_server("process_voice_feedback", payload)
        if "error" in response:
            return response["error"], []
        # Log the feedback as an event
        event_payload = {
            "user_id": user_id,
            "mantra": "Voice Feedback",
            "vibration_level": response.get("energy_level", 0.0),
            "emotional_state": parsed["emotional_state"]
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

    return "Unknown intent", []

def main():
    """Launch Gradio interface."""
    with gr.Blocks(title="Divyam Rishi - Whispering Vedas") as demo:
        gr.Markdown("# Divyam Rishi - Your Spiritual Assistant")
        query_input = gr.Textbox(label="Enter your query (e.g., 'I feel heavy, help my heart chakra' or 'Process voice feedback')")
        submit_button = gr.Button("Submit")
        response_output = gr.Textbox(label="Response", lines=10)
        logs_output = gr.Textbox(label="Recent Logs", lines=5)

        submit_button.click(
            fn=process_query,
            inputs=query_input,
            outputs=[response_output, logs_output]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860) #pledged to the heart of the cosmos)

if __name__ == "__main__":
    main()