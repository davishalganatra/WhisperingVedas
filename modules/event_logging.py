import os
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def log_event(user_id: str, mantra: str, vibration_level: float, emotional_state: str):
    """Log a user event to data/user_logs/{user_id}_event_log.json."""
    event_entry = {
        "user_id": user_id,
        "mantra": mantra,
        "vibration_level": vibration_level,
        "emotional_state": emotional_state,
        "timestamp": int(time.time())
    }

    os.makedirs("data/user_logs", exist_ok=True)
    log_file = f"data/user_logs/{user_id}_event_log.json"

    events = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            events = json.load(f)
    except:
        pass

    events.append(event_entry)
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        logger.info(f"Logged event for user {user_id}: {mantra}, {emotional_state}")
        return {"status": "success", "message": "Event logged"}
    except Exception as e:
        logger.error(f"Failed to log event for user {user_id}: {str(e)}")
        return {"status": "error", "message": f"Failed to log event: {str(e)}"}