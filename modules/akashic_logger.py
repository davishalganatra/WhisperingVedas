import os
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def log_chant_action(user_id: str, mantra: str, repetitions: int, context: str, emotional_state: str = "neutral"):
    """Log a chanting action to data/user_logs/{user_id}_log.json."""
    log_entry = {
        "user_id": user_id,
        "mantra": mantra,
        "repetitions": repetitions,
        "context": context,  # e.g., "scheduled", "manual", "voice_feedback"
        "emotional_state": emotional_state,
        "timestamp": int(time.time())
    }

    os.makedirs("data/user_logs", exist_ok=True)
    log_file = f"data/user_logs/{user_id}_log.json"

    logs = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except:
        pass

    logs.append(log_entry)
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        logger.info(f"Logged chant action for user {user_id}: {mantra}")
        return {"status": "success", "message": "Log entry added"}
    except Exception as e:
        logger.error(f"Failed to log for user {user_id}: {str(e)}")
        return {"status": "error", "message": f"Failed to log: {str(e)}"}

def get_user_logs(user_id: str):
    """Fetch all logs for a user."""
    log_file = f"data/user_logs/{user_id}_log.json"
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            logs = json.load(f)
        # Ensure logs include emotional_state for display
        for log in logs:
            log["emotional_state"] = log.get("emotional_state", "neutral")
        logger.info(f"Fetched {len(logs)} logs for user {user_id}")
        return logs
    except:
        logger.warning(f"No logs found for user {user_id}")
        return []