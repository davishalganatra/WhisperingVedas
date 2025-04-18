from modules.akashic_logger import log_chant_action
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def start_tap_sadhana(user_id: str, mantra: str, repetitions: int, emotional_state: str = "neutral"):
    """Start a tapasya session by logging the chant action."""
    try:
        result = log_chant_action(user_id, mantra, repetitions, "manual", emotional_state)
        logger.info(f"Started tapasya for user {user_id}: {mantra}")
        return {"status": "success", "message": f"Tapasya started: {mantra}, {repetitions} repetitions"}
    except Exception as e:
        logger.error(f"Failed to start tapasya for user {user_id}: {str(e)}")
        return {"status": "error", "message": f"Failed to start tapasya: {str(e)}"}