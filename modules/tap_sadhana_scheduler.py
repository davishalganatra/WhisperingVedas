from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from modules.akashic_logger import log_chant_action
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

scheduler = BackgroundScheduler()
scheduler.start()

def schedule_tapasya(user_id: str, schedule_name: str, mantra: str, repetitions: int, time_of_day: str = "06:00", emotional_state: str = "neutral"):
    """Schedule a daily mantra chanting routine for a user."""
    def chant_task():
        log_chant_action(user_id, mantra, repetitions, "scheduled", emotional_state)
        logger.info(f"Scheduled tapasya for {user_id}: {mantra}, {repetitions} reps at {time.time()}")

    try:
        hour, minute = map(int, time_of_day.split(":"))
        trigger = CronTrigger(hour=hour, minute=minute)
        job_id = f"{user_id}_{schedule_name}"
        scheduler.add_job(
            chant_task,
            trigger=trigger,
            id=job_id,
            replace_existing=True
        )
        logger.info(f"Scheduled {schedule_name} for {user_id} at {time_of_day}")
        return {"status": "success", "message": f"Tapasya {schedule_name} scheduled at {time_of_day}"}
    except Exception as e:
        logger.error(f"Failed to schedule tapasya for {user_id}: {str(e)}")
        return {"status": "error", "message": f"Failed to schedule tapasya: {str(e)}"}

def remove_tapasya(user_id: str, schedule_name: str):
    """Remove a scheduled tapasya."""
    job_id = f"{user_id}_{schedule_name}"
    try:
        scheduler.remove_job(job_id)
        logger.info(f"Removed tapasya {schedule_name} for {user_id}")
        return {"status": "success", "message": f"Tapasya {schedule_name} removed"}
    except Exception:
        logger.warning(f"Tapasya {schedule_name} not found for {user_id}")
        return {"status": "error", "message": f"Tapasya {schedule_name} not found"}