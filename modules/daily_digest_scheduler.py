import json
import os
from datetime import datetime

class DailyDigestScheduler:
    def __init__(self, log_dir: str = "data/usage_logs"):
        self.log_dir = log_dir

    def generate_digest(self, user_id: str) -> Dict:
        log_file = os.path.join(self.log_dir, f"{user_id}.json")
        if not os.path.exists(log_file):
            return {"message": "No logs found"}

        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        today = datetime.now().date().isoformat()
        today_logs = [log for log in logs if log["timestamp"].startswith(today)]

        return {
            "user_id": user_id,
            "date": today,
            "mantras_used": list(set(log.get("mantra") for log in today_logs if log.get("mantra"))),
            "chakra_trend": list(set(log.get("chakra") for log in today_logs if log.get("chakra"))),
            "suggestion": "Chant Om Shanti for peace tomorrow."
        }
