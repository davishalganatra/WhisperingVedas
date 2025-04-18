import json
import os
from datetime import datetime
from typing import Dict

class UsageTracker:
    def __init__(self, log_dir: str = "data/usage_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def log_interaction(self, user_id: str, interaction: Dict):
        log_file = os.path.join(self.log_dir, f"{user_id}.json")
        interaction["timestamp"] = datetime.now().isoformat()
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append(interaction)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
