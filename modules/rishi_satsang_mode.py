import json
import os
from datetime import datetime

class RishiSatsangMode:
    def __init__(self, satsang_dir: str = "data/satsang"):
        self.satsang_dir = satsang_dir
        os.makedirs(satsang_dir, exist_ok=True)

    def log_satsang(self, user_ids: list, intent: str):
        entry = {
            "user_ids": user_ids,
            "intent": intent,
            "timestamp": datetime.now().isoformat()
        }
        output_file = os.path.join(self.satsang_dir, "satsang_log.json")
        logs = []
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append(entry)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
