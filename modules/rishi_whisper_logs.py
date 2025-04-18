import json
import os
from datetime import datetime

class RishiWhisperLogs:
    def __init__(self, whisper_dir: str = "data/whispers"):
        self.whisper_dir = whisper_dir
        os.makedirs(whisper_dir, exist_ok=True)

    def log_conversation(self, user_id: str, query: str, response: str):
        log_file = os.path.join(self.whisper_dir, f"{user_id}_log.json")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response
        }
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        logs.append(entry)
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
