import json
import os
from typing import Dict

class MemoryReflector:
    def __init__(self, log_dir: str = "data/usage_logs", memory_dir: str = "data/memory_snapshots"):
        self.log_dir = log_dir
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)

    def summarize(self, user_id: str) -> Dict:
        log_file = os.path.join(self.log_dir, f"{user_id}.json")
        if not os.path.exists(log_file):
            return {"summary": "No interactions found"}

        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        chakras = [log.get("chakra") for log in logs if log.get("chakra")]
        mantras = [log.get("mantra") for log in logs if log.get("mantra")]
        emotions = [log.get("emotion") for log in logs if log.get("emotion")]

        summary = {
            "user_id": user_id,
            "chakra_evolution": list(set(chakras)),
            "mantra_history": list(set(mantras)),
            "emotion_trends": list(set(emotions))
        }

        output_file = os.path.join(self.memory_dir, f"{user_id}_summary.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        return summary
