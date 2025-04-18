import json
import os
from typing import Dict

class KarmicReflectionEngine:
    def __init__(self, karma_dir: str = "data/karmic_patterns"):
        self.karma_dir = karma_dir
        os.makedirs(karma_dir, exist_ok=True)

    def reflect(self, user_id: str, logs: list) -> Dict:
        patterns = {"emotions": [], "chakras": []}
        for log in logs:
            if log.get("emotion"):
                patterns["emotions"].append(log["emotion"])
            if log.get("chakra"):
                patterns["chakras"].append(log["chakra"])

        output_file = os.path.join(self.karma_dir, f"{user_id}.json")
        reflection = {
            "user_id": user_id,
            "recurring_emotions": list(set(patterns["emotions"])),
            "dominant_chakras": list(set(patterns["chakras"]))
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reflection, f, indent=2)
        return reflection
