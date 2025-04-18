import json
import os
from typing import Dict

class KathaEngine:
    def __init__(self, katha_dir: str = "knowledge/katha_modules"):
        self.katha_dir = katha_dir

    def generate_katha(self, theme: str) -> Dict:
        katha_file = os.path.join(self.katha_dir, f"{theme.lower()}.json")
        if os.path.exists(katha_file):
            with open(katha_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            "title": f"Katha on {theme.capitalize()}",
            "source": "Generic",
            "shloka": "ॐ शान्तिः शान्तिः शान्तिः",
            "meaning": "Peace, peace, peace",
            "story": f"A tale about {theme.lower()}...",
            "takeaway": f"Embrace {theme.lower()} in your life."
        }
