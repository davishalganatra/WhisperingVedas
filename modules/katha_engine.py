import json
import os
from typing import Dict

class KathaEngine:
    def __init__(self, katha_dir: str = "knowledge/katha_modules"):
        self.katha_dir = katha_dir
        os.makedirs(katha_dir, exist_ok=True)

    def generate_katha(self, theme: str) -> Dict:
        # Sample katha (real system would load from JSON)
        katha_files = [f for f in os.listdir(self.katha_dir) if f.endswith('.json')]
        for file in katha_files:
            with open(os.path.join(self.katha_dir, file), 'r', encoding='utf-8') as f:
                katha = json.load(f)
                if theme.lower() in katha.get("title", "").lower():
                    return katha
        
        # Fallback katha
        return {
            "title": f"Katha on {theme.capitalize()}",
            "source": "Vedic Wisdom",
            "shloka": "ॐ शान्ति शान्ति शान्तिः",
            "meaning": "Peace in body, mind, and soul",
            "story": f"A seeker once asked a Rishi about {theme}. The Rishi shared a tale of compassion and growth...",
            "takeaway": f"Embrace {theme} with an open heart."
        }