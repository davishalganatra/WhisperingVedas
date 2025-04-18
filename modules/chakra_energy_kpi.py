import json
from typing import Dict, List
from .nlp_processor import NLPProcessor

class ChakraEnergyKPI:
    def __init__(self):
        self.nlp = NLPProcessor()
        self.chakra_map = {
            "root": {"frequency": 396.0, "emotions": ["security", "stability"]},
            "sacral": {"frequency": 417.0, "emotions": ["creativity", "passion"]},
            "solar_plexus": {"frequency": 528.0, "emotions": ["confidence", "power"]},
            "heart": {"frequency": 639.0, "emotions": ["love", "peace"]},
            "throat": {"frequency": 741.0, "emotions": ["truth", "expression"]},
            "third_eye": {"frequency": 852.0, "emotions": ["intuition", "clarity"]},
            "crown": {"frequency": 963.0, "emotions": ["spirituality", "enlightenment"]}
        }

    def analyze(self, text: str, mantra: str = None) -> Dict:
        entities = self.nlp.extract_entities(text)
        chakra = entities.get("chakra")[0] if entities.get("chakra") else "heart"
        emotion = entities.get("emotion")[0] if entities.get("emotion") else "peace"

        vib_score = self.chakra_map[chakra]["frequency"]
        if mantra and "om" in mantra.lower():
            vib_score += 10.0

        resonance = 0.8 if emotion in self.chakra_map[chakra]["emotions"] else 0.5

        return {
            "chakra": chakra,
            "vibrational_score": vib_score,
            "emotional_resonance": resonance,
            "emotion": emotion,
            "mantra": mantra or "Om Shanti"
        }
