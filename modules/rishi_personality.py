from typing import Dict

class RishiPersonality:
    def __init__(self, tone: str = "gentle"):
        self.tone = tone

    def respond(self, query: str, context: Dict) -> str:
        if self.tone == "gentle":
            return f"Dear seeker, your heart seeks {context.get('goal', 'peace')}. Chant {context.get('mantra', 'Om Shanti')} with love."
        elif self.tone == "strict":
            return f"Seeker, focus on {context.get('chakra', 'heart')} chakra. Chant {context.get('mantra', 'Om Shanti')} daily."
        else:
            return f"O child of the Vedas, align with {context.get('chakra', 'heart')} through {context.get('mantra', 'Om Shanti')}."
