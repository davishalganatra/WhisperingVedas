from typing import Dict

class PersonalMantraComposer:
    def compose(self, user_id: str, context: Dict) -> str:
        chakra = context.get("chakra", "heart")
        emotion = context.get("emotion", "peace")
        return f"Om {chakra.capitalize()} {emotion.capitalize()} Swaha"
