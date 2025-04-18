import uuid
import json
import os
from typing import Dict, List
from .nlp_processor import NLPProcessor

class TemplateGenerator:
    def __init__(self, output_dir: str):
        self.nlp = NLPProcessor()
        self.output_dir = output_dir
        self.default_values = {
            "chakra": "heart",
            "mantra": "Om Shanti",
            "frequency": 528.0,
            "tone": "F",
            "repetitions": 54,
            "emotion": ["peace"],
            "goal": ["healing"]
        }

    def generate_template(self, text: str, source: str, file_name: str) -> Dict:
        entities = self.nlp.extract_entities(text)
        template = {
            "id": str(uuid.uuid4()),
            "source": source,
            "text": text[:500],
            "translation": text[:500],
            "emotion": entities.get("emotion") or self.default_values["emotion"],
            "goal": entities.get("goal") or self.default_values["goal"],
            "chakra": entities.get("chakra")[0] if entities.get("chakra") else self.default_values["chakra"],
            "mantra": entities.get("mantra")[0] if entities.get("mantra") else self.default_values["mantra"],
            "frequency": self.default_values["frequency"],
            "tone": self.default_values["tone"],
            "repetitions": self.default_values["repetitions"],
            "context": file_name
        }
        self.nlp.train_classifier([text], [template["chakra"]])
        return template

    def process_texts(self, texts: List[Dict]):
        os.makedirs(self.output_dir, exist_ok=True)
        for item in texts:
            template = self.generate_template(item["text"], f"{item['source_type'].upper()}_Source", item["file"])
            output_file = os.path.join(self.output_dir, f"{item['file'].split('.')[0]}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({"verses": [template]}, f, ensure_ascii=False, indent=2)
