import os
import json
from typing import List

class MarkdownKnowledgeBase:
    def __init__(self, md_dir: str = "data/knowledge", output_dir: str = "data/knowledge_parsed"):
        self.md_dir = md_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def parse_md_files(self) -> List[Dict]:
        qa_pairs = []
        for file in os.listdir(self.md_dir):
            if file.endswith('.md'):
                with open(os.path.join(self.md_dir, file), 'r', encoding='utf-8') as f:
                    content = f.read()
                    qa_pairs.append({
                        "question": content[:50],
                        "answer": content,
                        "source": file
                    })
        output_file = os.path.join(self.output_dir, "qa_pairs.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, indent=2)
        return qa_pairs
