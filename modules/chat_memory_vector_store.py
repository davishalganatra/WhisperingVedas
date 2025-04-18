from sentence_transformers import SentenceTransformer
import os
import json

class ChatMemoryVectorStore:
    def __init__(self, whisper_dir: str = "data/whispers"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.whisper_dir = whisper_dir

    def vectorize_conversations(self, user_id: str):
        log_file = os.path.join(self.whisper_dir, f"{user_id}_log.json")
        if not os.path.exists(log_file):
            return

        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)

        texts = [log["query"] + " " + log["response"] for log in logs]
        embeddings = self.model.encode(texts)

        for log, emb in zip(logs, embeddings):
            log["embedding"] = emb.tolist()

        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2)
