from sentence_transformers import SentenceTransformer
import os
import json

class OfflineEmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", knowledge_dir: str = "data/knowledge"):
        self.model = SentenceTransformer(model_name)
        self.knowledge_dir = knowledge_dir

    def embed_files(self):
        documents = []
        for file in os.listdir(self.knowledge_dir):
            if file.endswith(('.md', '.txt')):
                with open(os.path.join(self.knowledge_dir, file), 'r', encoding='utf-8') as f:
                    documents.append({"file": file, "content": f.read()})

        embeddings = self.model.encode([doc["content"] for doc in documents])
        for doc, emb in zip(documents, embeddings):
            doc["embedding"] = emb.tolist()

        output_file = os.path.join(self.knowledge_dir, "embeddings.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2)
