import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from typing import Dict, List

class NLPProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.training_data = []
        self.training_labels = []

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = self.nlp(text.lower())
        entities = {
            "chakra": [],
            "mantra": [],
            "emotion": [],
            "goal": []
        }
        chakra_keywords = {
            "root": ["root", "muladhara"],
            "sacral": ["sacral", "svadhisthana"],
            "solar_plexus": ["solar plexus", "manipura"],
            "heart": ["heart", "anahata"],
            "throat": ["throat", "vishuddha"],
            "third_eye": ["third eye", "ajna"],
            "crown": ["crown", "sahasrara"]
        }
        emotion_keywords = ["peace", "devotion", "courage", "detachment", "focus"]
        goal_keywords = ["clarity", "healing", "spiritual growth", "self-realization"]

        for token in doc:
            word = token.text
            for chakra, keywords in chakra_keywords.items():
                if any(kw in word for kw in keywords):
                    entities["chakra"].append(chakra)
            if "om" in word or "namah" in word:
                entities["mantra"].append(word)
            if word in emotion_keywords:
                entities["emotion"].append(word)
            if word in goal_keywords:
                entities["goal"].append(word)

        return entities

    def train_classifier(self, texts: List[str], labels: List[str]):
        if not texts or not labels:
            return
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        self.classifier.fit(X, y)
        self.training_data.extend(texts)
        self.training_labels.extend(labels)

    def predict_field(self, text: str, field: str) -> str:
        if not self.training_data:
            return ""
        X = self.vectorizer.transform([text])
        return self.classifier.predict(X)[0]
