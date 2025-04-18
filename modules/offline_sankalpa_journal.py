import json
import os
from datetime import datetime

class OfflineSankalpaJournal:
    def __init__(self, journal_dir: str = "data/sankalpa"):
        self.journal_dir = journal_dir
        os.makedirs(journal_dir, exist_ok=True)

    def log_sankalpa(self, user_id: str, intent: str, reflection: str):
        journal_file = os.path.join(self.journal_dir, f"{user_id}.json")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "intent": intent,
            "reflection": reflection
        }
        journals = []
        if os.path.exists(journal_file):
            with open(journal_file, 'r', encoding='utf-8') as f:
                journals = json.load(f)
        journals.append(entry)
        with open(journal_file, 'w', encoding='utf-8') as f:
            json.dump(journals, f, indent=2)
