import json
from datetime import datetime

class VedicCalendarAwareness:
    def __init__(self, calendar_file: str = "data/knowledge/panchang.json"):
        self.calendar_file = calendar_file

    def get_tithi(self) -> str:
        today = datetime.now().date().isoformat()
        return f"Purnima on {today}" if "15" in today else "Ekadashi"
