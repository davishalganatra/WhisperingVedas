from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

class SadhanaBookCreator:
    def __init__(self, output_dir: str = "data/sadhana_books"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_book(self, user_id: str, logs: list) -> str:
        output_file = os.path.join(self.output_dir, f"{user_id}_sadhana.pdf")
        c = canvas.Canvas(output_file, pagesize=letter)
        c.drawString(100, 750, f"Sadhana Journey for {user_id}")
        y = 700
        for log in logs[:10]:
            c.drawString(100, y, f"{log.get('timestamp')}: {log.get('mantra')}")
            y -= 20
        c.save()
        return output_file
