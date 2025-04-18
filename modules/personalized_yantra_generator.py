from PIL import Image, ImageDraw, ImageFont
import os

class PersonalizedYantraGenerator:
    def __init__(self, output_dir: str = "data/yantras"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_yantra(self, user_id: str, mantra: str, affirmation: str) -> str:
        img = Image.new('RGB', (300, 300), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 50), mantra, fill='black')
        draw.text((50, 100), affirmation, fill='black')
        output_file = os.path.join(self.output_dir, f"{user_id}_yantra.png")
        img.save(output_file)
        return output_file
