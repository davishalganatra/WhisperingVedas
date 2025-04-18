import httpx
import os
import time
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

async def generate_chant(mantra: str, frequency: float, tone: str) -> str:
    """Generate a chant and save as text (or audio via external service)."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "http://localhost:8081/generate",
                json={"mantra": mantra, "frequency": frequency, "tone": tone}
            )
            response.raise_for_status()
            audio_path = response.json().get("audio_path", "data/chant_output.wav")
            logger.info(f"Generated chant: {audio_path}")
            return audio_path
        except httpx.HTTPError:
            audio_path = f"data/chants/{mantra.replace(' ', '_')}_{int(time.time())}.txt"
            os.makedirs("data/chants", exist_ok=True)
            with open(audio_path, "w", encoding="utf-8") as f:
                f.write(f"Mantra: {mantra}, Frequency: {frequency} Hz, Tone: {tone}")
            logger.info(f"Generated fallback chant: {audio_path}")
            return audio_path

async def play_chant(audio_path: str) -> str:
    """Play a chant file using system default player."""
    if not os.path.exists(audio_path):
        logger.error(f"Audio file {audio_path} not found")
        return f"Error: {audio_path} not found"
    try:
        subprocess.run(['start', '', audio_path], shell=True, check=True)
        logger.info(f"Playing chant: {audio_path}")
        return f"Playing {audio_path}"
    except Exception as e:
        logger.error(f"Error playing {audio_path}: {str(e)}")
        return f"Error playing {audio_path}: {str(e)}"