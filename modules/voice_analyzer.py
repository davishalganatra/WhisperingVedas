import librosa
import numpy as np
from typing import Dict
import speech_recognition as sr
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class VoiceAnalyzer:
    def __init__(self, use_transcription: bool = False):
        self.chakra_freq = {
            "root": 396.0,
            "sacral": 417.0,
            "solar_plexus": 528.0,
            "heart": 639.0,
            "throat": 741.0,
            "third_eye": 852.0,
            "crown": 963.0
        }
        self.use_transcription = use_transcription
        self.recognizer = sr.Recognizer() if use_transcription else None

    def process_voice(self, audio_path: str) -> Dict:
        """Process a WAV audio file and return vibration KPIs, tempo, and suggested chakra."""
        path = Path(audio_path)

        if not path.suffix.lower() == ".wav":
            logger.error(f"Invalid audio format for {audio_path}. Only WAV files are supported.")
            return {"error": "Invalid audio format. Please upload a WAV file."}

        try:
            # Load audio with librosa
            y, sr_rate = librosa.load(str(path), sr=None)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr_rate)

            # Compute average pitch from non-zero values
            non_zero_pitches = pitches[magnitudes > 0]
            pitch = float(np.mean(non_zero_pitches[non_zero_pitches > 0])) if non_zero_pitches.any() else 0.0

            intensity = float(np.mean(librosa.feature.rms(y=y)))
            centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr_rate)))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr_rate)

            # Find closest chakra frequency
            chakra = min(self.chakra_freq, key=lambda k: abs(self.chakra_freq[k] - pitch))

            # Optional transcription
            transcription = "N/A"
            if self.use_transcription and self.recognizer:
                try:
                    with sr.AudioFile(str(path)) as source:
                        audio = self.recognizer.record(source)
                        transcription = self.recognizer.recognize_sphinx(audio)
                    logger.info(f"Transcription for {audio_path}: {transcription}")
                except Exception as e:
                    logger.warning(f"Transcription failed for {audio_path}: {e}")
                    transcription = f"Transcription failed: {str(e)}"

            result = {
                "energy_level": round(intensity * 100, 2),
                "clarity_score": round(centroid, 2),
                "tempo": round(tempo, 2),
                "pitch": round(pitch, 2),
                "message": f"Transcription: {transcription}",
                "suggested_chakra": chakra,
                "kpi": {
                    "vibrational_score": self.chakra_freq[chakra],
                    "energy_level": round(intensity * 100, 2)
                }
            }

            logger.info(f"Voice analysis result for {audio_path}: {result}")
            return result

        except Exception as e:
            logger.exception(f"Failed to analyze {audio_path}")
            return {"error": f"Error analyzing audio: {str(e)}"}
