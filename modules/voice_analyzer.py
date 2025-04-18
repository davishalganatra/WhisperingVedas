import librosa
import numpy as np
from typing import Dict

class VoiceAnalyzer:
    def __init__(self):
        self.chakra_freq = {
            "root": 396.0,
            "sacral": 417.0,
            "solar_plexus": 528.0,
            "heart": 639.0,
            "throat": 741.0,
            "third_eye": 852.0,
            "crown": 963.0
        }

    def analyze(self, audio_path: str) -> Dict:
        try:
            y, sr = librosa.load(audio_path)
            # Extract pitch (fundamental frequency)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch = np.mean([p for p in pitches[magnitudes > 0] if p > 0])
            # Extract intensity (RMS energy)
            intensity = np.mean(librosa.feature.rms(y=y))
            # Tone clarity (spectral centroid)
            centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # Suggest chakra based on closest frequency
            chakra = min(self.chakra_freq, key=lambda k: abs(self.chakra_freq[k] - pitch))
            
            return {
                "pitch": float(pitch),
                "intensity": float(intensity),
                "tone_clarity": float(centroid),
                "suggested_chakra": chakra,
                "kpi": {
                    "vibrational_score": self.chakra_freq[chakra],
                    "energy_level": intensity * 100
                }
            }
        except Exception as e:
            print(f"Error analyzing audio {audio_path}: {e}")
            return {"error": str(e)}