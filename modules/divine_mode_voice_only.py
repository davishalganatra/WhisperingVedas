from vosk import Model, KaldiRecognizer
import pyaudio
import json
from .rishi_voice_avatar import RishiVoiceAvatar

class DivineModeVoiceOnly:
    def __init__(self):
        self.model = Model("vosk-model-small-en-us-0.15")
        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.voice = RishiVoiceAvatar()

    def listen_and_respond(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()
        while True:
            data = stream.read(4000)
            if self.recognizer.AcceptWaveform(data):
                result = self.recognizer.Result()
                query = json.loads(result).get("text", "")
                if query:
                    response = f"Received: {query}. Chant Om Shanti."
                    self.voice.speak(response)
        stream.stop_stream()
        stream.close()
        p.terminate()
