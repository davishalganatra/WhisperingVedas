import os
import PyPDF2
import speech_recognition as sr
from pathlib import Path

def parse_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error parsing text file {file_path}: {e}")
        return ""

def parse_pdf_file(file_path: str) -> str:
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        print(f"Error parsing PDF {file_path}: {e}")
        return ""

def parse_audio_file(file_path: str) -> str:
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_sphinx(audio)
            return text
    except Exception as e:
        print(f"Error transcribing audio {file_path}: {e}")
        return ""

def parse_input_files(input_dir: str) -> list:
    texts = []
    input_dir = Path(input_dir)
    for file_path in input_dir.glob("*"):
        if file_path.suffix.lower() in ('.txt', '.pdf', '.wav'):
            if file_path.suffix == '.txt':
                text = parse_text_file(file_path)
            elif file_path.suffix == '.pdf':
                text = parse_pdf_file(file_path)
            else:
                text = parse_audio_file(file_path)
            if text:
                texts.append({"file": file_path.name, "text": text, "source_type": file_path.suffix[1:]})
    return texts
