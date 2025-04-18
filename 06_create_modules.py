import os
import sys
import json
from datetime import datetime

# Configuration
PROJECT_DIR = r"D:\mcpserver\mcp_server"
MODULES_DIR = os.path.join(PROJECT_DIR, "modules")
LOG_FILE = os.path.join(PROJECT_DIR, "update_log.txt")
TEMP_LOG = os.path.join(os.getenv("TEMP"), "update_log.txt")

def log_message(message, file_path=TEMP_LOG):
    """Log a message to the specified file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"Failed to write to log {file_path}: {e}")

def check_permissions(directory):
    """Check if the directory is writable."""
    test_file = os.path.join(directory, "test_write.txt")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except Exception:
        return False

def write_module(file_path, content):
    """Write a module file and log the result."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        log_message(f"Created {os.path.relpath(file_path, PROJECT_DIR)}", LOG_FILE)
    except Exception as e:
        log_message(f"ERROR: Failed to create {os.path.relpath(file_path, PROJECT_DIR)}: {e}", LOG_FILE)
        print(f"ERROR: Could not create {os.path.relpath(file_path, PROJECT_DIR)}")
        sys.exit(1)

def main():
    # Initialize temporary log
    log_message("Starting module creation", TEMP_LOG)

    # Verify project and modules directory
    if not os.path.exists(PROJECT_DIR):
        log_message(f"ERROR: Directory {PROJECT_DIR} not found", TEMP_LOG)
        print(f"ERROR: Project directory {PROJECT_DIR} does not exist")
        sys.exit(1)

    if not os.path.exists(MODULES_DIR):
        log_message(f"ERROR: Modules directory {MODULES_DIR} not found", TEMP_LOG)
        print(f"ERROR: Modules directory does not exist")
        sys.exit(1)

    # Check write permissions
    if not check_permissions(MODULES_DIR):
        log_message(f"ERROR: No write permissions in {MODULES_DIR}", TEMP_LOG)
        print("ERROR: Cannot write to modules directory. Run as Administrator.")
        sys.exit(1)

    # Initialize project log
    log_message("Starting module creation", LOG_FILE)

    # Define module contents (only new modules shown; others remain unchanged)
    modules = {
        # ... (previous modules: input_parser.py through sadhana_book_creator.py, katha_engine.py, __init__.py remain unchanged)
        "sacred_text_processor.py": """import os
import json
from pathlib import Path

class SacredTextProcessor:
    def __init__(self, input_dir: str = "sacred_texts/Dynamic_Inputs", output_dir: str = "data/knowledge"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def process_texts(self):
        texts = []
        input_dir = Path(self.input_dir)
        for file_path in input_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                texts.append({
                    "file": file_path.name,
                    "text": content,
                    "source_type": "text"
                })
                output_file = os.path.join(self.output_dir, f"{file_path.stem}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({"text": content, "source": file_path.name}, f, indent=2)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        return texts
""",
        "mantra_frequency_tuner.py": """from typing import Dict
from .chakra_energy_kpi import ChakraEnergyKPI

class MantraFrequencyTuner:
    def __init__(self):
        self.chakra_kpi = ChakraEnergyKPI()

    def tune_frequency(self, text: str, emotion: str = "peace") -> Dict:
        kpi = self.chakra_kpi.analyze(text)
        chakra = kpi["chakra"]
        base_freq = kpi["vibrational_score"]
        adjustment = 10.0 if emotion in ["peace", "love"] else 5.0
        return {
            "chakra": chakra,
            "adjusted_frequency": base_freq + adjustment,
            "mantra": kpi["mantra"]
        }
""",
        "astrological_context_engine.py": """import json
from datetime import datetime

class AstrologicalContextEngine:
    def __init__(self, calendar_file: str = "data/knowledge/panchang.json"):
        self.calendar_file = calendar_file

    def get_context(self, date: str = None) -> Dict:
        date = date or datetime.now().date().isoformat()
        try:
            with open(self.calendar_file, 'r', encoding='utf-8') as f:
                panchang = json.load(f)
            return panchang.get(date, {
                "tithi": "Purnima",
                "nakshatra": "Ashwini",
                "recommendation": "Chant Om Shanti for balance"
            })
        except FileNotFoundError:
            return {
                "tithi": "Purnima",
                "nakshatra": "Ashwini",
                "recommendation": "Chant Om Shanti for balance"
            }
""",
        "breathing_pattern_analyzer.py": """import librosa
import numpy as np
from typing import Dict

class BreathingPatternAnalyzer:
    def __init__(self):
        self.threshold = 0.1  # Intensity threshold for breath detection

    def analyze(self, audio_path: str) -> Dict:
        try:
            y, sr = librosa.load(audio_path)
            rms = librosa.feature.rms(y=y)
            breaths = np.where(rms[0] > self.threshold)[0]
            breath_rate = len(breaths) / (len(y) / sr) * 60  # Breaths per minute
            return {
                "breath_rate": float(breath_rate),
                "suggested_pace": "slow" if breath_rate < 12 else "normal"
            }
        except Exception as e:
            print(f"Error analyzing audio {audio_path}: {e}")
            return {"error": str(e)}
""",
        # ... (main.py remains unchanged)
    }

    # Create modules in modules directory
    for module_name, content in modules.items():
        write_module(os.path.join(MODULES_DIR, module_name), content)

    # Create main.py in project root
    main_content = """import os
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict
from modules.input_parser import parse_input_files
from modules.template_generator import TemplateGenerator
from modules.chakra_energy_kpi import ChakraEnergyKPI
from modules.auth import Auth
from modules.usage_tracker import UsageTracker
from modules.gradio_ui_dynamic import GradioUIDynamic

app = FastAPI()
auth = Auth()
tracker = UsageTracker()
chakra_kpi = ChakraEnergyKPI()
template_gen = TemplateGenerator("data/knowledge_parsed")

class UserRequest(BaseModel):
    user_id: str
    token: str
    query: str

@app.post("/process")
async def process_input(request: UserRequest):
    if not auth.verify_token(request.user_id, request.token):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    texts = parse_input_files("data/inputs")
    template_gen.process_texts(texts)
    kpi = chakra_kpi.analyze(request.query)
    
    tracker.log_interaction(request.user_id, {
        "query": request.query,
        "chakra": kpi["chakra"],
        "mantra": kpi["mantra"],
        "emotion": kpi["emotion"]
    })
    
    return kpi

@app.get("/launch_ui")
async def launch_ui():
    ui = GradioUIDynamic()
    ui.launch_ui()
    return {"message": "Gradio UI launched on localhost:7860"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
"""
    write_module(os.path.join(PROJECT_DIR, "main.py"), main_content)

    log_message("Module creation completed successfully", LOG_FILE)
    print("Module creation completed successfully.")
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()