import os
import json
import time

def log_event(user_id: str, mantra: str, vibration_level: float, emotional_state: str):
    """Log a user event to data/user_logs/{user_id}_event_log.json."""
    event_entry = {
        "user_id": user_id,
        "mantra": mantra,
        "vibration_level": vibration_level,
        "emotional_state": emotional_state,
        "timestamp": int(time.time())
    }
    
    os.makedirs("data/user_logs", exist_ok=True)
    log_file = f"data/user_logs/{user_id}_event_log.json"
    
    events = []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            events = json.load(f)
    except:
        pass
    
    events.append(event_entry)
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)
        return {"status": "success", "message": "Event logged"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to log event: {str(e)}"}
"""

**Instructions**:
1. Open `D:\mcp server\mcp_server\modules\event_logging.py` in a text editor (e.g., Notepad, VS Code).
2. Replace its contents with the code above, ensuring no extra lines or markers (e.g., """python or """) are included.
3. Save the file.

### Step 2: Verify Module Files
The `main.py` file imports several modules, and previous errors (e.g., missing `apscheduler`, `log_event`) suggest potential issues with other modules. Let’s confirm all required files are present and correct:

**Expected Files in `D:\mcp server\mcp_server\modules\`**:
"""
__init__.py
akashic_logger.py
chant_generation.py
event_logging.py
mantra_suggestion.py
tap_sadhana.py
tap_sadhana_scheduler.py
user_profiles.py
voice_feedback.py
"""

**Action**:
1. Check files:
   """cmd
   dir D:\mcp server\mcp_server\modules\
   """
2. If any are missing, ensure they match the versions provided earlier:
   - `tap_sadhana.py` and `voice_feedback.py` were provided in the previous response.
   - Others (`akashic_logger.py`, `chant_generation.py`, `mantra_suggestion.py`, `tap_sadhana_scheduler.py`, `user_profiles.py`) were provided initially.
3. If unsure, I can re-share any module’s code—just let me know which one.

### Step 3: Test FastAPI Server
With `event_logging.py` fixed:

1. **Run FastAPI Server**:
   """cmd
   cd D:\mcp server\mcp_server
   venv\Scripts\activate
   python main.py
   """
   **Expected Output**:
   """
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   """

2. **Test Endpoint**:
   Open a browser and visit:
   """
   http://localhost:8000/v1
   """
   **Expected Response** (JSON):
   """json
   {
     "message": "Welcome to MCP Server - Whispering Vedas API.",
     "usage": "..."
   }
   """

3. **If Errors Persist**:
   - Share the new error message.
   - Check for port conflicts:
     """cmd
     netstat -a -n -o | find "8000"
     """
     Kill conflicting process:
     """cmd
     taskkill /PID <PID> /F
     """

### Step 4: Test Gradio UI
With the FastAPI server running:

1. **Run Gradio UI** (in a new terminal):
   """cmd
   cd D:\mcp server\mcp_server
   venv\Scripts\activate
   python gradio_ui_dynamic.py
   """

2. **Test Queries**:
   - Open `http://localhost:7860`.
   - Try:
     - “I feel heavy, help my heart chakra”
     - “Suggest a mantra for peace”
     - “Start my daily tapasya”
   - **Expected Response** (e.g., for “I feel heavy, help my heart chakra”):
     """
     Mantra: Om Namah Shivaya
     Frequency: 432.0 Hz
     Tone: G
     Repetitions: 108
     Shloka: Default mantra
     Translation: Universal chant
     """

3. **Check Logs**:
   - Terminal logs from `gradio_ui_dynamic.py`:
     """
     2025-04-16 10:00:00,000 - INFO - Querying http://localhost:8000/suggest_mantra with payload: ...
     """
   - Files:
     - `D:\mcp server\mcp_server\data\user_logs\user123_log.json` (chant logs)
     - `D:\mcp server\mcp_server\data\user_logs\user123_event_log.json` (event logs)

### Step 5: Update `run_all.bat`
To ensure both servers start reliably, use the updated `run_all.bat` from the previous response, which includes dependency checks.

<xaiArtifact artifact_id="698d9713-2574-45f6-b94a-2fe9b102a715" artifact_version_id="6098db8f-317e-4414-a608-ab1cbab8ca17" title="run_all.bat" contentType="text/bat">
"""batch
@echo off
ECHO Starting MCP Server and Gradio UI for Whispering Vedas...

:: Set paths
SET "PROJECT_DIR=D:\mcp server\mcp_server"

:: Navigate to project directory
cd /d "%PROJECT_DIR%"
IF ERRORLEVEL 1 (
    ECHO Failed to navigate to %PROJECT_DIR%. Ensure it exists.
    pause
    exit /b 1
)

:: Activate virtual environment
call venv\Scripts\activate
IF ERRORLEVEL 1 (
    ECHO Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Verify dependencies
ECHO Verifying dependencies...
pip show fastapi uvicorn pydantic httpx apscheduler SpeechRecognition gradio PyAudio
IF ERRORLEVEL 1 (
    ECHO Installing missing dependencies...
    pip install fastapi==0.115.0 uvicorn==0.30.6 pydantic==2.9.2 httpx==0.27.2 apscheduler==3.10.4 SpeechRecognition==3.10.4 gradio==4.44.0 PyAudio
    IF ERRORLEVEL 1 (
        ECHO Failed to install dependencies.
        pause
        exit /b 1
    )
)

:: Start FastAPI server in a new terminal
ECHO Starting FastAPI server...
start cmd /k "python main.py"

:: Wait a few seconds to ensure FastAPI starts
timeout /t 5

:: Start Gradio UI in a new terminal
ECHO Starting Gradio UI...
start cmd /k "python gradio_ui_dynamic.py"

ECHO All services started. Press any key to exit...
pause
"""

Run:
"""cmd
run_all.bat
"""

### Step 6: Additional Checks
- **Dependencies**:
  Confirm all are installed:
  """cmd
  pip list
  """
  Expected:
  """
  fastapi            0.115.0
  uvicorn            0.30.6
  pydantic           2.9.2
  httpx              0.27.2
  apscheduler        3.10.4
  SpeechRecognition  3.10.4
  gradio             4orc
  PyAudio            (version varies)
  """

- **Vedic Knowledge JSON**:
  If queries return only the default mantra (`Om Namah Shivaya`):
  """cmd
  python merge_json.py
  """
  Check output for “Skipped: File not found”. Ensure `D:\sacred_texts\` has JSONs (e.g., `SrimadBhagvadGita\bhagavad_gita_chapter_1.json`). Share output if issues persist.

- **AnythingLLM Integration**:
  Ensure:
  - **Settings > LLM Preference > Custom LLM Provider**
  - **API Base URL**: `http://localhost:8000/v1`
  - **API Key**: `mcp-secret-key`
  - **Model Name**: `mcp-mantra`

### If Issues Persist
- Share:
  - Terminal output when running `main.py`.
  - Terminal output when running `gradio_ui_dynamic.py`.
  - Output of `python merge_json.py`.
  - Contents of `D:\mcp server\mcp_server\modules\`:
    """cmd
    dir D:\mcp server\mcp_server\modules\
    """
  - Contents of `D:\mcp server\mcp_server\data\user_logs\`.
  - Result of `http://localhost:8000/v1` in a browser.

### Summary
The `SyntaxError` in `event_logging.py` was caused by invalid content (likely Markdown markers). The corrected `event_logging.py` defines `log_event` properly. Run `main.py` to start the FastAPI server, then `gradio_ui_dynamic.py` to test the Gradio UI. Use `run_all.bat` for convenience. If new errors appear, share the requested outputs for further debugging. Let me know the results!