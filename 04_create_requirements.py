import os
import sys
from datetime import datetime

# Configuration
PROJECT_DIR = r"D:\mcpserver\mcp_server"
LOG_FILE = os.path.join(PROJECT_DIR, "update_log.txt")
TEMP_LOG = os.path.join(os.getenv("TEMP"), "update_log.txt")
REQUIREMENTS = [
    "fastapi",
    "uvicorn",
    "gradio",
    "librosa",
    "pydub",
    "sentence-transformers",
    "spacy",
    "PyPDF2",
    "speechrecognition",
    "pocketsphinx",
    "scikit-learn",
    "pyttsx3",
    "vosk",
    "pillow",
    "reportlab"
]

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

def main():
    # Initialize temporary log
    log_message("Starting requirements.txt creation", TEMP_LOG)

    # Verify project directory
    if not os.path.exists(PROJECT_DIR):
        log_message(f"ERROR: Directory {PROJECT_DIR} not found", TEMP_LOG)
        print(f"ERROR: Project directory {PROJECT_DIR} does not exist")
        sys.exit(1)

    # Check write permissions
    if not check_permissions(PROJECT_DIR):
        log_message(f"ERROR: No write permissions in {PROJECT_DIR}", TEMP_LOG)
        print("ERROR: Cannot write to project directory. Run as Administrator.")
        sys.exit(1)

    # Initialize project log
    log_message("Starting requirements.txt creation", LOG_FILE)

    # Create requirements.txt
    requirements_file = os.path.join(PROJECT_DIR, "requirements.txt")
    log_message("Creating requirements.txt", LOG_FILE)
    print("Creating requirements.txt...")
    try:
        with open(requirements_file, "w", encoding="utf-8") as f:
            for package in REQUIREMENTS:
                f.write(f"{package}\n")
        log_message("requirements.txt created", LOG_FILE)
    except Exception as e:
        log_message(f"ERROR: Failed to create requirements.txt: {e}", LOG_FILE)
        print("ERROR: Could not create requirements.txt")
        sys.exit(1)

    # Verify file creation
    if not os.path.exists(requirements_file):
        log_message("ERROR: requirements.txt was not created", LOG_FILE)
        print("ERROR: Could not verify requirements.txt creation")
        sys.exit(1)

    log_message("requirements.txt creation completed successfully", LOG_FILE)
    print("requirements.txt created successfully.")
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()