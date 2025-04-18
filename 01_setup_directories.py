import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Configuration
PROJECT_DIR = r"D:\mcpserver\mcp_server"
LOG_FILE = os.path.join(PROJECT_DIR, "update_log.txt")
TEMP_LOG = os.path.join(os.getenv("TEMP"), "update_log.txt")
DIRECTORIES = [
    "data/inputs", "sacred_texts/Dynamic_Inputs", "modules", "data/users",
    "data/voice_inputs", "data/usage_logs", "data/memory_snapshots",
    "data/karmic_patterns", "data/whispers", "data/knowledge",
    "data/knowledge_parsed", "knowledge/katha_modules", "data/yantras",
    "data/sankalpa", "data/sadhana_books"
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
    log_message("Starting directory setup", TEMP_LOG)

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
    log_message("Starting directory setup", LOG_FILE)
    log_message(f"Project directory: {PROJECT_DIR}", LOG_FILE)

    # Create directories
    error_flag = False
    for dir_path in DIRECTORIES:
        full_path = os.path.join(PROJECT_DIR, dir_path)
        try:
            if not os.path.exists(full_path):
                os.makedirs(full_path)
                log_message(f"Directory {dir_path} created", LOG_FILE)
            else:
                log_message(f"Directory {dir_path} already exists", LOG_FILE)
        except Exception as e:
            log_message(f"ERROR: Failed to create directory {dir_path}: {e}", LOG_FILE)
            print(f"ERROR: Could not create directory {dir_path}")
            error_flag = True

    if error_flag:
        log_message("ERROR: One or more directories could not be created", LOG_FILE)
        print("ERROR: Could not create required directories")
        sys.exit(1)

    # Create sample input files
    sample_files = {
        "data/inputs/sample.txt": "This text discusses heart chakra healing and peace through chanting Om Shanti.",
        # Note: sample.pdf and sample.wav are not created as they require valid PDF/WAV formats.
        # To test PDF/audio parsing, place valid .pdf and .wav files in data/inputs manually.
        "knowledge/katha_modules/forgiveness.json": json.dumps({
            "title": "Katha on Forgiveness",
            "source": "Mahabharata",
            "shloka": "क्षमा धर्मः सनातनः",
            "meaning": "Forgiveness is eternal dharma",
            "story": "A sage forgave a wrongdoer, teaching compassion...",
            "takeaway": "Forgive to free your soul."
        }, indent=2),
        "data/knowledge/peace.md": "# Vedic Wisdom on Peace\nChanting Om Shanti aligns the heart chakra, fostering inner peace and devotion."
    }

    for file_path, content in sample_files.items():
        full_path = os.path.join(PROJECT_DIR, file_path)
        try:
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            log_message(f"Created {file_path}", LOG_FILE)
        except Exception as e:
            log_message(f"ERROR: Failed to create {file_path}: {e}", LOG_FILE)
            print(f"ERROR: Could not create {file_path}")
            sys.exit(1)

    log_message("Directory setup completed", LOG_FILE)
    print("Directory setup completed successfully.")
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()