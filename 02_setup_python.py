import os
import sys
import subprocess
from datetime import datetime
import venv

# Configuration
PROJECT_DIR = r"D:\mcpserver\mcp_server"
VENV_DIR = os.path.join(PROJECT_DIR, "venv")
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

def get_python_version():
    """Get the Python version."""
    try:
        result = subprocess.run(["python", "--version"], capture_output=True, text=True, check=True)
        return result.stdout.strip().split()[1]
    except Exception as e:
        return None

def main():
    # Initialize temporary log
    log_message("Starting Python setup", TEMP_LOG)

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
    log_message("Starting Python setup", LOG_FILE)

    # Verify Python installation
    python_version = get_python_version()
    if not python_version:
        log_message("ERROR: Python is not installed or not in PATH", LOG_FILE)
        print("ERROR: Python is required. Install Python 3.8+ from https://www.python.org")
        sys.exit(1)
    log_message(f"Python version: {python_version}", LOG_FILE)

    # Verify Python version (3.8+)
    try:
        major, minor = map(int, python_version.split(".")[:2])
        if major < 3 or (major == 3 and minor < 8):
            log_message(f"ERROR: Python version {python_version} is too old. Requires 3.8+", LOG_FILE)
            print("ERROR: Python 3.8+ is required. Install from https://www.python.org")
            sys.exit(1)
    except ValueError as e:
        log_message(f"ERROR: Failed to parse Python version: {e}", LOG_FILE)
        print("ERROR: Could not determine Python version")
        sys.exit(1)

    # Create virtual environment
    if not os.path.exists(VENV_DIR):
        log_message(f"Creating virtual environment at {VENV_DIR}", LOG_FILE)
        print("Creating Python virtual environment...")
        try:
            venv.create(VENV_DIR, with_pip=True)
            log_message("Virtual environment created", LOG_FILE)
        except Exception as e:
            log_message(f"ERROR: Failed to create virtual environment: {e}", LOG_FILE)
            print("ERROR: Virtual environment creation failed")
            sys.exit(1)
    else:
        log_message(f"Virtual environment already exists at {VENV_DIR}", LOG_FILE)

    # Verify pip in virtual environment
    venv_pip = os.path.join(VENV_DIR, "Scripts", "pip.exe")
    if not os.path.exists(venv_pip):
        log_message(f"ERROR: pip not found in virtual environment at {venv_pip}", LOG_FILE)
        print("ERROR: Virtual environment is corrupted")
        sys.exit(1)

    log_message("Python setup completed successfully", LOG_FILE)
    print("Python setup completed successfully.")
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()