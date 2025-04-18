import os
import sys
import subprocess
from datetime import datetime
import socket

# Configuration
PROJECT_DIR = r"D:\mcpserver\mcp_server"
VENV_DIR = os.path.join(PROJECT_DIR, "venv")
VENV_PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe")
VENV_PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")
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

def check_internet():
    """Check internet connectivity by attempting to connect to PyPI."""
    try:
        socket.create_connection(("pypi.org", 443), timeout=5)
        return True
    except OSError:
        return False

def run_command(command, cwd):
    """Run a command and return the result, capturing both stdout and stderr."""
    try:
        result = subprocess.run(
            command, cwd=cwd, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        error_message = f"ERROR: {e.stderr.strip()}\nSTDOUT: {e.stdout.strip()}"
        return error_message

def main():
    # Initialize temporary log
    log_message("Starting dependency installation", TEMP_LOG)

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

    # Verify virtual environment
    if not os.path.exists(VENV_PIP):
        log_message(f"ERROR: Virtual environment pip not found at {VENV_PIP}", LOG_FILE)
        print("ERROR: Virtual environment is missing or corrupted. Re-run 02_setup_python.py.")
        sys.exit(1)

    # Initialize project log
    log_message("Starting dependency installation", LOG_FILE)

    # Check internet connectivity
    if not check_internet():
        log_message("ERROR: No internet connection detected", LOG_FILE)
        print("ERROR: No internet connection. Please check your network and try again.")
        sys.exit(1)
    log_message("Internet connection verified", LOG_FILE)

    # Upgrade pip
    log_message("Upgrading pip in virtual environment", LOG_FILE)
    print("Upgrading pip...")
    result = run_command([VENV_PYTHON, "-m", "pip", "install", "--upgrade", "pip"], PROJECT_DIR)
    if "ERROR" in result:
        log_message(f"ERROR: Failed to upgrade pip: {result}", LOG_FILE)
        print(f"ERROR: Could not upgrade pip: {result}")
        sys.exit(1)
    log_message("Pip upgraded successfully", LOG_FILE)

    # Install dependencies
    requirements_file = os.path.join(PROJECT_DIR, "requirements.txt")
    if not os.path.exists(requirements_file):
        log_message(f"ERROR: requirements.txt not found at {requirements_file}", LOG_FILE)
        print("ERROR: requirements.txt is missing. Re-run 04_create_requirements.py.")
        sys.exit(1)

    log_message("Installing Python dependencies", LOG_FILE)
    print("Installing dependencies... This may take a few minutes")
    result = run_command([VENV_PIP, "install", "-r", requirements_file], PROJECT_DIR)
    if "ERROR" in result:
        log_message(f"ERROR: Failed to install dependencies: {result}", LOG_FILE)
        print(
            f"ERROR: Dependency installation failed: {result}\n"
            "Check the following:\n"
            "- Internet connection\n"
            "- Run as Administrator if permission errors occur\n"
            "- Ensure Python 3.10.11 is compatible with packages\n"
            "- Try installing packages individually with: "
            f"{VENV_PIP} install <package>\n"
            f"- Check pip logs in {VENV_DIR}\\Scripts"
        )
        sys.exit(1)
    log_message("Dependencies installed", LOG_FILE)

    # Install SpaCy model
    log_message("Installing SpaCy model en_core_web_sm", LOG_FILE)
    result = run_command([VENV_PYTHON, "-m", "spacy", "download", "en_core_web_sm"], PROJECT_DIR)
    if "ERROR" in result:
        log_message(f"ERROR: Failed to install SpaCy model: {result}", LOG_FILE)
        print(
            f"ERROR: SpaCy model installation failed: {result}\n"
            "Try running manually: "
            f"{VENV_PYTHON} -m spacy download en_core_web_sm"
        )
        sys.exit(1)
    log_message("SpaCy model installed", LOG_FILE)

    log_message("Dependency installation completed successfully", LOG_FILE)
    print("Dependency installation completed successfully.")
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()