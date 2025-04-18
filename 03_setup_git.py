import os
import sys
import subprocess
from datetime import datetime

# Configuration
PROJECT_DIR = r"D:\mcpserver\mcp_server"
REPO_URL = "https://github.com/davishalganatra/WhisperingVedas.git"
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

def run_git_command(args, cwd):
    """Run a Git command and return the result."""
    try:
        result = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.stderr.strip()}"

def main():
    # Initialize temporary log
    log_message("Starting Git setup", TEMP_LOG)

    # Verify project directory
    if not os.path.exists(PROJECT_DIR):
        log_message(f"ERROR: Directory {PROJECT_DIR} not found", TEMP_LOG)
        print(f"ERROR: Project directory {PROJECT_DIR} does not exist")
        sys.exit(1)

    # Initialize project log
    log_message("Starting Git setup", LOG_FILE)

    # Verify Git installation
    result = run_git_command(["--version"], PROJECT_DIR)
    if "ERROR" in result:
        log_message("ERROR: Git is not installed", LOG_FILE)
        print("ERROR: Git is required. Install from https://git-scm.com")
        sys.exit(1)
    log_message("Git installed", LOG_FILE)

    # Change to project directory
    os.chdir(PROJECT_DIR)

    # Initialize Git repository
    if not os.path.exists(os.path.join(PROJECT_DIR, ".git")):
        log_message("Initializing Git repository", LOG_FILE)
        result = run_git_command(["init"], PROJECT_DIR)
        if "ERROR" in result:
            log_message(f"ERROR: Failed to run git init: {result}", LOG_FILE)
            print("ERROR: Could not initialize Git repository")
            sys.exit(1)
        log_message("Git repository initialized", LOG_FILE)

        # Set remote origin
        result = run_git_command(["remote", "add", "origin", REPO_URL], PROJECT_DIR)
        if "ERROR" in result:
            log_message(f"ERROR: Failed to set remote origin: {result}", LOG_FILE)
            print("ERROR: Could not set Git remote")
            sys.exit(1)
        log_message(f"Set remote origin to {REPO_URL}", LOG_FILE)
    else:
        log_message("Git repository already exists", LOG_FILE)

    # Verify or set remote URL
    result = run_git_command(["remote", "get-url", "origin"], PROJECT_DIR)
    if "ERROR" in result or not result:
        log_message(f"WARNING: No remote URL found, setting to {REPO_URL}", LOG_FILE)
        result = run_git_command(["remote", "add", "origin", REPO_URL], PROJECT_DIR)
        if "ERROR" in result:
            log_message(f"ERROR: Failed to set remote URL: {result}", LOG_FILE)
            print("ERROR: Could not set Git remote URL")
            sys.exit(1)
    elif result != REPO_URL:
        log_message(f"WARNING: Remote URL is {result}, setting to {REPO_URL}", LOG_FILE)
        result = run_git_command(["remote", "set-url", "origin", REPO_URL], PROJECT_DIR)
        if "ERROR" in result:
            log_message(f"ERROR: Failed to set remote URL: {result}", LOG_FILE)
            print("ERROR: Could not set Git remote URL")
            sys.exit(1)
    log_message(f"Git remote URL: {REPO_URL}", LOG_FILE)

    # Checkout or create main branch
    result = run_git_command(["checkout", "main"], PROJECT_DIR)
    if "ERROR" in result:
        log_message("WARNING: Main branch not found, creating main branch", LOG_FILE)
        result = run_git_command(["checkout", "-b", "main"], PROJECT_DIR)
        if "ERROR" in result:
            log_message(f"ERROR: Failed to create or checkout main branch: {result}", LOG_FILE)
            print("ERROR: Could not checkout to the main branch")
            sys.exit(1)
    log_message("Checked out to the main branch", LOG_FILE)

    log_message("Git setup completed successfully", LOG_FILE)
    print("Git setup completed successfully.")
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()