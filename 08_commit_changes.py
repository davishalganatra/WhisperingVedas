import os
import sys
import json
from datetime import datetime
from pathlib import Path
from modules.input_parser import parse_input_files
from modules.template_generator import TemplateGenerator
from modules.usage_tracker import UsageTracker
from modules.chakra_energy_kpi import ChakraEnergyKPI

# Configuration
PROJECT_DIR = r"D:\mcpserver\mcp_server"
INPUT_DIR = os.path.join(PROJECT_DIR, "data", "inputs")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "knowledge_parsed")
LOG_FILE = os.path.join(PROJECT_DIR, "update_log.txt")
TEMP_LOG = os.path.join(os.getenv("TEMP"), "update_log.txt")
VENV_PYTHON = os.path.join(PROJECT_DIR, "venv", "Scripts", "python.exe")

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
    log_message("Starting input processing", TEMP_LOG)

    # Verify project directory
    if not os.path.exists(PROJECT_DIR):
        log_message(f"ERROR: Directory {PROJECT_DIR} not found", TEMP_LOG)
        print(f"ERROR: Project directory {PROJECT_DIR} does not exist")
        sys.exit(1)

    # Verify input and output directories
    if not os.path.exists(INPUT_DIR):
        log_message(f"ERROR: Input directory {INPUT_DIR} not found", TEMP_LOG)
        print(f"ERROR: Input directory does not exist")
        sys.exit(1)
    if not os.path.exists(OUTPUT_DIR):
        log_message(f"ERROR: Output directory {OUTPUT_DIR} not found", TEMP_LOG)
        print(f"ERROR: Output directory does not exist")
        sys.exit(1)

    # Check write permissions
    if not check_permissions(OUTPUT_DIR):
        log_message(f"ERROR: No write permissions in {OUTPUT_DIR}", TEMP_LOG)
        print("ERROR: Cannot write to output directory. Run as Administrator.")
        sys.exit(1)

    # Initialize project log
    log_message("Starting input processing", LOG_FILE)

    # Initialize modules
    try:
        template_gen = TemplateGenerator(OUTPUT_DIR)
        tracker = UsageTracker()
        chakra_kpi = ChakraEnergyKPI()
    except Exception as e:
        log_message(f"ERROR: Failed to initialize modules: {e}", LOG_FILE)
        print(f"ERROR: Could not initialize modules: {e}")
        sys.exit(1)

    # Parse input files
    log_message("Parsing input files", LOG_FILE)
    print("Processing input files...")
    try:
        texts = parse_input_files(INPUT_DIR)
        if not texts:
            log_message("WARNING: No valid input files found", LOG_FILE)
            print("WARNING: No valid input files found in data/inputs")
        else:
            log_message(f"Found {len(texts)} input files", LOG_FILE)
    except Exception as e:
        log_message(f"ERROR: Failed to parse input files: {e}", LOG_FILE)
        print(f"ERROR: Could not parse input files: {e}")
        sys.exit(1)

    # Process texts and generate templates
    try:
        template_gen.process_texts(texts)
        log_message(f"Generated templates for {len(texts)} files", LOG_FILE)
    except Exception as e:
        log_message(f"ERROR: Failed to generate templates: {e}", LOG_FILE)
        print(f"ERROR: Could not generate templates: {e}")
        sys.exit(1)

    # Log a sample interaction
    sample_user_id = "setup_user"
    sample_query = "Process initial inputs for setup"
    try:
        for item in texts:
            kpi = chakra_kpi.analyze(item["text"])
            tracker.log_interaction(sample_user_id, {
                "query": sample_query,
                "chakra": kpi["chakra"],
                "mantra": kpi["mantra"],
                "emotion": kpi["emotion"]
            })
            log_message(f"Logged interaction for {item['file']}", LOG_FILE)
    except Exception as e:
        log_message(f"ERROR: Failed to log interaction: {e}", LOG_FILE)
        print(f"ERROR: Could not log interaction: {e}")
        sys.exit(1)

    # Verify output
    output_files = list(Path(OUTPUT_DIR).glob("*.json"))
    if not output_files:
        log_message("WARNING: No output files generated", LOG_FILE)
        print("WARNING: No output files were created in data/knowledge_parsed")
    else:
        log_message(f"Created {len(output_files)} output files", LOG_FILE)

    log_message("Input processing completed successfully", LOG_FILE)
    print("Input processing completed successfully.")
    input("Press Enter to continue...")

if __name__ == "__main__":
    main()