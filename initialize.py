"""
## Initialization script

Verify dependencies
- ollama (ollama --version)
- docker (docker --version)
- AI model tensor files are available
    /finetuned_models/deepseekcoder_cve/deepseekcoder-lora-adapter-final/model_adapter.safetensors
    /finetuned_models/llama31_model_cve/llama31-lora-adapter-final/model_adapter.safetensors
    /finetuned_models/mistral_nemo_cve/mistral-lora-adapter-final/model_adapter.safetensors
- relational database is available
    ./db/cve_database.db
- CPE string file available
    ./db/cpe_list.csv

 Create vector database 
 - check if container exists
 - check if image exists
 - pull image
 - create container
 - start container
 - fill inn data

 


Set up models
start ollama in background (ollama serve --background) # This process must keep running after this script finishes
using tensor file, set up the 3 cve models (requires running commands)

"""

import subprocess
import os
import json


# Dependency checks

def check_ollama():
    try:
        result = subprocess.run(['ollama', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("Ollama is installed: ", result.stdout.strip())
            return True
        else:
            print("Ollama is not installed or not found in PATH.")
            print("Error: ", result.stderr.strip())
            return False
    except FileNotFoundError:
        print("Ollama command not found. Please install Ollama and ensure it's in your PATH.")
        return False


def check_docker():
    try:
        result = subprocess.run(['docker', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("Docker is installed: ", result.stdout.strip())
            return True
        else:
            print("Docker is not installed or not found in PATH.")
            print("Error: ", result.stderr.strip())
            return False
    except FileNotFoundError:
        print("Docker command not found. Please install Docker and ensure it's in your PATH.")
        return False
    

def check_tensor_files():
    """
    Ensure that the required tensor files for the AI models are present in the specified locations.
    """
    required_files = [
        "./finetuned_models/deepseekcoder_cve/deepseekcoder-lora-adapter-final/model_adapter.safetensors",
        "./finetuned_models/llama31_model_cve/llama31-lora-adapter-final/model_adapter.safetensors",
        "./finetuned_models/mistral_nemo_cve/mistral-lora-adapter-final/model_adapter.safetensors"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.isfile(file):
            missing_files.append(file)
    
    if missing_files:
        print("The following required tensor files are missing:")
        for file in missing_files:
            print(f" - {file}")
        return False
    else:
        print("All required tensor files are present.")
        return True


def database_check():
    """
    Check if the relational database is available at the specified location.
    """
    db_path = "./db/cve_database.db"
    if os.path.isfile(db_path):
        print("Relational database is available: ", db_path)
        return True
    else:
        print("Relational database not found at: ", db_path)
        return False
    

def cpe_file_check():
    """
    Check if the CPE string file is available at the specified location.
    """
    cpe_file_path = "./db/cpe_list.csv"
    if os.path.isfile(cpe_file_path):
        print("CPE string file is available: ", cpe_file_path)
        return True
    else:
        print("CPE string file not found at: ", cpe_file_path)
        return False


def start_ollama():
    """
    Start the Ollama server in the background if it's not already running.
    """
    try:
        # Check if Ollama is already running
        result = subprocess.run(['ollama', 'list'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("Ollama is already running.")
            return True
        else:
            print("Starting Ollama server in the background...")
            subprocess.Popen(['ollama', 'serve', '--background'])
            print("Ollama server started.")
            return True
    except FileNotFoundError:
        print("Ollama command not found. Please install Ollama and ensure it's in your PATH.")
        return False
    except Exception as e:
        print(f"Failed to start Ollama: {str(e)}")
        return False


def _get_installed_models() -> set[str]:
    """Return a set of all model names currently installed in Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list", "--json"],
            capture_output=True, text=True, check=True,
        )
        return {model["name"] for model in json.loads(result.stdout)}
    except (json.JSONDecodeError, KeyError, subprocess.CalledProcessError):
        # Fallback to plain-text parsing
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, check=True,
        )
        lines = result.stdout.strip().splitlines()[1:]  # skip header
        return {line.split()[0] for line in lines if line.strip()}


def check_ollama_models() -> dict[str, bool]:
    """
    Check if specific AI models are installed and available through Ollama.

    Returns:
        A dictionary mapping model names to their availability (True/False).
    """
    required_models = [
        "deepseek-coder:1.3b",
        "llama3.1:8b",
        "mistral-nemo:12b-instruct-2407-q8_0",
        "all-minilm:l6-v2",
    ]

    try:
        installed = _get_installed_models()
    except FileNotFoundError:
        raise RuntimeError("Ollama is not installed or not found in PATH.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to query Ollama: {e.stderr.strip()}")

    return {model: model in installed for model in required_models}


def print_model_status(status: dict[str, bool]) -> None:
    """Pretty-print the availability status of each model."""
    print(f"\n{'Model':<45} {'Status'}")
    print("-" * 55)
    for model, available in status.items():
        indicator = "✅ Installed" if available else "❌ Not found"
        print(f"{model:<45} {indicator}")

    missing = [m for m, ok in status.items() if not ok]
    if missing:
        print("\nTo install missing models, run:")
        for model in missing:
            print(f"  ollama pull {model}")
        exit(1)
    else:
        print("\nAll models are installed and available.")


# Vector database setup

from rag_and_lora.qdrant.FullSetup import qdrant_data_initialization


# Model setup

def model_setup():
    """
    Set up the AI models using Ollama and the provided tensor files.
    This function assumes that Ollama is already running in the background.
    Skips creation if the model already exists.
    Command format: ollama create <new_model_name>:latest -f Modelfile
    Model names: deepseek_coder_cve, llama3.1_cve, mistral_nemo_cve
    Modelfiles:
        /finetuned_models/deepseekcoder_cve/Modelfile
        /finetuned_models/llama31_model_cve/Modelfile
        /finetuned_models/mistral_nemo_cve/Modelfile
    """
    modelfiles = {
        "deepseek_coder_cve:latest": "./finetuned_models/deepseekcoder_cve/Modelfile",
        "llama3.1_cve:latest":       "./finetuned_models/llama31_model_cve/Modelfile",
        "mistral_nemo_cve:latest":   "./finetuned_models/mistral_nemo_cve/Modelfile",
    }

    try:
        installed_models = _get_installed_models()
    except Exception as e:
        print(f"✗ Could not retrieve installed models from Ollama: {e}")
        return

    for model_name, modelfile in modelfiles.items():
        # ── Skip if already installed ──────────────────────────────────────
        if model_name in installed_models:
            print(f"– Skipping '{model_name}': already installed.")
            continue

        # ── Validate Modelfile exists before attempting creation ───────────
        if not os.path.isfile(modelfile):
            print(f"✗ Modelfile not found for '{model_name}': {modelfile}")
            continue

        # ── Create the model ───────────────────────────────────────────────
        print(f"Creating Ollama model '{model_name}' from '{modelfile}'...")
        try:
            subprocess.run(
                ["ollama", "create", model_name, "-f", modelfile],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"✓ Successfully created model '{model_name}'")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create model '{model_name}': {e.stderr.strip()}")
        except Exception as e:
            print(f"✗ Unexpected error while creating model '{model_name}': {e}")


def full_setup():

    # print some ascii startup banner to grab attention
    print("\033[91m\033[1m")  # Bold red
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║             ACTION REQUIRED BEFORE CONTINUING                ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Ensure Docker daemon is running. Please:                    ║")
    print("║                                                              ║")
    print("║   1. Start Docker Desktop (or the Docker service)            ║")
    print("║   2. Ensure you have permission to access the daemon         ║")
    print("║   3. Confirm that you have seen this message                 ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print("\033[0m")  # Reset
    while True:
        user_input = input("Type 'DOCKER' to confirm: ").strip().lower()
        if user_input == 'docker':
            break
        elif user_input == 'exit':
            print("Please start Docker and then run this script again.")
            exit(1)
        else:
            print("Invalid input. Please enter 'docker' or 'exit'.")

    # Verify ollama and docker installation
    if not check_ollama():
        print("Please install Ollama before running this script.")
        exit(1)

    if not check_docker():
        print("Please install Docker before running this script.")
        print("If docker is already installed, ensure that the Docker daemon is running and that you have permission to access it.")
        exit(1)

    # Verify database, tensorfile and CPE file availability
    if not check_tensor_files():
        print("Please ensure all required tensor files are present before running this script.")
        exit(1)

    if not database_check():
        print("Please ensure the relational database is available before running this script.")
        exit(1)

    if not cpe_file_check():
        print("Please ensure the CPE string file is available before running this script.")
        exit(1)


    # Model setup
    status = check_ollama_models()
    print_model_status(status)

    model_setup()


    # Vector database setup
    qdrant_data_initialization()


if __name__ == "__main__":
    full_setup()