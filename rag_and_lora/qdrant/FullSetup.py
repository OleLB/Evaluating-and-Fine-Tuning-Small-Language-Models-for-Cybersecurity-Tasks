"""
This script should download and initialize qdrant if not already present
Then load the CPE data into the database if not already loaded
"""

from rag_and_lora.qdrant.loadCPEs import load_cpe_to_qdrant
from rag_and_lora.qdrant.startQdrant import start_qdrant
from rag_and_lora.qdrant.qdrantManager import QdrantManager

print("Please ensure that Docker and ollama is running before executing this script.")
confirm = input("Have you started Docker and Ollama? (y/n): ").strip().lower()
if confirm != 'y':
    print("Please start Docker and Ollama and then run this script again.")
    exit(1)

qdrant_available = start_qdrant()
if not qdrant_available:
    print("Failed to start Qdrant. Please check the logs.")
    exit(1)

print("Qdrant is running. Attempting to connect...")
qdrant_manager = QdrantManager()

if not qdrant_manager.is_connected():
    print("Failed to connect to Qdrant. Please check the logs.")
    exit(1)

print("Connected to Qdrant successfully!")
print("Checking if CPE data is already loaded...")

collections = qdrant_manager.list_collections()
if "cpe_vulnerabilities" in collections:
    print("CPE collection already exists. Assuming data is loaded.")
else:
    print("CPE collection not found. Loading CPE data...")
    load_cpe_to_qdrant()

print("\nSetup complete! Qdrant is running and CPE data is loaded.")