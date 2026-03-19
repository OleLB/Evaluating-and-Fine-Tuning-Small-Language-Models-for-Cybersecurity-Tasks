"""
Load CPE List to Qdrant
Reads cpe_list.csv and adds entries to Qdrant vector database in batches.
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Any
from rag_and_lora.qdrant.qdrantManager import QdrantManager

CSV_FILE = "db/cpe_list.csv"


def read_cpe_csv(file_path: str) -> List[Dict[str, Any]]:
    """
    Read CPE list from CSV file.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        List of documents with text and metadata
    """
    documents = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                cve_id = row.get('CVE_ID', '').strip()
                human_readable = row.get('HUMAN_READABLE', '').strip()
                cpe = row.get('CPE', '').strip()
                
                # Skip rows with missing data
                if not cve_id or not human_readable:
                    continue
                
                documents.append({
                    'text': human_readable,
                    'metadata': {
                        'CVE_ID': cve_id,
                        'CPE': cpe
                    }
                })
        
        return documents
    
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        print(f"  Please ensure '{CSV_FILE}' is in the current directory.")
        return []
    
    except Exception as e:
        print(f"✗ Error reading CSV file: {str(e)}")
        return []


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
    
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def load_cpe_to_qdrant(
    csv_file: str = CSV_FILE,
    collection_name: str = "cpe_vulnerabilities",
    qdrant_url: str = "http://localhost:6333",
    vector_size: int = 384,
    batch_size: int = 100
):
    """
    Load CPE data from CSV into Qdrant in batches.
    
    Args:
        csv_file: Path to CSV file
        collection_name: Name of the Qdrant collection
        qdrant_url: Qdrant server URL
        vector_size: Vector dimension (384 for all-minilm:l6-v2)
        batch_size: Number of documents to process in each batch
    """
    print(f"Loading CPE data from '{csv_file}' into Qdrant collection '{collection_name}'...\n")
    
    # Initialize Qdrant manager
    manager = QdrantManager(qdrant_url)
    
    if not manager.is_connected():
        print("\n✗ Failed to connect to Qdrant. Exiting.")
        sys.exit(1)
    
    # Read CSV data
    print(f"\nReading CSV file '{csv_file}'...")
    documents = read_cpe_csv(csv_file)
    
    if not documents:
        print("\n✗ No valid documents found in CSV. Exiting.")
        sys.exit(1)
    
    print(f"✓ Found {len(documents)} entries in CSV\n")
    
    # Create collection if it doesn't exist
    print(f"Creating collection '{collection_name}'...")
    if not manager.create_collection(collection_name, vector_size=vector_size):
        print("\n✗ Failed to create collection. Exiting.")
        sys.exit(1)
    
    print()
    
    # Split documents into batches
    batches = chunk_list(documents, batch_size)
    total_batches = len(batches)
    
    print(f"Processing {len(documents)} documents in {total_batches} batches of {batch_size}...\n")
    
    # Process each batch
    successful_batches = 0
    total_added = 0
    
    for i, batch in enumerate(batches, 1):
        print(f"Processing batch {i}/{total_batches} ({len(batch)} documents)...", end=" ")
        
        if manager.add_documents_batch(collection_name, batch):
            successful_batches += 1
            total_added += len(batch)
            print(f"✓ ({total_added}/{len(documents)} total)")
        else:
            print(f"✗ Failed")
            print(f"⚠ Continuing with remaining batches...")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total documents: {len(documents)}")
    print(f"  Successfully added: {total_added}")
    print(f"  Failed: {len(documents) - total_added}")
    print(f"  Successful batches: {successful_batches}/{total_batches}")
    print(f"{'='*60}\n")
    
    if total_added > 0:
        print(f"✓ Successfully loaded {total_added} CPE entries into Qdrant!")
        
        # Show collection info
        print()
        manager.get_collection_info(collection_name)
    else:
        print("✗ No documents were added to the collection.")
        sys.exit(1)


if __name__ == "__main__":
    # Check if custom CSV file is provided as argument
    csv_file = sys.argv[1] if len(sys.argv) > 1 else CSV_FILE
    
    # Optional: batch size as second argument
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Check if file exists
    if not Path(csv_file).exists():
        print(f"✗ File '{csv_file}' not found in current directory.")
        print(f"\nUsage: python {sys.argv[0]} [csv_file] [batch_size]")
        print(f"Example: python {sys.argv[0]} {CSV_FILE} 100")
        sys.exit(1)
    
    # Load data
    load_cpe_to_qdrant(csv_file=csv_file, batch_size=batch_size)