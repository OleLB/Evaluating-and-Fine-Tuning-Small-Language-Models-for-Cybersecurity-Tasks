'''
This file is meant to be run manually by an admin or developer to update the Qdrant database.
Run this script after adding new documents to the ./db/qdrant_docs/<course_id>/ directory.
It will read all .txt files in that directory, embed their content using OpenAI embeddings, and store them in Qdrant.
'''

import uuid
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from utils.get_embedding import get_embeddings
from utils.service_manager import check_qdrant_running
from utils.log import log_error


def get_collection_name(docs_dir):
    """
    This function lists available collections (folders) in ./db/qdrant_docs/ and prompts the user to select one.
    The function return the name of the selected collection (folder) or False if the user exits or an error occurs.
    """

    if not docs_dir.exists():
        print("No documents directory found. Please create './db/qdrant_docs/' and add your documents.")
        log_error(
            "No documents directory found. Please create './db/qdrant_docs/' and add your documents.", "update_qdrant.py"
        )
        return False

    folders = [f for f in docs_dir.iterdir() if f.is_dir()]
    if not folders:
        print("No collections found in './db/qdrant_docs/'. Please use the 'database_tool.py' to create a new course.")
        log_error(
            "No collections found in './db/qdrant_docs/'. Please use the 'database_tool.py' to create a new course.", "update_qdrant.py"
        )
        return False

    print("Available collections:")
    for idx, folder in enumerate(folders):
        print(f"{idx + 1}: {folder.name}")

    choice = input("Select a collection by number or course id (or type 'exit' to cancel): ").strip()
    if choice.lower() == "exit" or choice.lower() == "q" or choice.lower() == "":
        return False

    if choice in [folder.name for folder in folders]:
        return choice

    try:
        index = int(choice) - 1
        if 0 <= index < len(folders):
            print(f"You selected: {folders[index].name}")
            return folders[index].name
        else:
            print("Invalid selection. Exiting.")
            return False
    except ValueError:
        print("Invalid input. Exiting.")
        return False


def update_qdrant_collection():
    """ 
    Update the Qdrant collection with new documents.
    """

    if not check_qdrant_running():
        print("Qdrant is not running. Please start it using the service manager.")
        log_error("Qdrant is not running. Please start it using the service manager.", "update_qdrant.py")
        exit(1)

    client = QdrantClient(host="localhost", port=6333)
    docs_dir = Path("./db/qdrant_docs")


    try:
        collection_name = get_collection_name(docs_dir)
        if not collection_name:
            print("No valid collection selected. Exiting.")
            return False
    except Exception as e:
        print(f"Error selecting collection: {e}")
        log_error(f"Error selecting collection: {e}", "update_qdrant.py")
        return False

    print(f"Selected collection: {collection_name}")

    print("Checking if the collection exists in Qdrant...")
    # print(client.get_collections().collections)

    # Check if the collection exists
    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name not in existing_collections:
        confirm = (
            input(
                f"Collection '{collection_name}' does not exist. Do you want to create it? (y/n): "
            )
            .strip()
            .lower()
        )
        if confirm != "y":
            print("Exiting without creating the collection.")
            return False
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )


    # --- Read already processed documents ---
    already_read_path = Path("./db/already_read.txt")
    already_read = set()
    if already_read_path.exists():
        with already_read_path.open("r", encoding="utf-8") as f:
            already_read = set(line.strip() for line in f.readlines())
    else:
        already_read_path.touch() # Create the file if it doesn't exist

    # --- Read documents ---
    course_docs_dir = Path(f"./db/qdrant_docs/{collection_name}")
    # if not course_docs_dir.exists():
    #     print(f"Directory {course_docs_dir} does not exist.")
    #     exit(1)

    for doc_file in course_docs_dir.glob("*.txt"):
        doc_key = f"{collection_name}/{doc_file.name}"
        if doc_key in already_read:
            print(f"The document '{doc_file.name}' has already been read. Skipping.")
            continue

        with doc_file.open("r", encoding="utf-8") as f:
            content = f.read()

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            print(f"'{doc_file.name}' has no valid paragraphs.")
            continue

        # print(paragraphs)

        print(f"Embedding {len(paragraphs)} paragraphs from {doc_file.name}...")
        embeddings = get_embeddings(paragraphs)[0]

        # print(embeddings)

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": paragraph,
                    "document_name": doc_file.name,
                },
            )
            for paragraph, embedding in zip(paragraphs, embeddings)
        ]

        client.upsert(collection_name=collection_name, points=points)

        # Mark document as processed
        with already_read_path.open("a", encoding="utf-8") as f:
            f.write(doc_key + "\n")

        print(f"Finished processing {doc_file.name}.")

    print(f"All documents for collection {collection_name} processed.")
    return True


if __name__ == "__main__":
    update_qdrant_collection()