"""
Qdrant Collection Manager
A script to create collections and add documents to Qdrant vector database.
"""

from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rag_and_lora.qdrant.getEmbedding import generate_embedding
import uuid


class QdrantManager:
    """Manages Qdrant collections and operations."""
    
    def __init__(self, url: str = "http://localhost:6333"):
        """
        Initialize the Qdrant client.
        
        Args:
            url: Qdrant server URL
        """
        self.url = url
        self.client = None
        self._connect()
    
    def _connect(self) -> bool:
        """
        Establish connection to Qdrant server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client = QdrantClient(url=self.url)
            # Test connection
            self.client.get_collections()
            print(f"✓ Connected to Qdrant at {self.url}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to Qdrant at {self.url}")
            print(f"  Error: {str(e)}")
            print("  Please ensure Qdrant is running.")
            self.client = None
            return False
    
    def is_connected(self) -> bool:
        """Check if client is connected to Qdrant."""
        return self.client is not None
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 384,  # Default for all-minilm:l6-v2
        distance: Distance = Distance.COSINE
    ) -> bool:
        """
        Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (384 for all-minilm:l6-v2)
            distance: Distance metric (COSINE, EUCLID, DOT)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            print("✗ Not connected to Qdrant. Cannot create collection.")
            return False
        
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(col.name == collection_name for col in collections):
                print(f"✓ Collection '{collection_name}' already exists.")
                return True
            
            # Create collection
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            print(f"✓ Created collection '{collection_name}' (size={vector_size}, distance={distance.value})")
            return True
        
        except Exception as e:
            print(f"✗ Failed to create collection '{collection_name}'")
            print(f"  Error: {str(e)}")
            return False
    
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            True if collection exists, False otherwise
        """
        if not self.is_connected():
            return False
        
        try:
            collections = self.client.get_collections().collections
            return any(col.name == collection_name for col in collections)
        except Exception as e:
            print(f"✗ Error checking collection existence: {str(e)}")
            return False
    
    def add_document(
        self,
        collection_name: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        point_id: Optional[str] = None
    ) -> bool:
        """
        Add a document to a collection.
        
        Args:
            collection_name: Name of the collection
            text: Text content to embed and store
            metadata: Optional metadata to store with the document
            point_id: Optional custom ID for the point (UUID generated if not provided)
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            print("✗ Not connected to Qdrant. Cannot add document.")
            return False
        
        # Check if collection exists
        if not self.collection_exists(collection_name):
            print(f"✗ Collection '{collection_name}' does not exist.")
            print(f"  Please create the collection first using create_collection().")
            return False
        
        # Generate embedding
        embedding_result = generate_embedding(text)
        
        # Check if embedding generation returned an error
        if isinstance(embedding_result, dict) and "error" in embedding_result:
            print(f"✗ Failed to generate embedding for text")
            print(f"  Error: {embedding_result.get('error', 'Unknown error')}")
            return False
        
        # Prepare payload
        payload = {"text": text}
        if metadata:
            payload.update(metadata)
        
        # Generate ID if not provided
        if point_id is None:
            point_id = str(uuid.uuid4())
        
        try:
            # Add point to collection
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=embedding_result,
                        payload=payload
                    )
                ]
            )
            print(f"✓ Added document to '{collection_name}' (ID: {point_id})")
            return True
        
        except Exception as e:
            print(f"✗ Failed to add document to '{collection_name}'")
            print(f"  Error: {str(e)}")
            return False
    
    def add_documents_batch(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]]
    ) -> bool:
        """
        Add multiple documents to a collection in batch.
        
        Args:
            collection_name: Name of the collection
            documents: List of dicts with 'text' and optional 'metadata' and 'id' keys
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_connected():
            print("✗ Not connected to Qdrant. Cannot add documents.")
            return False
        
        # Check if collection exists
        if not self.collection_exists(collection_name):
            print(f"✗ Collection '{collection_name}' does not exist.")
            print(f"  Please create the collection first using create_collection().")
            return False
        
        points = []
        for i, doc in enumerate(documents):
            text = doc.get("text")
            if not text:
                print(f"⚠ Skipping document {i}: No 'text' field")
                continue
            
            # Generate embedding
            embedding_result = generate_embedding(text)
            
            # Check for errors
            if isinstance(embedding_result, dict) and "error" in embedding_result:
                print(f"⚠ Skipping document {i}: Embedding generation failed")
                print(f"  Error: {embedding_result.get('error', 'Unknown error')}")
                continue
            
            # Prepare payload
            payload = {"text": text}
            if "metadata" in doc and doc["metadata"]:
                payload.update(doc["metadata"])
            
            # Get or generate ID
            point_id = doc.get("id", str(uuid.uuid4()))
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding_result,
                    payload=payload
                )
            )
        
        if not points:
            print("✗ No valid documents to add")
            return False
        
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"✓ Added {len(points)} documents to '{collection_name}'")
            return True
        
        except Exception as e:
            print(f"✗ Failed to add documents to '{collection_name}'")
            print(f"  Error: {str(e)}")
            return False
    
    def list_collections(self) -> Optional[List[str]]:
        """
        List all collections.
        
        Returns:
            List of collection names or None if error
        """
        if not self.is_connected():
            print("✗ Not connected to Qdrant. Cannot list collections.")
            return None
        
        try:
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if collection_names:
                print(f"✓ Found {len(collection_names)} collection(s):")
                for name in collection_names:
                    print(f"  - {name}")
            else:
                print("✓ No collections found")
            
            return collection_names
        
        except Exception as e:
            print(f"✗ Failed to list collections")
            print(f"  Error: {str(e)}")
            return None
    
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a collection.
        
        Args:
            collection_name: Name of the collection
        
        Returns:
            Collection info dict or None if error
        """
        if not self.is_connected():
            print("✗ Not connected to Qdrant. Cannot get collection info.")
            return None
        
        if not self.collection_exists(collection_name):
            print(f"✗ Collection '{collection_name}' does not exist.")
            return None
        
        try:
            info = self.client.get_collection(collection_name)
            print(f"✓ Collection '{collection_name}' info:")
            
            # Handle different attribute names in different Qdrant versions
            points_count = getattr(info, 'points_count', getattr(info, 'vectors_count', 'N/A'))
            
            print(f"  - Points count: {points_count}")
            
            # Try to get vector config info
            if hasattr(info, 'config') and hasattr(info.config, 'params'):
                vector_size = info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else 'N/A'
                print(f"  - Vector size: {vector_size}")
            
            return info.dict() if hasattr(info, 'dict') else info.__dict__
        
        except Exception as e:
            print(f"✗ Failed to get collection info for '{collection_name}'")
            print(f"  Error: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = QdrantManager("http://localhost:6333")
    
    if manager.is_connected():
        # Create a collection
        manager.create_collection("my_documents", vector_size=384)
        
        # Add a single document
        manager.add_document(
            collection_name="my_documents",
            text="This is a sample document about machine learning.",
            metadata={"category": "AI", "author": "John Doe"}
        )
        
        # Add multiple documents
        documents = [
            {
                "text": "Python is a versatile programming language.",
                "metadata": {"category": "Programming"}
            },
            {
                "text": "Vector databases are useful for semantic search.",
                "metadata": {"category": "Databases"}
            }
        ]
        manager.add_documents_batch("my_documents", documents)
        
        # List all collections
        manager.list_collections()
        
        # Get collection info
        manager.get_collection_info("my_documents")
        
        # Try adding to non-existent collection (will show error)
        manager.add_document(
            collection_name="non_existent_collection",
            text="This will fail."
        )