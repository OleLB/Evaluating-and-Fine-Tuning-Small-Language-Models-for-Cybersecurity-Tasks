"""
Query Qdrant vector database using text prompts.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from qdrant.getEmbedding import generate_embedding

# Initialize Qdrant client
qdrant = QdrantClient(host="localhost", port=6333)

def query_qdrant(input_text: str, collection_name: str = "", top_k: int = 20) -> str:
    """
    Query Qdrant collection with text input.
    
    Args:
        input_text: Text to search for
        collection_name: Name of the collection to search
        top_k: Number of top results to return
    
    Returns:
        Formatted string with results
    """
    print(f"Querying collection '{collection_name}' with input: {input_text}")
    
    if collection_name == "" or input_text == "":
        print("A collection name and a query must be provided.")
        return ""
    
    # Strip text
    query_text = input_text.strip()
    
    # Convert text to embedding
    query_embedding = generate_embedding(query_text)
    
    # Check if embedding generation failed
    if isinstance(query_embedding, dict) and "error" in query_embedding:
        print(f"Error generating embedding: {query_embedding.get('error')}")
        return ""
    
    # Perform search on Qdrant
    try:
        # Use 'query_points' instead of 'search' for newer versions
        results = qdrant.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k
        )
        
        # Access the points from the response
        points = results.points
        
    except AttributeError:
        # Fallback for older versions that might use 'search'
        try:
            results = qdrant.search(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
            points = results
        except Exception as e:
            print(f"Error querying Qdrant: {e}")
            return ""
    except Exception as e:
        print(f"Error querying Qdrant: {e}")
        return ""
    
    # Format results
    formatted_results = []
    for point in points:
        if point.payload is None:
            continue
        
        text = point.payload.get("text", "")
        doc = point.payload.get("document_name", "Unknown document")
        cve_id = point.payload.get("CVE_ID", "")
        score = point.score
        
        # Format with CVE_ID if available
        if cve_id:
            formatted_results.append(f"[{cve_id}] (Score: {score:.4f})\n{text}")
        else:
            formatted_results.append(f"[{doc}] (Score: {score:.4f})\n{text}")
    
    if not formatted_results:
        return "No results found."
    
    context_for_gpt = "\n\n---\n\n".join(formatted_results)
    return context_for_gpt


# Example usage
if __name__ == "__main__":
    print("Qdrant Query Tool")
    print("=" * 60)
    
    while True:
        print("\nType 'exit' to quit.")
        query = input("Write your query> ")
        
        if query == "" or query.lower() == "exit":
            print("Exiting...")
            exit(0)
        
        results = query_qdrant(query, "cpe_vulnerabilities")
        
        print("\n" + "=" * 60)
        print("RESULTS:")
        print("=" * 60)
        print(results)
        print("=" * 60)