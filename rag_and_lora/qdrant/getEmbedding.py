"""
This script generates text embeddings using a local Ollama server.

pip install ollama
ollama pull all-minilm:l6-v2

# Start the Ollama server
ollama serve
"""

from typing import List, Union

def generate_embedding(
    text: str,
    model: str = "all-minilm:l6-v2",
) -> Union[List[float], dict]:
    """
    Generate an embedding using Ollama.

    Returns:
        - List[float]: embedding on success
        - dict: error info on failure
    """

    # Basic input validation
    if not isinstance(text, str) or not text.strip():
        return {
            "error": "invalid_input",
            "message": "Input must be a non-empty string",
        }

    try:
        import ollama
    except ImportError:
        return {
            "error": "ollama_not_installed",
            "message": "The 'ollama' package is not installed (pip install ollama)",
        }

    try:
        response = ollama.embeddings(
            model=model,
            prompt=text,
        )

        # Response validation
        embedding = response.get("embedding")
        if not embedding or not isinstance(embedding, list):
            return {
                "error": "invalid_response",
                "message": "Ollama returned an invalid embedding response",
                "raw_response": response,
            }

        return embedding

    except Exception as e:
        error_msg = str(e).lower()

        if "connection" in error_msg or "refused" in error_msg:
            return {
                "error": "ollama_not_running",
                "message": "Ollama is not running. Start it with `ollama serve`.",
            }

        if "model" in error_msg and "not found" in error_msg:
            return {
                "error": "model_not_available",
                "message": f"Model '{model}' is not available. Run `ollama pull {model}`.",
            }

        return {
            "error": "unknown_error",
            "message": str(e),
        }


if __name__ == "__main__":
    # Test the embedding generation
    test_text = "Hello, world!"
    embedding = generate_embedding(test_text)

    if isinstance(embedding, dict) and "error" in embedding:
        print(f"Error generating embedding: {embedding['message']}")
    else:
        print(f"Generated embedding of length {len(embedding)} for text: '{test_text}'")