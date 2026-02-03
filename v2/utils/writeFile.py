import os

def writeFile(filepath: str, content: str) -> None:
    """Writes content to a file at the specified filepath with error handling."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Write content to the file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
    except FileNotFoundError:
        print(f"Error: The file path '{filepath}' is invalid.")
    except PermissionError:
        print(f"Error: Permission denied when writing to '{filepath}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")