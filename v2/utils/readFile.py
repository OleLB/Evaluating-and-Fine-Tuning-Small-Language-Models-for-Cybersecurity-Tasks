import os

def readFile(file_path: str) -> str:
    """Read the content of a file and return it as a string."""
    if not os.path.isfile(file_path):
        raise ValueError(f"The path '{file_path}' is not a valid file.")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except PermissionError:
        raise PermissionError(f"Permission denied to read the file '{file_path}'.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file '{file_path}': {e}")

