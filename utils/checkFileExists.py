def checkFileExists(file_path: str) -> bool:
    """Check if a file exists at the given path."""
    try:
        with open(file_path, 'r'):
            return True
    except FileNotFoundError:
        return False