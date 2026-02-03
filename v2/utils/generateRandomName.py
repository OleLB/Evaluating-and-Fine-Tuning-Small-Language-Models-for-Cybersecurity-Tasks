import random

def generateRandomName(length: int = 8) -> str:
    """Generate a random name consisting of uppercase and lowercase letters and digits."""
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890'
    return ''.join(random.choice(letters) for _ in range(length))