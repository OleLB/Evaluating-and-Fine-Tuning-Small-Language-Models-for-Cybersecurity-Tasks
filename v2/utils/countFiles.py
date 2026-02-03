import os

def countFiles(directory: str) -> int:

    count = 0
    for root, dirs, files in os.walk(directory):
        count += len(files)
    return count

if __name__ == "__main__":
    directory_path = "LoRA/traning_data/"
    total_files = countFiles(directory_path)
    print(f"Total number of files in '{directory_path}': {total_files}")