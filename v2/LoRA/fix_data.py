import os
import json

def fix_data_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            # read whole JSON object
            with open(file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")
                    continue

            # modify
            if "response" in data:
                data["output"] = data.pop("response")

            # write back nicely formatted
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)


def remove_single_quote():
    # Soem files contains lines with only one double quote
    # delete these lines
    directory_path = "./LoRA/traning_data"
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # filter out lines that are just a single double quote
            filtered_lines = [line for line in lines if line.strip() != '"']

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(filtered_lines)


if __name__ == "__main__":
    fix_data_in_directory("LoRA/traning_data")
    # remove_single_quote()