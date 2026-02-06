import os


def fix_line(line: str) -> str:
    stripped = line.strip()

    # Skip lines that aren't key-value pairs
    if '": "' not in stripped:
        return line

    indent = line[:len(line) - len(line.lstrip())]

    # Split safely at key/value boundary
    key_part, value_part = stripped.split('": "', 1)

    key = key_part + '"'          # add back closing quote

    value = value_part

    # Remove ending ", or "
    if value.endswith('",'):
        value = value[:-2]
        trailing = ','
    elif value.endswith('"'):
        value = value[:-1]
        trailing = ''
    else:
        trailing = ''

    # Replace internal double quotes
    value = value.replace('"', "'")

    # Rebuild line
    new_line = f'{indent}{key}: "{value}"{trailing}\n'

    return new_line


def fix_data_in_directory(directory_path="./LoRA/traning_data"):
    for filename in os.listdir(directory_path):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(directory_path, filename)

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        fixed_lines = [fix_line(line) for line in lines]

        with open(path, "w", encoding="utf-8") as f:
            f.writelines(fixed_lines)

        print(f"Fixed {filename}")


if __name__ == "__main__":
    fix_data_in_directory()
