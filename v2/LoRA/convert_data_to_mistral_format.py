"""
This is NOT correct, dont use
"""
import json
from pathlib import Path

INPUT_DIR = Path("LoRA/training_data")
OUTPUT_DIR = Path("LoRA/training_data_mistral_format")
OUTPUT_DIR.mkdir(exist_ok=True)


def to_mistral_text(instruction, input_text, output_text):
    instruction = (instruction or "").strip()
    input_text = (input_text or "").strip()
    output_text = (output_text or "").strip()

    if input_text:
        prompt = f"{instruction}\n\n{input_text}"
    else:
        prompt = instruction

    return f"<s>[INST] {prompt} [/INST] {output_text} </s>"


def write_example(example, f):
    text = to_mistral_text(
        example.get("instruction"),
        example.get("input"),
        example.get("output"),
    )
    json.dump({"text": text}, f, ensure_ascii=False)
    f.write("\n")


def process_file(path: Path):
    out_path = OUTPUT_DIR / path.name.replace(".json", ".jsonl")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()

    with open(out_path, "w", encoding="utf-8") as out_f:
        try:
            # Try parsing as full JSON first (object or array)
            data = json.loads(raw)

            if isinstance(data, list):
                for example in data:
                    write_example(example, out_f)
            else:
                write_example(data, out_f)

        except json.JSONDecodeError:
            # Fallback: JSONL
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                write_example(example, out_f)


def main():
    for file in INPUT_DIR.glob("*.json"):
        print(f"Processing {file.name}")
        process_file(file)

    print("Done.")


if __name__ == "__main__":
    main()
