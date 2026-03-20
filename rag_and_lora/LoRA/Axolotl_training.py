"""
LoRA Fine-tuning Pipeline for Llama 3.1 using Axolotl
======================================================
Hardware target: RTX 4070 Ti (12GB VRAM) + 32GB RAM

Pipeline:
  1. Load and convert your custom Alpaca-format JSON files
  2. Sample 9000 examples from the open-source Alpaca dataset
  3. Merge and shuffle everything into a single JSONL file
     formatted as Llama 3.1 chat messages
  4. Write the Axolotl config YAML
  5. Launch training via subprocess

Usage:
  pip install axolotl datasets transformers
  python prepare_and_train.py
"""

import json
import random
import subprocess
import sys
import yaml
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import login

from utils.readFile import readFile

# ─────────────────────────────────────────────
#  PATHS  –  adjust if your CWD is different
# ─────────────────────────────────────────────
HUGGINGFACE_TOKEN_PATH = "keys/huggingface_token.key"

GENERAL_DATA_DIR   = Path("rag_and_lora/LoRA/training_data/general_data")
CVE_DATA_DIR       = Path("rag_and_lora/LoRA/training_data/training_data_explain_CVE")
OUTPUT_DIR         = Path("finetuned_models")
MERGED_DATASET     = OUTPUT_DIR / "merged_dataset.jsonl"
CONFIG_PATH        = OUTPUT_DIR / "axolotl_config.yaml"

ALPACA_SAMPLES     = 9000
RANDOM_SEED        = 42

# Cybersecurity system prompt injected into every custom sample
CYBER_SYSTEM_PROMPT = (
    "You are a cybersecurity expert. Use precise technical terminology, "
    "standard industry acronyms (e.g., CVE, CVSS, RCE, SQLi, XSS, MITM, "
    "IoC, TTP, APT, SIEM, EDR, WAF, IAM, MFA, PKI), and professional "
    "language in all responses. Ensure accuracy and clarity when describing "
    "vulnerabilities, attack vectors, mitigations, and security concepts."
)

# Generic system prompt for Alpaca samples (preserves general capability)
ALPACA_SYSTEM_PROMPT = (
    "You are a helpful, accurate, and professional assistant."
)


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def alpaca_to_messages(sample: dict, system_prompt: str) -> dict | None:
    """
    Convert an Alpaca-style record  {instruction, input, output}
    into a Llama 3.1 chat messages list.
    Returns None if the record is malformed / empty.
    """
    instruction = sample.get("instruction", "").strip()
    inp         = sample.get("input", "").strip()
    output      = sample.get("output", "").strip()

    if not instruction or not output:
        return None

    user_content = instruction
    if inp:
        user_content = f"{instruction}\n\n{inp}"

    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def load_jsonl_or_json(path: Path) -> list[dict]:
    """
    Load a file that is one of:
      - A single JSON object  { ... }          ← your current format
      - A JSON array          [ {...}, {...} ]
      - Newline-delimited JSON  (one object per line)

    Uses utf-8-sig so files saved with a BOM (common on Windows) are handled.
    """
    text = path.read_text(encoding="utf-8-sig").strip()

    if text.startswith("{"):
        # Single JSON object – wrap in a list
        return [json.loads(text)]

    if text.startswith("["):
        # JSON array
        return json.loads(text)

    # JSONL – one JSON object per line
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_custom_data(directory: Path, system_prompt: str) -> list[dict]:
    """Recursively load all .json / .jsonl files from a directory."""
    samples = []
    files   = list(directory.rglob("*.json")) + list(directory.rglob("*.jsonl"))

    if not files:
        print(f"  [WARNING] No JSON files found in {directory}")
        return samples

    for f in sorted(files):
        try:
            records = load_jsonl_or_json(f)
            for rec in records:
                converted = alpaca_to_messages(rec, system_prompt)
                if converted:
                    samples.append(converted)
            print(f"  Loaded {len(records):>5} records  ←  {f.name}")
        except Exception as e:
            print(f"  [ERROR] Could not parse {f}: {e}")

    return samples


# ─────────────────────────────────────────────
#  STEP 1 – Load custom cybersecurity data
# ─────────────────────────────────────────────

def load_all_custom_samples() -> list[dict]:
    print("\n── Loading custom cybersecurity data ──────────────────────")

    general_samples = []
    if GENERAL_DATA_DIR.exists():
        print(f"  Scanning: {GENERAL_DATA_DIR}")
        general_samples = load_custom_data(GENERAL_DATA_DIR, CYBER_SYSTEM_PROMPT)
    else:
        print(f"  [WARNING] Directory not found: {GENERAL_DATA_DIR}")

    cve_samples = []
    if CVE_DATA_DIR.exists():
        print(f"  Scanning: {CVE_DATA_DIR}")
        cve_samples = load_custom_data(CVE_DATA_DIR, CYBER_SYSTEM_PROMPT)
    else:
        print(f"  [WARNING] Directory not found: {CVE_DATA_DIR}")

    combined = general_samples + cve_samples
    print(f"\n  ✓ Total custom samples: {len(combined)}")
    return combined


# ─────────────────────────────────────────────
#  STEP 2 – Sample from open-source Alpaca
# ─────────────────────────────────────────────

def load_alpaca_samples(n: int) -> list[dict]:
    print(f"\n── Loading {n} samples from tatsu-lab/alpaca ───────────────")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    
    rng     = random.Random(RANDOM_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    
    samples = []
    for idx in indices:
        rec = ds[idx]
        converted = alpaca_to_messages(rec, ALPACA_SYSTEM_PROMPT)
        if converted:
            samples.append(converted)

    print(f"  ✓ Alpaca samples loaded: {len(samples)}")
    return samples


# ─────────────────────────────────────────────
#  STEP 3 – Merge, shuffle, write JSONL
# ─────────────────────────────────────────────

def write_merged_dataset(custom: list[dict], alpaca: list[dict]) -> None:
    print("\n── Merging and writing dataset ─────────────────────────────")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    combined = custom + alpaca
    random.seed(RANDOM_SEED)
    random.shuffle(combined)

    with MERGED_DATASET.open("w", encoding="utf-8") as f:
        for sample in combined:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"  ✓ Total samples : {len(combined)}")
    print(f"    Custom (cyber) : {len(custom)}")
    print(f"    Alpaca         : {len(alpaca)}")
    print(f"  ✓ Written to     : {MERGED_DATASET}")


# ─────────────────────────────────────────────
#  STEP 4 – Generate Axolotl config
# ─────────────────────────────────────────────

def write_axolotl_config() -> None:
    print("\n── Writing Axolotl config ──────────────────────────────────")

    config = {
        # ── Model ────────────────────────────────────────────────────
        "base_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "model_type": "LlamaForCausalLM",
        "tokenizer_type": "AutoTokenizer",

        # ── Dataset ──────────────────────────────────────────────────
        # 'chat_template' type tells Axolotl the data is already in
        # messages format and to apply the model's native chat template.
        "datasets": [
            {
                "path": str(MERGED_DATASET.resolve()),
                "type": "chat_template",
                "chat_template": "llama3",          # applies Llama 3.1 special tokens
                "field_messages": "messages",
                "message_field_role": "role",
                "message_field_content": "content",
                "roles": {
                    "system":    ["system"],
                    "user":      ["human", "user"],
                    "assistant": ["gpt", "assistant"],
                },
            }
        ],
        "dataset_prepared_path": str((OUTPUT_DIR / "prepared_data").resolve()),
        "val_set_size": 0.02,           # 2 % held-out validation split
        "output_dir": str((OUTPUT_DIR / "checkpoints").resolve()),

        # ── Sequence length ──────────────────────────────────────────
        # 4096 fits comfortably in 12 GB VRAM with the settings below.
        # Increase to 8192 if your samples are longer and you have headroom.
        "sequence_len": 4096,
        "sample_packing": True,         # packs short samples to fill the window
        "pad_to_sequence_len": True,

        # ── LoRA / QLoRA ─────────────────────────────────────────────
        "adapter": "lora",
        "load_in_4bit": True,           # QLoRA – cuts VRAM roughly in half
        "lora_r": 32,
        "lora_alpha": 64,               # typically 2 × r
        "lora_dropout": 0.05,
        "lora_target_linear": True,     # targets all linear layers (recommended)
        # Explicit target modules as a fallback / for documentation:
        "lora_target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],

        # ── Quantization (bitsandbytes) ───────────────────────────────
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,

        # ── Training hyperparameters ─────────────────────────────────
        # Effective batch = micro_batch × gradient_accumulation = 1 × 8 = 8
        # This keeps VRAM usage safe on a 4070 Ti (12 GB).
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "lr_scheduler": "cosine",
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "optimizer": "paged_adamw_8bit",    # 8-bit optimizer saves ~1 GB VRAM

        # ── Precision ────────────────────────────────────────────────
        "bf16": True,           # RTX 4070 Ti supports BF16 natively
        "fp16": False,
        "tf32": False,

        # ── Gradient checkpointing ───────────────────────────────────
        # Trades compute for memory – essential for 12 GB VRAM
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},

        # ── Flash Attention 2 ────────────────────────────────────────
        # Large speed + VRAM savings. Requires:  pip install flash-attn
        "flash_attention": True,

        # ── Logging & checkpointing ───────────────────────────────────
        "logging_steps": 10,
        "eval_steps": 100,
        "save_steps": 200,
        "save_total_limit": 3,
        "load_best_model_at_end": True,

        # ── Reproducibility ───────────────────────────────────────────
        "seed": RANDOM_SEED,

        # ── Tokenizer settings ────────────────────────────────────────
        # Llama 3.1's EOS token list – prevents runaway generation
        "special_tokens": {
            "eos_token": "<|eot_id|>",
        },
        "tokens": ["<|eot_id|>", "<|start_header_id|>", "<|end_header_id|>"],

        # ── Hugging Face hub (optional) ───────────────────────────────
        # "hub_model_id": "your-username/llama3.1-cybersec-lora",
        # "push_to_hub": True,
    }

    with CONFIG_PATH.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ Config written to: {CONFIG_PATH}")


# ─────────────────────────────────────────────
#  STEP 5 – Launch Axolotl training
# ─────────────────────────────────────────────

def run_training() -> None:
    print("\n── Launching Axolotl training ──────────────────────────────")
    print("   This will take a while. Monitor GPU with: watch -n2 nvidia-smi\n")

    cmd = [
        sys.executable, "-m", "axolotl.cli.train",
        str(CONFIG_PATH),
    ]

    print(f"  Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("\n  [ERROR] Training exited with a non-zero return code.")
        print("  Check the output above for details.")
        sys.exit(result.returncode)

    print("\n  ✓ Training complete!")
    print(f"  LoRA adapter saved in: {OUTPUT_DIR / 'checkpoints'}")


# ─────────────────────────────────────────────
#  ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Llama 3.1 LoRA Fine-tune  –  Cybersecurity Domain")
    print("  Target HW : RTX 4070 Ti (12 GB) + 32 GB RAM")
    print("=" * 60)

    print("\n── Authenticating with Hugging Face ────────────────────────")
    HUGGINGFACE_TOKEN = readFile(HUGGINGFACE_TOKEN_PATH).strip()
    login(token=HUGGINGFACE_TOKEN)
    print("  ✓ Logged in to Hugging Face")

    custom_samples = load_all_custom_samples()
    alpaca_samples = load_alpaca_samples(ALPACA_SAMPLES)
    write_merged_dataset(custom_samples, alpaca_samples)
    write_axolotl_config()

    print("\n── Pre-flight summary ──────────────────────────────────────")
    print(f"  Dataset : {MERGED_DATASET}")
    print(f"  Config  : {CONFIG_PATH}")
    print(f"  Output  : {OUTPUT_DIR / 'checkpoints'}")
    print("\nStarting training in 5 seconds  (Ctrl-C to abort)...")

    import time; time.sleep(5)
    run_training()