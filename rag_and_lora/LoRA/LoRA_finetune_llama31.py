"""LoRA fine-tuning of Llama 3.1 8B Instruct for cybersecurity terminology."""

import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from pathlib import Path
from huggingface_hub import login
import torch
from utils.readFile import readFile

HUGGINGFACE_TOKEN_PATH = "keys/huggingface_token.key"
HUGGINGFACE_TOKEN = readFile(HUGGINGFACE_TOKEN_PATH).strip()
login(token=HUGGINGFACE_TOKEN)

# ──────────────────────────────────────────────
#  PATHS
# ──────────────────────────────────────────────
GENERAL_DATA_DIR  = Path("rag_and_lora/LoRA/training_data/general_data")
CVE_DATA_DIR      = Path("rag_and_lora/LoRA/training_data/training_data_explain_CVE")
CHECKPOINT_DIR    = "finetuned_models/llama31/checkpoints"
FINAL_ADAPTER_DIR = "finetuned_models/llama31-lora-adapter-final"

# ──────────────────────────────────────────────
#  SYSTEM PROMPTS
# ──────────────────────────────────────────────
CYBER_SYSTEM_PROMPT = (
    "You are a cybersecurity expert. Use precise technical terminology, "
    "standard industry acronyms (e.g., CVE, CVSS, RCE, SQLi, XSS, MITM, "
    "IoC, TTP, APT, SIEM, EDR, WAF, IAM, MFA, PKI), and professional "
    "language in all responses. Ensure accuracy and clarity when describing "
    "vulnerabilities, attack vectors, mitigations, and security concepts."
)
ALPACA_SYSTEM_PROMPT = "You are a helpful, accurate, and professional assistant."


# ──────────────────────────────────────────────
#  HELPER: Find latest checkpoint
# ──────────────────────────────────────────────
def get_latest_checkpoint(checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        return None
    checkpoints = [
        d for d in checkpoint_path.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    return str(checkpoints[-1])


# ──────────────────────────────────────────────
#  HELPER: Load a single file (object / array / jsonl)
# ──────────────────────────────────────────────
def load_jsonl_or_json(path: Path) -> list[dict]:
    """
    Handles three formats:
      { ... }           – single JSON object  (your current format, one sample per file)
      [ {...}, {...} ]  – JSON array
      {...}\n{...}      – newline-delimited JSON (JSONL)
    utf-8-sig strips the Windows BOM if present.
    """
    text = path.read_text(encoding="utf-8-sig").strip()

    if text.startswith("{"):
        return [json.loads(text)]

    if text.startswith("["):
        return json.loads(text)

    # JSONL fallback
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


# ──────────────────────────────────────────────
#  HELPER: Convert Alpaca → Llama 3.1 messages
# ──────────────────────────────────────────────
def alpaca_to_messages(example: dict, system_prompt: str) -> dict | None:
    """
    Converts {instruction, input, output} → Llama 3.1 messages list.
    Returns None for malformed / empty records.
    """
    instruction = example.get("instruction", "").strip()
    inp         = example.get("input", "").strip()
    output      = example.get("output", "").strip()

    if not instruction or not output:
        return None

    user_content = f"{instruction}\n\n{inp}" if inp else instruction

    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


# ──────────────────────────────────────────────
#  1. Load custom cybersecurity data
# ──────────────────────────────────────────────
def load_custom_data(directory: Path, system_prompt: str) -> list[dict]:
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
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")

    return samples


print("\n" + "="*60)
print("  Loading custom cybersecurity data")
print("="*60)

general_samples = load_custom_data(GENERAL_DATA_DIR, CYBER_SYSTEM_PROMPT) if GENERAL_DATA_DIR.exists() else []
cve_samples     = load_custom_data(CVE_DATA_DIR,     CYBER_SYSTEM_PROMPT) if CVE_DATA_DIR.exists()     else []
custom_data     = general_samples + cve_samples

print(f"  General samples : {len(general_samples)}")
print(f"  CVE samples     : {len(cve_samples)}")
print(f"  Total custom    : {len(custom_data)}")


# ──────────────────────────────────────────────
#  2. Load Alpaca dataset (9 000 samples)
# ──────────────────────────────────────────────
print("\nLoading Alpaca dataset...")
alpaca_raw  = load_dataset("tatsu-lab/alpaca", split="train")
alpaca_list = []
for ex in alpaca_raw.select(range(9000)):
    converted = alpaca_to_messages(ex, ALPACA_SYSTEM_PROMPT)
    if converted:
        alpaca_list.append(converted)
print(f"  Alpaca samples loaded: {len(alpaca_list)}")


# ──────────────────────────────────────────────
#  3. Merge & split
# ──────────────────────────────────────────────
all_data = custom_data + alpaca_list
print(f"\n{'='*60}")
print(f"  Total combined samples : {len(all_data)}")
print(f"    Custom (cybersec)    : {len(custom_data)}")
print(f"    Alpaca               : {len(alpaca_list)}")
print(f"{'='*60}\n")

dataset = Dataset.from_list(all_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
print(f"Train samples : {len(dataset['train'])}")
print(f"Test  samples : {len(dataset['test'])}")


# ──────────────────────────────────────────────
#  4. Check for existing checkpoint
# ──────────────────────────────────────────────
latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR)
if latest_checkpoint:
    print(f"\n[CHECKPOINT] Resuming from: {latest_checkpoint}")
else:
    print("\n[CHECKPOINT] No checkpoint found – starting fresh.")


# ──────────────────────────────────────────────
#  5. Load tokenizer
# ──────────────────────────────────────────────
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Llama 3.1 has no pad token by default – use eos token as pad.
# We tell the collator to mask pad positions with -100 so loss is not computed on them.
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"   # Required for causal LM training


# ──────────────────────────────────────────────
#  6. Load model (4-bit QLoRA)
# ──────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # RTX 4070 Ti supports BF16
)

print(f"Loading model: {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Required for gradient checkpointing + PEFT to play nicely together
model.config.use_cache = False
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

print("✓ Model loaded")


# ──────────────────────────────────────────────
#  7. Tokenise
# ──────────────────────────────────────────────
MAX_LENGTH = 1024   # Conservative for 12 GB VRAM; increase to 2048 if VRAM allows

def tokenize_fn(example):
    """
    Apply Llama 3.1's native chat template then tokenise.
    apply_chat_template automatically inserts:
      <|begin_of_text|>, <|start_header_id|>, <|eot_id|>
    so EOS / stop behaviour is preserved correctly.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


print("\nTokenizing datasets...")
tokenized_dataset = dataset.map(
    tokenize_fn,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing",
    num_proc=None,  # Avoids Windows multiprocessing issues
)
print("✓ Tokenization complete")
print(f"  Sample length: {len(tokenized_dataset['train'][0]['input_ids'])} tokens")


# ──────────────────────────────────────────────
#  8. LoRA config
# ──────────────────────────────────────────────
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,          # 2 × r is the standard starting point
    target_modules=[        # All linear projection layers in Llama 3.1
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("\nApplying LoRA...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


# ──────────────────────────────────────────────
#  9. Training arguments
# ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,

    # ── Batch / accumulation ──────────────────
    # Effective batch = 1 × 8 = 8 (safe for 12 GB VRAM)
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,

    # ── Schedule ─────────────────────────────
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",   # 8-bit optimizer saves ~1 GB VRAM

    # ── Precision ────────────────────────────
    bf16=True,
    fp16=False,

    # ── Memory ───────────────────────────────
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # ── Logging & saving ─────────────────────
    logging_steps=50,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=3,
    load_best_model_at_end=False,   # Saves memory; adapter is small anyway

    # ── Misc ─────────────────────────────────
    report_to="none",
    seed=42,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    label_pad_token_id=-100,    # Mask padding so loss is not computed on it
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)


# ──────────────────────────────────────────────
#  10. Train
# ──────────────────────────────────────────────
print("\n" + "="*60)
if latest_checkpoint:
    print(f"▶ RESUMING from checkpoint: {latest_checkpoint}")
else:
    print("▶ STARTING training from scratch")
print("="*60 + "\n")

trainer.train(resume_from_checkpoint=latest_checkpoint)


# ──────────────────────────────────────────────
#  11. Save final adapter
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("  Saving final LoRA adapter...")
Path(FINAL_ADAPTER_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
print(f"✓ Adapter saved  →  {FINAL_ADAPTER_DIR}")
print(f"✓ Checkpoints    →  {CHECKPOINT_DIR}")
print("="*60)