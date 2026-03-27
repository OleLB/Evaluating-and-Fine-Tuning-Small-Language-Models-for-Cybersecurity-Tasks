"""
LoRA fine-tuning of DeepSeek-Coder 1.3B Instruct
for cybersecurity terminology + code reasoning.

Training data distribution (3 000 total samples):
  50 %  Cybersecurity  (1 500)  –  general_data + training_data_explain_CVE
  30 %  Code           (  900)  –  microsoft/rStar-Coder  (synthetic_sft)
  20 %  General        (  600)  –  tatsu-lab/alpaca
"""

import json
import random
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from utils.readFile import readFile

# ──────────────────────────────────────────────
#  AUTH
# ──────────────────────────────────────────────
HUGGINGFACE_TOKEN_PATH = "keys/huggingface_token.key"
HUGGINGFACE_TOKEN = readFile(HUGGINGFACE_TOKEN_PATH).strip()
login(token=HUGGINGFACE_TOKEN)

# ──────────────────────────────────────────────
#  PATHS & CONSTANTS
# ──────────────────────────────────────────────
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"

GENERAL_DATA_DIR = Path("rag_and_lora/training_data/general_data")
CVE_DATA_DIR     = Path("rag_and_lora/training_data/training_data_explain_CVE")

CHECKPOINT_DIR    = "finetuned_models/deepseek-coder/checkpoints"
FINAL_ADAPTER_DIR = "finetuned_models/deepseek-coder-lora-adapter-final"

# Target sample counts per category
N_CYBER   = 1500   # 50 %
N_CODE    =  900   # 30 %
N_GENERAL =  600   # 20 %

MAX_LENGTH = 2048   # DeepSeek-Coder 1.3B supports 16K context; 2 048 is safe for VRAM

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
CODE_SYSTEM_PROMPT = (
    "You are an expert software engineer and coding assistant. "
    "Write clean, correct, well-commented code. Explain your reasoning "
    "step-by-step when solving algorithmic or engineering problems."
)
GENERAL_SYSTEM_PROMPT = "You are a helpful, accurate, and professional assistant."


# ──────────────────────────────────────────────
#  HELPER: Find latest checkpoint
# ──────────────────────────────────────────────
def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    p = Path(checkpoint_dir)
    if not p.exists():
        return None
    checkpoints = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    return str(checkpoints[-1])


# ──────────────────────────────────────────────
#  HELPER: Load a JSON / JSONL file
# ──────────────────────────────────────────────
def load_jsonl_or_json(path: Path) -> list[dict]:
    """Handles single-object, array, and JSONL files. Strips Windows BOM."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if text.startswith("{"):
        return [json.loads(text)]
    if text.startswith("["):
        return json.loads(text)
    # JSONL fallback
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ──────────────────────────────────────────────
#  HELPER: Alpaca-format → messages list
# ──────────────────────────────────────────────
def alpaca_to_messages(example: dict, system_prompt: str) -> dict | None:
    """Converts {instruction, input, output} → DeepSeek messages list."""
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
#  HELPER: rStar-Coder record → messages list
#
#  The microsoft/rStar-Coder synthetic_sft split can come in two shapes:
#    A) Already-formatted  {"messages": [...]}
#    B) Alpaca-style       {"instruction": ..., "output": ...}
#  This function handles both gracefully.
# ──────────────────────────────────────────────
def rstar_to_messages(example: dict) -> dict | None:
    # Shape A – pre-formatted conversation
    if "messages" in example and isinstance(example["messages"], list):
        msgs = example["messages"]
        # Inject our code system prompt if no system message exists
        if not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": CODE_SYSTEM_PROMPT}] + msgs
        else:
            # Replace whatever system prompt is there with ours
            msgs = [
                {"role": "system", "content": CODE_SYSTEM_PROMPT} if m.get("role") == "system" else m
                for m in msgs
            ]
        return {"messages": msgs}

    # Shape B – Alpaca-style
    converted = alpaca_to_messages(example, CODE_SYSTEM_PROMPT)
    if converted:
        return converted

    # Shape C – some datasets use "problem" / "solution" keys
    problem  = example.get("problem", example.get("query", "")).strip()
    solution = example.get("solution", example.get("response", "")).strip()
    if problem and solution:
        return {
            "messages": [
                {"role": "system",    "content": CODE_SYSTEM_PROMPT},
                {"role": "user",      "content": problem},
                {"role": "assistant", "content": solution},
            ]
        }

    return None  # Unrecognised format – skip


# ──────────────────────────────────────────────
#  1. Load cybersecurity data
# ──────────────────────────────────────────────
def load_custom_data(directory: Path, system_prompt: str) -> list[dict]:
    samples = []
    files   = list(directory.rglob("*.json")) + list(directory.rglob("*.jsonl"))
    if not files:
        print(f"  [WARNING] No JSON files found in {directory}")
        return samples
    for f in sorted(files):
        try:
            for rec in load_jsonl_or_json(f):
                converted = alpaca_to_messages(rec, system_prompt)
                if converted:
                    samples.append(converted)
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
    return samples


print("\n" + "=" * 60)
print("  1 / 4  Loading cybersecurity data")
print("=" * 60)

general_samples = load_custom_data(GENERAL_DATA_DIR, CYBER_SYSTEM_PROMPT) if GENERAL_DATA_DIR.exists() else []
cve_samples     = load_custom_data(CVE_DATA_DIR,     CYBER_SYSTEM_PROMPT) if CVE_DATA_DIR.exists()     else []
all_cyber       = general_samples + cve_samples
random.seed(42)
random.shuffle(all_cyber)
cyber_data = all_cyber[:N_CYBER]  # cap at target count

print(f"  General cyber samples available : {len(general_samples)}")
print(f"  CVE samples available           : {len(cve_samples)}")
print(f"  Using (capped at {N_CYBER})           : {len(cyber_data)}")


# ──────────────────────────────────────────────
#  2. Load rStar-Coder (code data)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  2 / 4  Loading rStar-Coder code data  (streaming)")
print("=" * 60)

rstar_raw    = load_dataset("microsoft/rStar-Coder", "synthetic_sft", split="train", streaming=True)
rstar_sample = list(rstar_raw.take(N_CODE))    # Pull exactly N_CODE rows without downloading the full 15 GB

code_data = []
for ex in rstar_sample:
    converted = rstar_to_messages(ex)
    if converted:
        code_data.append(converted)

print(f"  rStar-Coder samples loaded : {len(code_data)}")


# ──────────────────────────────────────────────
#  3. Load Alpaca (general data)
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  3 / 4  Loading Alpaca general data")
print("=" * 60)

alpaca_raw  = load_dataset("tatsu-lab/alpaca", split="train")
alpaca_list = []
for ex in alpaca_raw.shuffle(seed=42).select(range(N_GENERAL * 2)):  # over-sample then filter
    converted = alpaca_to_messages(ex, GENERAL_SYSTEM_PROMPT)
    if converted:
        alpaca_list.append(converted)
    if len(alpaca_list) >= N_GENERAL:
        break

print(f"  Alpaca samples loaded : {len(alpaca_list)}")


# ──────────────────────────────────────────────
#  4. Merge, shuffle & split
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  4 / 4  Merging datasets")
print("=" * 60)

all_data = cyber_data + code_data + alpaca_list
random.shuffle(all_data)

total = len(all_data)
print(f"  Cybersecurity  : {len(cyber_data):>5}  ({len(cyber_data)/total*100:.1f} %)")
print(f"  Code           : {len(code_data):>5}  ({len(code_data)/total*100:.1f} %)")
print(f"  General        : {len(alpaca_list):>5}  ({len(alpaca_list)/total*100:.1f} %)")
print(f"  ─────────────────────────")
print(f"  Total combined : {total:>5}")

dataset = Dataset.from_list(all_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42, shuffle=True)
print(f"\n  Train samples : {len(dataset['train'])}")
print(f"  Test  samples : {len(dataset['test'])}")


# ──────────────────────────────────────────────
#  Check for existing checkpoint
# ──────────────────────────────────────────────
latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR)
if latest_checkpoint:
    print(f"\n[CHECKPOINT] Resuming from : {latest_checkpoint}")
else:
    print("\n[CHECKPOINT] No checkpoint found – starting fresh.")


# ──────────────────────────────────────────────
#  5. Tokenizer
# ──────────────────────────────────────────────
print(f"\nLoading tokenizer : {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# DeepSeek-Coder uses <|EOT|> as eos; pad with eos to avoid dedicated pad-token overhead.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"   # required for causal LM


# ──────────────────────────────────────────────
#  6. Model  (4-bit QLoRA)
# ──────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print(f"Loading model    : {MODEL_NAME}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model.config.use_cache = False
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def _make_inputs_require_grad(module, inp, out):
        out.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(_make_inputs_require_grad)

print("✓ Model loaded")


# ──────────────────────────────────────────────
#  7. Tokenise with DeepSeek-Coder chat template
#
#  DeepSeek-Coder-Instruct's built-in template produces:
#    <|begin▁of▁sentence|>
#    System: {system}
#    User: {user}
#    Assistant: {assistant}<|EOT|>
#  apply_chat_template handles all special tokens automatically.
# ──────────────────────────────────────────────
def tokenize_fn(example):
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
    num_proc=None,   # Avoids Windows multiprocessing issues
)
print("✓ Tokenization complete")
print(f"  Sample token length : {len(tokenized_dataset['train'][0]['input_ids'])}")


# ──────────────────────────────────────────────
#  8. LoRA config
#
#  DeepSeek-Coder 1.3B is a standard transformer with the same
#  projection layer names as LLaMA.  r=16 is plenty for 1.3B
#  (higher r is more useful on larger models).
# ──────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,          # 2 × r
    target_modules=[        # All linear projections in DeepSeek-Coder
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
#
#  DeepSeek-Coder 1.3B is much smaller than Llama 3.1 8B, so we
#  can safely double the per-device batch size and use more epochs.
#  Effective batch = 2 × 8 = 16 samples per gradient step.
# ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,

    # ── Batch / accumulation ─────────────────
    per_device_train_batch_size=2,      # doubled vs Llama – 1.3B fits easily
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,      # effective batch = 16

    # ── Schedule ─────────────────────────────
    num_train_epochs=4,                 # 3-5 epochs recommended for small dataset
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,                  # ~5 % of steps as warm-up
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",

    # ── Precision ────────────────────────────
    bf16=True,
    fp16=False,

    # ── Memory ───────────────────────────────
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # ── Logging & saving ─────────────────────
    logging_steps=25,
    save_strategy="steps",
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=3,
    load_best_model_at_end=False,

    # ── Misc ─────────────────────────────────
    report_to="none",
    seed=42,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    label_pad_token_id=-100,   # Mask pad tokens from loss
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
print("\n" + "=" * 60)
if latest_checkpoint:
    print(f"▶ RESUMING from checkpoint : {latest_checkpoint}")
else:
    print("▶ STARTING training from scratch")
print("=" * 60 + "\n")

trainer.train(resume_from_checkpoint=latest_checkpoint)


# ──────────────────────────────────────────────
#  11. Save final adapter
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Saving final LoRA adapter...")
Path(FINAL_ADAPTER_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(FINAL_ADAPTER_DIR)
tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
print(f"✓ Adapter saved  →  {FINAL_ADAPTER_DIR}")
print(f"✓ Checkpoints    →  {CHECKPOINT_DIR}")
print("=" * 60)