"""This script was used to LoRA fine-tune mistral-nemo"""

import json
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from pathlib import Path
from huggingface_hub import login
import torch

HUGGINGFACE_TOKEN_PATH = "keys/huggingface_token.key"
with open(HUGGINGFACE_TOKEN_PATH, "r") as f:
    HUGGINGFACE_TOKEN = f.read().strip()
login(token=HUGGINGFACE_TOKEN)

# --------------------------
# Helper: Find latest checkpoint
# --------------------------
def get_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the output directory.
    
    Returns:
        str: Path to latest checkpoint, or None if no checkpoints found
    """
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None
    
    # Find all checkpoint directories
    checkpoints = [
        d for d in checkpoint_path.iterdir() 
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]
    
    if not checkpoints:
        return None
    
    # Sort by checkpoint number (e.g., checkpoint-500 -> 500)
    checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
    
    latest = checkpoints[-1]
    return str(latest)


# --------------------------
# 1. Load JSON datasets
# --------------------------

TRAINING_DATA_FOLDER = "rag_and_lora/LoRA/training_data/training_data_mistral_all"

def load_jsonl_files_from_folder(folder_path):
    """Load JSONL files (one JSON object per line)"""
    data = []
    folder = Path(folder_path)
    
    # Get all .jsonl files in the folder
    jsonl_files = list(folder.glob("*.jsonl"))
    
    print(f"Found {len(jsonl_files)} JSONL files in {folder_path}")
    
    for jsonl_file in jsonl_files:
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        data.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error loading {jsonl_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(data)} custom samples")
    return data


def format_alpaca_to_mistral(example):
    """
    Convert Alpaca format to Mistral message format.
    """
    user_prompt = example["instruction"]
    
    # Add input if it exists and is not empty
    if example.get("input") and example["input"].strip():
        user_prompt += f"\n\n{example['input']}"
    
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["output"]}
        ]
    }


# --------------------------
# 2. Check for existing checkpoint
# --------------------------
CHECKPOINT_DIR = "./mistral-lora-finetuned"
latest_checkpoint = get_latest_checkpoint(CHECKPOINT_DIR)

if latest_checkpoint:
    print("\n" + "="*60)
    print("🔄 CHECKPOINT FOUND")
    print("="*60)
    print(f"Found existing checkpoint: {latest_checkpoint}")
    print("Training will resume from this checkpoint")
    print("="*60 + "\n")
else:
    print("\n" + "="*60)
    print("🆕 STARTING FRESH TRAINING")
    print("="*60)
    print("No existing checkpoints found")
    print("Training will start from the beginning")
    print("="*60 + "\n")


# --------------------------
# 3. Load datasets
# --------------------------

# Load your custom data (already in Mistral format with tool calls)
custom_data = load_jsonl_files_from_folder(TRAINING_DATA_FOLDER)

# Load Alpaca data
print("\nLoading Alpaca dataset...")
alpaca_data = load_dataset("tatsu-lab/alpaca")["train"]

# Convert Alpaca data to Mistral format
print("Converting Alpaca data to Mistral format...")
alpaca_standardized = alpaca_data.map(format_alpaca_to_mistral)

# Select subset and convert to list
alpaca_list = alpaca_standardized.select(range(9000)).to_list()

print(f"Alpaca samples converted: {len(alpaca_list)}")

# --------------------------
# 4. Merge datasets
# --------------------------
all_data = custom_data + alpaca_list
print(f"\n{'='*60}")
print(f"Total combined samples: {len(all_data)}")
print(f"  - Custom tool-call samples: {len(custom_data)}")
print(f"  - Alpaca samples: {len(alpaca_list)}")
print(f"{'='*60}\n")

# --------------------------
# 5. Convert to Hugging Face Dataset
# --------------------------
dataset = Dataset.from_list(all_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# --------------------------
# 6. Load model & tokenizer
# --------------------------
model_name = "mistralai/Mistral-Nemo-Instruct-2407"
print(f"\nLoading tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the tokenizer knows it's for training (handles special tokens correctly)
tokenizer.pad_token = "[PAD]"

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print(f"Loading model: {model_name}")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model.resize_token_embeddings(len(tokenizer))

# Enable gradient checkpointing before applying LoRA
model.config.use_cache = False
if hasattr(model, 'enable_input_require_grads'):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

print("✓ Model loaded successfully")

# --------------------------
# 7. Tokenize function
# --------------------------
max_length = 512

def tokenize_fn(example):
    """
    Tokenize using apply_chat_template.
    """
    # Apply chat template
    result = tokenizer.apply_chat_template(
        example["messages"], 
        truncation=True, 
        max_length=max_length,
        add_generation_prompt=False,
        return_dict=False,
        tokenize=True
    )
    
    # Handle edge case where dict might still be returned
    if isinstance(result, dict):
        tokens = result['input_ids']
    else:
        tokens = result
    
    return {
        "input_ids": tokens, 
        "labels": tokens
    }


print("\nTokenizing datasets...")
tokenized_dataset = dataset.map(
    tokenize_fn, 
    remove_columns=dataset["train"].column_names,
    desc="Tokenizing",
    num_proc=None  # No multiprocessing to avoid Windows issues
)

print("✓ Tokenization complete")
print(f"Sample tokenized length: {len(tokenized_dataset['train'][0]['input_ids'])} tokens")

# --------------------------
# 8. LoRA configuration
# --------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

print("\nApplying LoRA configuration...")
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --------------------------
# 9. Training arguments
# --------------------------
training_args = TrainingArguments(
    output_dir=CHECKPOINT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=False,
    bf16=True,
    logging_steps=50,
    save_strategy="steps",        # ← Save every N steps (more frequent than epoch)
    save_steps=100,               # ← Save checkpoint every 100 steps
    eval_strategy="steps",        # ← Evaluate every N steps
    eval_steps=100,               # ← Evaluate every 100 steps
    save_total_limit=3,           # ← Keep only last 3 checkpoints (saves disk space)
    load_best_model_at_end=False, # ← Don't load best model (saves memory)
    report_to="none",
    gradient_checkpointing=True,
    resume_from_checkpoint=latest_checkpoint,  # ← AUTO-RESUME from latest checkpoint
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    pad_to_multiple_of=8, 
    return_tensors="pt",
    label_pad_token_id=-100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

# --------------------------
# 10. Start LoRA fine-tuning
# --------------------------
print("\n" + "="*60)
if latest_checkpoint:
    print("▶ RESUMING training from checkpoint...")
    print(f"Checkpoint: {latest_checkpoint}")
else:
    print("▶ STARTING training from scratch...")
print("="*60 + "\n")

# Train (will automatically resume if checkpoint exists)
trainer.train(resume_from_checkpoint=latest_checkpoint)

# --------------------------
# 11. Save final LoRA adapter
# --------------------------
print("\n" + "="*60)
print("💾 Saving final LoRA adapter...")
print("="*60)
model.save_pretrained("./mistral-lora-adapter-final")
tokenizer.save_pretrained("./mistral-lora-adapter-final")
print("✓ Training complete!")
print(f"✓ Final adapter saved to: ./mistral-lora-adapter-final")
print(f"✓ Checkpoints saved in: {CHECKPOINT_DIR}")
print("="*60)