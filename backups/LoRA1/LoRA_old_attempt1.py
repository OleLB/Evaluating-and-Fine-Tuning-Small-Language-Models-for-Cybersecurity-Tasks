import json
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from pathlib import Path
from huggingface_hub import login
import torch

HUGGINGFACE_TOKEN = "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"
login(token=HUGGINGFACE_TOKEN)

# --------------------------
# 1. Load JSON datasets
# --------------------------

TRANING_DATA_FOLDER = "LoRA/training_data"

def load_json_files_from_folder(folder_path):
    data = []
    folder = Path(folder_path)
    
    # Get all .json files in the folder
    json_files = list(folder.glob("*.json"))
    
    print(f"Found {len(json_files)} JSON files in {folder_path}")
    
    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data.append(json.load(f))
        except json.JSONDecodeError as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(data)} samples")
    return data

# Load your custom data
custom_data = load_json_files_from_folder(TRANING_DATA_FOLDER)

alpaca_data = load_dataset("tatsu-lab/alpaca")["train"]  # ✅ Get the 'train' split

# --------------------------
# 2. Standardize format
# --------------------------
# Alpaca uses "output" instead of "response"
def standardize_alpaca(example):
    return {
        "instruction": example["instruction"],
        "input": example.get("input", ""),
        "output": example["output"]  # ✅ Map output to response
    }

alpaca_standardized = alpaca_data.map(standardize_alpaca)
alpaca_list = alpaca_standardized.select(range(9000)).to_list()  # Take 9000 samples

# --------------------------
# 3. Merge datasets
# --------------------------
all_data = custom_data + alpaca_list  # ✅ Both are lists now

# --------------------------
# 4. Convert to Hugging Face Dataset
# --------------------------
dataset = Dataset.from_list(all_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)  # ✅ Add validation split

# --------------------------
# 5. Load model & tokenizer
# --------------------------
model_name = "mistralai/Mistral-Nemo-Instruct-2407"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # ✅ Pass quantization config
    device_map="auto",
    trust_remote_code=True
    # Remove: load_in_8bit, torch_dtype, or dtype
)

# --------------------------
# 6. Tokenize function with proper masking
# --------------------------
max_length = 512

def format_prompt(example):
    if example.get("input") and example["input"].strip():
        prompt = f"Input: {example['input']}\nInstruction: {example['instruction']}\noutput: {example['output']}"
    else:
        prompt = f"Instruction: {example['instruction']}\noutput: {example['output']}"
    return prompt

def tokenize_fn(example):
    prompt = format_prompt(example)
    tokens = tokenizer(
        prompt, 
        truncation=True, 
        max_length=max_length,
        padding=False  # ✅ Let data collator handle padding
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset["train"].column_names)

# --------------------------
# 7. LoRA configuration
# --------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# --------------------------
# 8. Training arguments
# --------------------------
training_args = TrainingArguments(
    output_dir="./mistral-lora-finetuned",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,  # ✅ Added eval batch size
    gradient_accumulation_steps=16,
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",  # ✅ Added evaluation
    save_total_limit=2,
    report_to="none",
    gradient_checkpointing=True,
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer, 
    pad_to_multiple_of=8, 
    return_tensors="pt",
    label_pad_token_id=-100  # ✅ Proper padding for labels
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],  # ✅ Added eval dataset
    data_collator=data_collator,
)

# --------------------------
# 9. Start LoRA fine-tuning
# --------------------------
trainer.train()

# --------------------------
# 10. Save LoRA adapter
# --------------------------
model.save_pretrained("./mistral-lora-adapter")
tokenizer.save_pretrained("./mistral-lora-adapter")