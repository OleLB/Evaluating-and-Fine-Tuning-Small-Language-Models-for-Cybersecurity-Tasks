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

TRANING_DATA_FOLDER = "LoRA/training_data_mistral_format"

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

def format_alpaca_for_template(example):
    """Convert Alpaca format to a structured message list"""
    user_prompt = example["instruction"]
    if example.get("input"):
        user_prompt += f"\n\n{example['input']}"
    
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": example["output"]}
        ]
    }



# Load your custom data
custom_data = load_jsonl_files_from_folder(TRANING_DATA_FOLDER)

alpaca_data = load_dataset("tatsu-lab/alpaca")["train"]
# Convert Alpaca data to same format as custom data

# Apply this to your Alpaca data
alpaca_standardized = alpaca_data.map(format_alpaca_for_template)


alpaca_list = alpaca_standardized.select(range(9000)).to_list()


# --------------------------
# 3. Merge datasets
# --------------------------
all_data = custom_data + alpaca_list
print(f"Total combined samples: {len(all_data)}")

# --------------------------
# 4. Convert to Hugging Face Dataset
# --------------------------
dataset = Dataset.from_list(all_data)
dataset = dataset.train_test_split(test_size=0.1, seed=42)  # ✅ Add validation split

# --------------------------
# 5. Load model & tokenizer
# --------------------------
model_name = "mistralai/Mistral-Nemo-Instruct-2407"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Ensure the tokenizer knows it's for training (handles special tokens correctly)
tokenizer.pad_token = "[PAD]" # Adding a dedicated pad token is safer


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

model.resize_token_embeddings(len(tokenizer))

# Enable gradient checkpointing before applying LoRA
model.config.use_cache = False  # Required for gradient checkpointing
if hasattr(model, 'enable_input_require_grads'):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

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
    # apply_chat_template handles the BOS/EOS and [INST] tokens automatically
    # It also handles tool_calls if they are in the message dict
    tokens = tokenizer.apply_chat_template(
        example["messages"], 
        truncation=True, 
        max_length=512,
        add_generation_prompt=False # False because the assistant response is already in the data
    )
    
    # Mistral-Nemo training usually masks user prompts (labels = -100) 
    # but for a simple LoRA, setting labels = input_ids is often sufficient.
    return {"input_ids": tokens, "labels": tokens}

tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset["train"].column_names)

# --------------------------
# 7. LoRA configuration
# --------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    fp16=False,
    bf16=True,
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
trainer.train(resume_from_checkpoint="./mistral-lora-finetuned/checkpoint-1124")

# --------------------------
# 10. Save LoRA adapter
# --------------------------
model.save_pretrained("./mistral-lora-adapter")
tokenizer.save_pretrained("./mistral-lora-adapter")