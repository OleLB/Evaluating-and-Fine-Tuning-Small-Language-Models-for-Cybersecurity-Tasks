import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model

# --------------------------
# 1. Load JSON datasets
# --------------------------
def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

custom_data = load_jsonl("custom_data.jsonl")   # 1000 samples
alpaca_data = load_jsonl("alpaca_data.jsonl")   # 9000 samples

# --------------------------
# 2. Merge datasets
# --------------------------
all_data = custom_data + alpaca_data

# --------------------------
# 3. Convert to Hugging Face Dataset
# --------------------------
dataset = Dataset.from_list(all_data)

# --------------------------
# 4. Load model & tokenizer
# --------------------------
model_name = "mistral-nemo-small"  # adjust to your model
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# --------------------------
# 5. Tokenize function
# --------------------------
max_length = 512

def format_prompt(example):
    if example.get("input"):
        prompt = f"Input: {example['input']}\nInstruction: {example['instruction']}\nResponse: {example['response']}"
    else:
        prompt = f"Instruction: {example['instruction']}\nResponse: {example['response']}"
    return prompt

def tokenize_fn(example):
    prompt = format_prompt(example)
    tokens = tokenizer(prompt, truncation=True, max_length=max_length)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_fn)

# --------------------------
# 6. LoRA configuration
# --------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],  # typical for Mistral-style attention
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# --------------------------
# 7. Training arguments
# --------------------------
training_args = TrainingArguments(
    output_dir="./mistral-lora-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# --------------------------
# 8. Start LoRA fine-tuning
# --------------------------
trainer.train()

# --------------------------
# 9. Save LoRA adapter
# --------------------------
model.save_pretrained("./mistral-lora-adapter")
tokenizer.save_pretrained("./mistral-lora-adapter")
