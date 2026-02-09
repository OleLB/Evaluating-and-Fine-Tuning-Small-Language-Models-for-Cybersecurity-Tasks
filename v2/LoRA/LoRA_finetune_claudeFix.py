import json
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from pathlib import Path
from huggingface_hub import login
import torch

HUGGINGFACE_TOKEN = "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"
login(token=HUGGINGFACE_TOKEN)

TRAINING_DATA_FOLDER = "LoRA/training_data_mistral_format"

# --------------------------
# 1. Load JSON datasets
# --------------------------

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
    """Convert Alpaca format to Mistral chat format"""
    instruction = example["instruction"].strip()
    input_text = example.get("input", "").strip()
    output = example["output"].strip()

    if input_text:
        user = f"{instruction}\n\n{input_text}"
    else:
        user = instruction

    formatted_text = f"<s>[INST] {user} [/INST] {output} </s>"
    return {"text": formatted_text}

def tokenize_fn(example):
    """Tokenize pre-formatted text"""
    tokens = tokenizer(
        example["text"], 
        truncation=True, 
        max_length=512,
        padding=False,  # Let data collator handle padding
        return_tensors=None  # Return lists, not tensors
    )
    # Create labels as a copy of input_ids
    tokens["labels"] = tokens["input_ids"][:]  # Use slice to create a proper copy
    return tokens


# =================================================================
# MAIN EXECUTION - Prevent re-execution in multiprocessing workers
# =================================================================
if __name__ == "__main__":
    # Load your custom data
    custom_data = load_jsonl_files_from_folder(TRAINING_DATA_FOLDER)

    # Load Alpaca dataset
    alpaca_data = load_dataset("tatsu-lab/alpaca")["train"]

    # --------------------------
    # 2. Standardize format - Convert everything to "text" field
    # --------------------------

    # Convert Alpaca data to same format as custom data
    alpaca_standardized = alpaca_data.map(format_alpaca_to_mistral)
    alpaca_list = alpaca_standardized.select(range(9000)).to_list()

    # --------------------------
    # 3. Merge datasets
    # --------------------------
    # Both datasets now have the same structure: {"text": "..."}
    all_data = custom_data + alpaca_list

    print(f"Total combined samples: {len(all_data)}")

    # --------------------------
    # 4. Convert to Hugging Face Dataset
    # --------------------------
    dataset = Dataset.from_list(all_data)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Training samples: {len(dataset['train'])}")
    print(f"Validation samples: {len(dataset['test'])}")

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
    
    # Ensure padding is set to left for decoder-only models
    tokenizer.padding_side = "right"  # Standard for training
    tokenizer.truncation_side = "right"

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Enable gradient checkpointing before applying LoRA
    model.config.use_cache = False  # Required for gradient checkpointing
    if hasattr(model, 'enable_input_require_grads'):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # --------------------------
    # 6. Tokenize function - Now much simpler!
    # --------------------------
    max_length = 512

    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=dataset["train"].column_names)

    # --------------------------
    # 7. LoRA configuration
    # --------------------------
    print("\nApplying LoRA configuration...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print(f"LoRA applied. Trainable parameters: {model.print_trainable_parameters()}")

    # Note: torch.compile() is incompatible with quantized models
    # Skip compilation when using 4-bit/8-bit quantization

    # --------------------------
    # 8. Training arguments
    # --------------------------
    training_args = TrainingArguments(
        output_dir="./mistral-lora-finetuned",

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,

        num_train_epochs=3,
        learning_rate=2e-4,

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        report_to="none",

        dataloader_num_workers=0,  # Set to 0 to avoid Windows multiprocessing issues
        dataloader_pin_memory=True,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Pad to multiple of 8 for efficiency
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    # --------------------------
    # 9. Start LoRA fine-tuning
    # --------------------------
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    trainer.train()

    # --------------------------
    # 10. Save LoRA adapter
    # --------------------------
    print("\nSaving model...")
    model.save_pretrained("./mistral-lora-adapter")
    tokenizer.save_pretrained("./mistral-lora-adapter")
    print("Training complete! Model saved to ./mistral-lora-adapter")