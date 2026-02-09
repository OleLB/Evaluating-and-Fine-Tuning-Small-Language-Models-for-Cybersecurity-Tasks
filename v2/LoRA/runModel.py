from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

def load_finetuned_model():
    """Load the fine-tuned model with LoRA adapter"""
    base_model_name = "mistralai/Mistral-Nemo-Instruct-2407"
    
    # Same quantization config as training
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, "./mistral-lora-adapter")
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text="", max_length=512):
    """Generate a response from the model"""
    if input_text:
        prompt = f"Input: {input_text}\nInstruction: {instruction}\noutput:"
    else:
        prompt = f"Instruction: {instruction}\noutput:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Main execution
if __name__ == "__main__":
    model, tokenizer = load_finetuned_model()
    
    print("\n" + "="*50)
    print("Model loaded! Ready for inference.")
    print("="*50 + "\n")
    
    # Example usage
    instruction = "Explain what SQL injection is and how to prevent it"
    response = generate_response(model, tokenizer, instruction)
    
    print(f"Instruction: {instruction}")
    print(f"\nResponse:\n{response}")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive mode (type 'quit' to exit)")
    print("="*50 + "\n")
    
    while True:
        instruction = input("\nEnter instruction: ")
        if instruction.lower() == 'quit':
            break
        
        input_text = input("Enter input (press Enter to skip): ")
        response = generate_response(model, tokenizer, instruction, input_text)
        print(f"\nResponse:\n{response}\n")