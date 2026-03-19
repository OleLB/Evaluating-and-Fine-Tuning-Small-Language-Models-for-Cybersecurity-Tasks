"""
The script attempts to transform the training data to a mistral compatible format.
This script transforms this format:

{
    "instruction": "Rewrite the description in clear, professional cybersecurity language.",
    "input": "wolfSSL wolfMQTT 1.9 has a heap-based buffer overflow (8 bytes) in MqttDecode_Publish (called from MqttClient_DecodePacket and MqttClient_HandlePacket).",
    "output": "The wolfSSL wolfMQTT version 1.9 contains a heap-based buffer overflow vulnerability in the MqttDecode_Publish function. This function is invoked by MqttClient_DecodePacket and MqttClient_HandlePacket."
}



To this format:

{
  "messages": [
    {"role": "user", "content": "How do I make a cup of Earl Grey tea?"},
    {"role": "assistant", "content": "To make a perfect cup of Earl Grey, boil fresh water to 100°C, steep the tea bag or leaves for 3 to 5 minutes, and add a slice of lemon if desired."}
  ]
}
"""

import json
from pathlib import Path

def convert_instruction_to_messages(data):
    """
    Convert instruction format to messages format.
    
    Args:
        data: Dictionary with 'instruction', 'input', and 'output' keys
    
    Returns:
        Dictionary with 'messages' key containing the conversation
    """
    # Combine instruction and input for the user message
    # user_content = data['instruction']
    user_content = ''
    if data.get('input') and data['input'].strip():
        user_content += f"\n\n{data['input']}"
    
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": data['output']}
        ]
    }

def process_training_data(input_folder, output_folder):
    """
    Process all JSON files in the input folder and save converted versions to output folder.
    
    Args:
        input_folder: Path to folder containing original JSON files
        output_folder: Path to folder where converted files will be saved
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get all JSON files from input folder
    input_path = Path(input_folder)
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        print(f"No JSON files found in {input_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    successful = 0
    failed = 0
    
    for json_file in json_files:
        try:
            # Read the original file
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert to messages format
            converted_data = convert_instruction_to_messages(data)
            
            # Write to output folder with same filename
            output_file = Path(output_folder) / json_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)
            
            successful += 1
            # print(f"✓ Processed: {json_file.name}")
            
        except Exception as e:
            failed += 1
            print(f"✗ Failed to process {json_file.name}: {str(e)}")
    
    print(f"\n--- Summary ---")
    print(f"Successfully converted: {successful} files")
    print(f"Failed: {failed} files")
    print(f"Output saved to: {output_folder}")

if __name__ == "__main__":
    # Define input and output folders
    input_folder = "rag_and_lora/LoRA/training_data/training_data_explain_CVE"
    output_folder = "rag_and_lora/LoRA/training_data/training_data_explain_CVE_mistral"
    
    # Process the data
    process_training_data(input_folder, output_folder)