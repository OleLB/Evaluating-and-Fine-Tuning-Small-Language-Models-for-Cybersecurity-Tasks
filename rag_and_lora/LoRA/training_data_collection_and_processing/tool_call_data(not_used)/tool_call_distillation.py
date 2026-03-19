from utils.readFile import readFile
from utils.writeFile import writeFile
from utils.querySQLite import getRandomCVEs, getCVEInfo
from utils.queryGPT import queryGPT
from utils.checkFileExists import checkFileExists
from utils.generateRandomName import generateRandomName
import json
import re

PROMPT_PATH = "prompts/tool_call_prompt.txt"

# Check if prompt file exists, if not create it
if not checkFileExists(PROMPT_PATH):
    print(f"Creating prompt file at {PROMPT_PATH}")
    prompt_template = """You are generating training data for a small language model that needs to learn when and how to call a tool for vulnerability searches.

Your task: Generate 3-5 diverse user queries that would require searching for vulnerabilities in specific software.

Given information:
- Software: {SOFTWARE}
- Version: {VERSION}
- CVE ID: {CVE_ID}
- Severity: {SEVERITY}

Generate user queries in these styles:
1. DIRECT: Straightforward requests for vulnerability information
2. NATURAL: Casual questions about security concerns
3. CONVERSATIONAL: Questions framed as someone asking for help

Requirements:
- Each query should clearly mention the software name and version
- Vary the phrasing naturally
- Use different query styles (direct, natural, conversational)
- Keep queries realistic - how a real user would ask
- DO NOT include the CVE ID in the query (users don't know it yet)

Output format: JSON array of objects.
Each object must have:
- "user_query": The question a user would ask
- "query_type": One of "direct", "natural", or "conversational"

Example output structure:
[
  {
    "user_query": "Find vulnerabilities for Apache HTTP Server version 2.4.49",
    "query_type": "direct"
  },
  {
    "user_query": "Is Apache HTTP Server 2.4.49 safe to use in production?",
    "query_type": "natural"
  },
  {
    "user_query": "We're running Apache 2.4.49, should I be worried about security issues?",
    "query_type": "conversational"
  }
]

Generate 3-5 examples now in valid JSON format. Only output the JSON array, nothing else."""

    writeFile(PROMPT_PATH, prompt_template)

SYSTEM_PROMPT = readFile(PROMPT_PATH)

PROCESSED_CVES_FILE = "LoRA/processed_cves.json"
TARGET_SAMPLES = 400
BATCH_SIZE = 200

def loadProcessedCVEs() -> set:
    """Load the set of already processed CVEs from file."""
    try:
        if checkFileExists(PROCESSED_CVES_FILE):
            content = readFile(PROCESSED_CVES_FILE)
            return set(json.loads(content))
        return set()
    except Exception as e:
        print(f"Error loading processed CVEs: {e}")
        return set()

def saveProcessedCVEs(processed_cves: set) -> None:
    """Save the set of processed CVEs to file."""
    try:
        content = json.dumps(list(processed_cves), indent=2)
        writeFile(PROCESSED_CVES_FILE, content)
    except Exception as e:
        print(f"Error saving processed CVEs: {e}")

def extractVersionFromCPE(cpe_list_str: str) -> str:
    """
    Extract version number from CPE list.
    CPE format: cpe:2.3:a:vendor:product:version:...
    Returns version if found, None otherwise.
    """
    try:
        # Parse the CPE list (it's stored as a string representation of a list)
        cpe_list = eval(cpe_list_str) if isinstance(cpe_list_str, str) else cpe_list_str
        
        for cpe in cpe_list:
            # Split CPE string by colons
            parts = cpe.split(':')
            if len(parts) >= 6:
                version = parts[5]
                # Check if version matches pattern X.Y or X.Y.Z (not wildcards like *)
                if re.match(r'^\d+\.\d+(\.\d+)*$', version):
                    return version
    except Exception as e:
        print(f"Error parsing CPE: {e}")
    
    return None

def extractSoftwareFromCPE(cpe_list_str: str) -> str:
    """
    Extract software name from CPE list.
    CPE format: cpe:2.3:a:vendor:product:version:...
    """
    try:
        cpe_list = eval(cpe_list_str) if isinstance(cpe_list_str, str) else cpe_list_str
        
        for cpe in cpe_list:
            parts = cpe.split(':')
            if len(parts) >= 5:
                vendor = parts[3]
                product = parts[4]
                # Format as "Vendor Product" with proper capitalization
                software_name = f"{vendor.replace('_', ' ').title()} {product.replace('_', ' ').title()}"
                return software_name.strip()
    except Exception as e:
        print(f"Error parsing CPE: {e}")
    
    return "Unknown Software"

def hasValidVersion(cve_info: dict) -> bool:
    """Check if CVE has a valid version number in CPE list."""
    cpe_list = cve_info.get('CPE_List', '')
    version = extractVersionFromCPE(cpe_list)
    return version is not None

def preparePrompt(cve_data: dict) -> str:
    """Prepare the prompt for GPT model to generate tool-calling training data."""
    # print("Preparing prompt for CVE:", cve_data['cve_id'])
    # print(f"System prompt {SYSTEM_PROMPT[:100]}...")  # Print first 100 chars of system prompt for verification
    # input("Press Enter to continue...")
    prompt = SYSTEM_PROMPT.replace("{SOFTWARE}", cve_data['software'])
    prompt = prompt.replace("{VERSION}", cve_data['version'])
    prompt = prompt.replace("{CVE_ID}", cve_data['cve_id'])
    prompt = prompt.replace("{SEVERITY}", cve_data['severity'])

    # print(f"Completed prompt construction")
    # print(prompt)
    # input("Press Enter to continue...")  # Pause to verify prompt before sending to GPT
    # Verify prompt was created properly
    if not prompt or len(prompt) < 100:
        raise ValueError(f"Prompt creation failed. Length: {len(prompt)}")
    
    return prompt

def cleanResponse(response: str) -> str:
    """Extract JSON array from response."""
    start_idx = response.find('[')
    end_idx = response.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return response[start_idx:end_idx+1]
    return response

def validateToolCallResponse(response) -> bool:
    """Validate that the response contains proper tool-calling examples."""
    try:
        if isinstance(response, str):
            data = json.loads(response)
        else:
            data = response

        if not isinstance(data, list):
            print("Response is not a list.")
            return False

        for item in data:
            if not isinstance(item, dict):
                print("Item is not a dictionary.")
                return False
            if 'user_query' not in item or 'query_type' not in item:
                print("Missing user_query or query_type.")
                return False

        return True

    except (json.JSONDecodeError, TypeError) as e:
        print("JSON validation error:", e)
        return False

def convertToJSONL(cve_data: dict, llm_examples: list) -> str:
    """
    Convert CVE data and LLM examples into JSONL format for training.
    Each line is a complete conversation with tool calling.
    """
    jsonl_lines = []
    
    software = cve_data['software']
    version = cve_data['version']
    
    for example in llm_examples:
        user_query = example.get('user_query', '')
        
        # Create the training example in Mistral format
        training_example = {
            "messages": [
                {
                    "role": "user",
                    "content": user_query
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "type": "function",
                            "function": {
                                "name": "vector_database_retrieval",
                                "arguments": json.dumps({
                                    "query": f"vulnerability for {software} version {version}"
                                })
                            }
                        }
                    ]
                }
            ]
        }
        
        # Convert to JSON string (one line per example)
        jsonl_lines.append(json.dumps(training_example, ensure_ascii=False))
    
    return '\n'.join(jsonl_lines)

def saveToFile(cve_data: dict, llm_examples: list) -> None:
    """Save training data to JSONL file with random name."""
    filename = ""
    while True:
        filename = generateRandomName(10) + ".jsonl"
        if not checkFileExists(f"LoRA/training_data_tool_calls/{filename}"):
            break
    
    jsonl_content = convertToJSONL(cve_data, llm_examples)
    file_path = f"LoRA/training_data_tool_calls/{filename}"
    writeFile(file_path, jsonl_content)
    print(f"Saved {len(llm_examples)} tool-calling examples to {filename}")

def main():
    processed_cves = loadProcessedCVEs()
    total_samples = 0
    total_price = 0.0
    successful = 0
    failed = 0
    rejected_no_version = 0
    
    print(f"Starting tool-call training data generation...")
    print(f"Target: {TARGET_SAMPLES} samples")
    print(f"Already processed: {len(processed_cves)} CVEs")
    print()
    
    while total_samples < TARGET_SAMPLES:
        print(f"\n=== Pulling batch of {BATCH_SIZE} CVEs ===")
        print(f"Current progress: {total_samples}/{TARGET_SAMPLES} samples")
        
        # Pull random CVE IDs
        random_cve_ids = getRandomCVEs(limit=BATCH_SIZE)
        
        batch_processed = 0
        
        for cve_id in random_cve_ids:
            # Skip if already processed
            if cve_id in processed_cves:
                continue
            
            # Get full CVE information
            cve_info = getCVEInfo(cve_id)
            
            # Check if valid response
            if isinstance(cve_info, str):  # Error message
                print(f"  ✗ Error getting info for {cve_id}: {cve_info}")
                processed_cves.add(cve_id)
                failed += 1
                continue
            
            # Check if it has a valid version number
            if not hasValidVersion(cve_info):
                rejected_no_version += 1
                processed_cves.add(cve_id)
                continue
            
            # Extract data from CVE info
            version = extractVersionFromCPE(cve_info['CPE_List'])
            software = extractSoftwareFromCPE(cve_info['CPE_List'])
            severity = cve_info['Severity']
            
            cve_data = {
                'cve_id': cve_id,
                'software': software,
                'version': version,
                'severity': severity
            }
            
            print(f"\nProcessing: {cve_id}")
            print(f"  Software: {software} {version}")
            print(f"  Severity: {severity}")
            
            try:
                # Generate training examples
                prompt = preparePrompt(cve_data)
                
                # Debug: verify prompt
                print(f"  Prompt length: {len(prompt)}")
                
                response, usage_info = queryGPT(prompt, "gpt-4o")
                response = cleanResponse(response)
                
                if validateToolCallResponse(response):
                    response_array = json.loads(response)
                    saveToFile(cve_data, response_array)
                    
                    # Update counters
                    num_examples = len(response_array)
                    total_samples += num_examples
                    successful += 1
                    batch_processed += 1
                    
                    print(f"  ✓ Generated {num_examples} examples (Total: {total_samples}/{TARGET_SAMPLES})")
                else:
                    print(f"  ✗ Invalid response")
                    print(f"  Response preview: {response[:200]}...")
                    failed += 1
                
                # Mark as processed
                processed_cves.add(cve_id)
                total_price += usage_info["total_cost"]
                
            except Exception as e:
                print(f"  ✗ Error processing CVE: {e}")
                import traceback
                traceback.print_exc()
                failed += 1
                processed_cves.add(cve_id)
            
            # Save progress periodically
            if batch_processed % 10 == 0:
                saveProcessedCVEs(processed_cves)
            
            # Check if we've reached target
            if total_samples >= TARGET_SAMPLES:
                print(f"\n✓ Target of {TARGET_SAMPLES} samples reached!")
                break
        
        # Save processed CVEs after each batch
        saveProcessedCVEs(processed_cves)
        
        print(f"\nBatch complete: {batch_processed} CVEs processed")
        
        # Safety check to avoid infinite loop
        if batch_processed == 0:
            print("\n⚠ Warning: No new CVEs processed in this batch.")
            print("This might mean you've exhausted CVEs with version numbers.")
            user_input = input("Continue? (y/n): ")
            if user_input.lower() != 'y':
                break
    
    print(f"\n{'='*50}")
    print(f"=== Final Summary ===")
    print(f"{'='*50}")
    print(f"Total samples generated: {total_samples}/{TARGET_SAMPLES}")
    print(f"CVEs successfully processed: {successful}")
    print(f"CVEs failed: {failed}")
    print(f"CVEs rejected (no version): {rejected_no_version}")
    print(f"Total CVEs processed: {len(processed_cves)}")
    print(f"Total cost: ${total_price:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()