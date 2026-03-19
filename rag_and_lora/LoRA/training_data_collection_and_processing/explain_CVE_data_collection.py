"""This script generates training data for a small LLM by making a bug LLM generate a summary of a CVE"""

from utils.readFile import readFile
from utils.writeFile import writeFile
from utils.querySQLite import getRandomCVEs
from utils.queryGPT import queryGPT
import json
import sqlite3

PROMPT_PATH = "prompts/cve_summary_prompt.txt"
DB_PATH = "db/cve_database.db"

def sanitizeCVEDetails(cve_details: dict) -> dict:
    """
    Sanitize CVE details to ensure all values can be properly JSON-serialized.
    Handles None values, ensures all values are strings, and validates structure.
    """
    if not cve_details:
        return None
    
    sanitized = {}
    
    for key, value in cve_details.items():
        if value is None:
            sanitized[key] = ""
        elif isinstance(value, (list, dict)):
            # Convert complex types to JSON strings
            sanitized[key] = json.dumps(value)
        else:
            # Convert to string to ensure consistency
            sanitized[key] = str(value)
    
    # Verify the sanitized dict can be JSON-serialized
    try:
        json.dumps(sanitized)
    except (TypeError, ValueError) as e:
        print(f"Error sanitizing CVE details: {e}")
        return None
    
    return sanitized


def getCVEDetails(cve_id: str) -> dict:
    """
    Fetch detailed CVE information from the database.
    Returns a dictionary with all relevant CVE details.
    """
    conn = sqlite3.connect(DB_PATH)  # Adjust path as needed
    cursor = conn.cursor()
    
    # Fetch comprehensive CVE details
    query = """
    SELECT 
        cve_id,
        description,
        cwe_name,
        cwe_description,
        label_attack,
        severity,
        known_vulnerable_software,
        mitigation
    FROM cves
    WHERE cve_id = ?
    """
    
    cursor.execute(query, (cve_id,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        cve_details = {
            'CVE_ID': result[0],
            'Description': result[1],
            'CWE Name': result[2],
            'CWE Description': result[3],
            'Associated MITRE ATT&CK Technique': result[4],
            'Severity': result[5],
            'Known Vulnerable Software': result[6],
            'Mitigation Techniques': result[7]
        }
        # Sanitize the details to ensure JSON compatibility
        return sanitizeCVEDetails(cve_details)
    
    return None


def escapeCVEDetails(cve_details: dict) -> str:
    """
    Properly escape CVE details for inclusion in JSON.
    Converts the dict to a JSON string which handles all escaping,
    then formats it for the prompt.
    """
    # Use json.dumps to properly escape all special characters
    escaped_dict = json.dumps(cve_details)
    
    # Format as it would appear in the actual system
    cve_context = f"\\n\\n\\nThe user mentioned a CVE, the following CVE information is available:\\n\\n{escaped_dict}\\n"
    
    return cve_context


def preparePrompt(cve_details: dict) -> str:
    """
    Prepare the prompt for the large LLM to generate both:
    1. A varied user prompt asking about the CVE
    2. An accurate summary based on the detailed CVE information
    """
    
    system_prompt = readFile(PROMPT_PATH)
    
    # Properly escape CVE details
    cve_context = escapeCVEDetails(cve_details)
    
    prompt = system_prompt.replace("{CVE_ID}", cve_details['CVE_ID'])
    prompt = prompt.replace("{CVE_CONTEXT}", cve_context)
    
    return prompt


def cleanResponse(response: str) -> str:
    """Extract JSON content from response and clean it."""
    # Remove markdown code blocks if present
    response = response.replace('```json', '').replace('```', '')
    response = response.strip()
    
    # Extract JSON object
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return response[start_idx:end_idx+1]
    return response


def validateAndFixJSON(response: str) -> tuple[bool, dict]:
    """
    Validate and attempt to fix JSON response.
    Returns (is_valid, data) tuple.
    """
    try:
        # First attempt: parse as-is
        data = json.loads(response)
        
        # Validate structure
        if not isinstance(data, dict):
            print("Response is not a dictionary.")
            return False, None

        required_fields = ['instruction', 'input', 'output']
        for field in required_fields:
            if field not in data:
                print(f"Missing required field: {field}")
                return False, None

        # Verify data can be re-serialized (ensures proper escaping)
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            print(f"Data cannot be re-serialized: {e}")
            return False, None

        return True, data

    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Attempting to fix JSON...")
        
        # Attempt to fix common issues
        try:
            # Try to fix unescaped quotes in strings
            # This is a simple heuristic - parse and re-serialize
            fixed_response = response
            
            # Sometimes LLMs forget to escape quotes in the middle of strings
            # We'll try a few common fixes
            
            # Replace smart quotes with regular quotes
            fixed_response = fixed_response.replace('"', '"').replace('"', '"')
            fixed_response = fixed_response.replace("'", "'").replace("'", "'")
            
            data = json.loads(fixed_response)
            
            # Validate structure
            if not isinstance(data, dict):
                return False, None
                
            required_fields = ['instruction', 'input', 'output']
            for field in required_fields:
                if field not in data:
                    return False, None
            
            print("Successfully fixed JSON!")
            return True, data
            
        except json.JSONDecodeError:
            print("Could not fix JSON automatically")
            return False, None
    
    except (TypeError, AttributeError) as e:
        print(f"Validation error: {e}")
        return False, None


def saveTrainingData(data: dict, index: int) -> None:
    """
    Save a single training data object to a JSON file.
    Uses json.dumps to ensure all special characters are properly escaped.
    """
    filename = f"cve_summary_{index:04d}.json"
    file_path = f"LoRA/training_data_explain_CVE/{filename}"
    
    # Use json.dumps with ensure_ascii=False to preserve Unicode characters
    # and indent for readability
    file_content = json.dumps(data, indent=4, ensure_ascii=False)
    
    # Verify it's valid JSON by attempting to parse it back
    try:
        json.loads(file_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Generated invalid JSON: {e}")
    
    writeFile(file_path, file_content)


def main():
    num_samples = 400
    total_price = 0.0
    successful_samples = 0
    max_retries = 3  # Retry up to 3 times for failed JSON
    
    print(f"Starting generation of {num_samples} CVE summary training samples...")
    
    # Get random CVEs (we'll cycle through them if we need more than available)
    random_cves = getRandomCVEs()
    cve_index = 0
    
    for i in range(num_samples):
        # Cycle through CVEs if we run out
        cve_id = random_cves[cve_index % len(random_cves)]
        cve_index += 1
        
        # Get detailed CVE information
        cve_details = getCVEDetails(cve_id)
        
        if not cve_details:
            print(f"Could not fetch details for CVE {cve_id}. Skipping...")
            continue
        
        # Retry logic for JSON validation failures
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            # Prepare prompt for large LLM
            prompt = preparePrompt(cve_details)
            
            # Query the large LLM
            response, usage_info = queryGPT(prompt, "gpt-4o")
            total_price += usage_info["total_cost"]
            
            # Clean and validate
            response = cleanResponse(response)
            is_valid, training_data = validateAndFixJSON(response)
            
            if is_valid:
                # Double-check by re-serializing to ensure proper escaping
                try:
                    # Save with proper JSON formatting
                    saveTrainingData(training_data, i + 1)
                    successful_samples += 1
                    print(f"✓ Generated sample {successful_samples}/{num_samples} for {cve_id}")
                    success = True
                except Exception as e:
                    print(f"Error saving training data: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying... (attempt {retry_count + 1}/{max_retries})")
            else:
                retry_count += 1
                if retry_count < max_retries:
                    print(f"✗ Invalid JSON for {cve_id}. Retrying... (attempt {retry_count + 1}/{max_retries})")
                else:
                    print(f"✗ Failed to generate valid JSON for {cve_id} after {max_retries} attempts")
                    print(f"  Last response (first 300 chars): {response[:300]}...")
        
        # Progress update every 50 samples
        if (i + 1) % 50 == 0:
            print(f"\nProgress: {i + 1}/{num_samples} samples processed")
            print(f"Successful: {successful_samples}, Cost so far: ${total_price:.4f}\n")

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"Successfully generated: {successful_samples}/{num_samples} samples")
    print(f"Success rate: {(successful_samples/num_samples)*100:.1f}%")
    print(f"Total cost: ${total_price:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()