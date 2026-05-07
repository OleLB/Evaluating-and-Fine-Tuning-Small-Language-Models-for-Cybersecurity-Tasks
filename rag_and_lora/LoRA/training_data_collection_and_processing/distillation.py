"""This script uses AI knowlege distillation to generate training data for a small LLM by generating question-answer pairs based on CVE descriptions."""

from utils.readFile import readFile
from utils.writeFile import writeFile
from utils.querySQLite import getDescription, getRandomCVEs
from utils.queryGPT import queryGPT
from utils.checkFileExists import checkFileExists
from utils.generateRandomName import generateRandomName
import json

PROMPT_PATH = "prompts/security_text_prompt.txt"
SYSTEM_PROMPT = readFile(PROMPT_PATH)

# Example response format
"""
[
  {
    "instruction": "Rewrite the following CVE description using professional cybersecurity terminology.",
    "output": "This vulnerability is a buffer overflow in the XYZ parser that can be exploited remotely, potentially resulting in remote code execution (RCE)."
  },
  {
    "instruction": "What type of vulnerability is described in the following CVE description?",
    "output": "The vulnerability is a buffer overflow that may lead to remote code execution."
  },
  {
    "instruction": "What is the security impact of the following vulnerability?",
    "output": "Successful exploitation could allow an unauthenticated remote attacker to execute arbitrary code on the affected system."
  },
  {
    "instruction": "Which Common Weakness Enumeration (CWE) category is most applicable to the following vulnerability?",
    "output": "This vulnerability most closely aligns with CWE-120: Buffer Copy without Checking Size of Input. This classification is inferred based on the description."
  }
]
"""

def arrayToFile(response_array: list, description) -> None:
    """
    Converts a JSON array to files. Each element in the array is saved as a separate file.
    Save the description in the filename for context.

    File should look like:
    {
        "instruction": "Rewrite the following CVE description using professional cybersecurity terminology.",
        "input": "A buffer overflow in the XYZ parser allows remote attackers to execute arbitrary code via crafted input.",
        "output": "This vulnerability is a buffer overflow in the XYZ parser that can be exploited remotely, potentially resulting in remote code execution (RCE)."
    }
  """
    filename = ""
    while True:
        filename = generateRandomName(10) + ".json"
        if not checkFileExists(f"rag_and_lora/LoRA/training_data/general_data/{filename}"):
            break

    for idx, item in enumerate(response_array):
        file_content = json.dumps(item, indent=4)
        file_content = file_content[:-2] + f',\n    "input": "{description}"\n}}'
        file_path = f"rag_and_lora/LoRA/training_data/general_data/{filename[:-5]}_{idx+1}.json"
        writeFile(file_path, file_content)
    # print(f"Saved {len(response_array)} files with base name '{filename[:-5]}_<index>.json'.")


def validateJSONResponse(response) -> bool:
    print("validating json")

    try:
        if isinstance(response, str):
            data = json.loads(response)
            print("JSON string parsed successfully.")
        else:
            data = response
            print("Response already parsed.")

        if not isinstance(data, list):
            print("Response is not a list.")
            return False

        for item in data:
            if not isinstance(item, dict):
                print("Item is not a dictionary.")
                return False
            if 'instruction' not in item or 'output' not in item:
                print("Missing instruction or output.")
                return False

        return True

    except (json.JSONDecodeError, TypeError) as e:
        print("JSON validation error:", e)
        return False



def preparePrompt(description: str) -> str:
    """Prepare the prompt for GPT model."""
    # insert description into prompt at "{CVE_TEXT}"
    prompt = SYSTEM_PROMPT.replace("{CVE_TEXT}", description)
    return prompt

def cleanResponse(response: str) -> str:
    """only save the content inside the first [ and last ], including them."""
    start_idx = response.find('[')
    end_idx = response.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        return response[start_idx:end_idx+1]
    

    # replace double quote with single quote
    response = response.replace('"', "'")

    # remove new lines
    response = response.replace('\n', ' ')
    return response

def main():
    randomCVEs = getRandomCVEs()
    total_price = 0.0
    for cve in randomCVEs:
        descriptions = getDescription(cve)
        prompt = preparePrompt(descriptions)
        response, usage_info = queryGPT(prompt, "gpt-4o")
        response = cleanResponse(response)
        if validateJSONResponse(response):
            response_array = json.loads(response)
            arrayToFile(response_array, descriptions)
        else:
            print(f"Invalid JSON response for CVE {cve}. Response: {response}")
        total_price += usage_info["total_cost"]


    print(f"Total cost for processing CVEs: ${total_price:.4f}")



if __name__ == "__main__":
    main()
