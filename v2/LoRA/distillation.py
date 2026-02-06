from utils.readFile import readFile
from utils.writeFile import writeFile
from utils.querySQLite import getDescription, getRandomCVEs
from utils.queryGPT import queryGPT
from utils.checkFileExists import checkFileExists
from utils.generateRandomName import generateRandomName
import json

PROMPT_PATH = "LoRA/prompt.txt"
SYSTEM_PROMPT = readFile(PROMPT_PATH)

# Example response format
"""
[
  {
    "instruction": "Rewrite the following CVE description using professional cybersecurity terminology.",
    "response": "This vulnerability is a buffer overflow in the XYZ parser that can be exploited remotely, potentially resulting in remote code execution (RCE)."
  },
  {
    "instruction": "What type of vulnerability is described in the following CVE description?",
    "response": "The vulnerability is a buffer overflow that may lead to remote code execution."
  },
  {
    "instruction": "What is the security impact of the following vulnerability?",
    "response": "Successful exploitation could allow an unauthenticated remote attacker to execute arbitrary code on the affected system."
  },
  {
    "instruction": "Which Common Weakness Enumeration (CWE) category is most applicable to the following vulnerability?",
    "response": "This vulnerability most closely aligns with CWE-120: Buffer Copy without Checking Size of Input. This classification is inferred based on the description."
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
        "response": "This vulnerability is a buffer overflow in the XYZ parser that can be exploited remotely, potentially resulting in remote code execution (RCE)."
    }
  """
    filename = ""
    while True:
        filename = generateRandomName(10) + ".json"
        if not checkFileExists(f"LoRA/training_data/{filename}"):
            break

    for idx, item in enumerate(response_array):
        file_content = json.dumps(item, indent=4)
        file_content = file_content[:-2] + f',\n    "input": "{description}"\n}}'
        file_path = f"LoRA/training_data/{filename[:-5]}_{idx+1}.json"
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
            if 'instruction' not in item or 'response' not in item:
                print("Missing instruction or response.")
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
        description = getDescription(cve)
        prompt = preparePrompt(description)
        response, usage_info = queryGPT(prompt, "gpt-4o")
        response = cleanResponse(response)
        if validateJSONResponse(response):
            response_array = json.loads(response)
            arrayToFile(response_array, description)
        else:
            print(f"Invalid JSON response for CVE {cve}. Response: {response}")
        total_price += usage_info["total_cost"]


    print(f"Total cost for processing CVEs: ${total_price:.4f}")


def test():
    cve = "CVE-2023-4863"
    description = """MaxKB is an open-source knowledge base system that leverages language models and retrieval-augmented generation to provide intelligent responses. A remote command execution vulnerability (CVE-2023-4863) was identified in MaxKB's function library module prior to version 1.9.0. This vulnerability allowed privileged users to execute operating system commands via custom scripts. The issue was addressed and patched in version 1.9.0."""
    test_response = """
[
    {
        "instruction": "Rewrite the description in clear, professional cybersecurity language.",
        "response": "MaxKB, an open-source knowledge base system utilizing language models and retrieval-augmented generation, contained a remote command execution vulnerability in its function library module prior to version 1.9.0. This flaw permitted privileged users to execute operating system commands via custom scripts. The issue was addressed and patched in version 1.9.0."
    },
    {
        "instruction": "Identify the vulnerability type using standard terminology.",
        "response": "The described issue is a Remote Command Execution (RCE) vulnerability, which occurs when an attacker is able to remotely execute arbitrary commands on the host operating system."
    },
    {
        "instruction": "Explain the impact using correct acronyms where applicable.",
        "response": "The vulnerability allows unauthorized execution of OS commands, leading to potential breaches such as privilege escalation or unauthorized access. This can compromise the CIA triad: confidentiality, integrity, and availability of the system."
    },
    {
        "instruction": "Describe the conditions under which this vulnerability can be exploited.",
        "response": "Exploiting this vulnerability requires privileged access to the system where an attacker can execute custom scripts, enabling remote command execution on the server."
    },
    {
        "instruction": "Provide a preventive measure to mitigate this type of vulnerability.",
        "response": "Regularly update software to the latest versions to apply security patches. Implement the principle of least privilege (PoLP) to minimize the permissions necessary for user accounts, reducing the risk of privilege escalation."
    }
]
"""
    if validateJSONResponse(test_response):
        print("Valid JSON response.")
        response_array = json.loads(test_response)
        arrayToFile(response_array, description)
    else:
        print(f"Invalid JSON response for CVE {cve}.")

if __name__ == "__main__":
    main()
    # test()