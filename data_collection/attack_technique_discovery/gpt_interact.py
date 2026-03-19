"""File to send CVE information to chat-gpt and get MITRE ATT&CK technique labels. Costs $14 per 10.000 requests approx."""

# from xmlrpc import client
from openai import OpenAI
from utils.queryGPT import queryGPT


# Read file to get the OpenAI API key
api_key = ""
with open("keys/openai.key", "r") as key_file:
    api_key = key_file.read().strip()

client = OpenAI(api_key=api_key)

AI_MODEL = "gpt-5-mini"

client.api_key = api_key


def calculate_cost(input_tokens, output_tokens):
    """
    Calculate the cost of the API call based on token usage.

    Args:
        input_tokens (int): Number of input tokens.
        output_tokens (int): Number of output tokens.
    Returns:
        float: The total cost in USD.
    """
    # Cost is in USD per million tokens
    cost_per_1M_input_tokens = 0.25
    cost_per_1M_output_tokens = 2.0

    total_cost = (input_tokens / 1000000) * cost_per_1M_input_tokens + (output_tokens / 1000000) * cost_per_1M_output_tokens
    return total_cost


def get_attack_technique(CVE):
    """
    Sends a CVE to chat-gpt and retrieves the MITRE ATT&CK technique label.

    Args:
        CVE (str): The CVE data.
    Returns:
        str: The model's response containing the technique label.
    """
    prompt = f"""
    You are a cybersecurity expert. Your job is to map CVEs to MITRE ATT&CK techniques. Use the following rules to decide between multiple matching techniques:
    1. Prioritize techniques that are more specific to the CVE.
    2. When deciding between multiple techniques, select the technique that would come first in the cyber kill chain.

    Do NOT:
        provide a sub-technique.

    Do:
        Only provide the technique ID of the most suitable technique, nothing else.
        Think hard before answering.

    Answer format:
        Txxxx

    Here are the CVE details, it includes the id, description, cwe mapping, vulnerable software, and severity:
    {CVE}.
    """
    response = client.responses.create(
        model=AI_MODEL,
        input=prompt
    )

    label = response.output_text.strip()

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens

    total_cost = calculate_cost(input_tokens, output_tokens)

    return label, {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": total_cost
    }


if __name__ == "__main__":
    sample_cve = '''
{"cve_id": "CVE-2016-20008", "description": "The REST/JSON project 7.x-1.x for Drupal allows session enumeration, aka SA-CONTRIB-2016-033. NOTE: This project is not covered by Drupal's security advisory policy.", "exploit_instruction": "Not implemented", "cwe_name": "NVD-CWE-Other", "cwe_description": "N/A", "label_attack": "Mitre ATT&CK mapping not implemented", "severity": "7.5", "known_vulnerable_software": "[\"{'operator': 'OR', 'negate': False, 'cpeMatch': [{'vulnerable': True, 'criteria': 'cpe:2.3:a:rest\\\\\\\\/json_project:rest\\\\\\\\/json:*:*:*:*:*:drupal:*:*', 'versionEndIncluding': '7.x-1.5', 'matchCriteriaId': '82EE33D8-72D6-49AC-8367-58A829C53152'}]}\"]", "cpe_list": "[\"cpe:2.3:a:rest\\\\/json_project:rest\\\\/json:*:*:*:*:*:drupal:*:*\"]"}
'''
    response, token_usage = get_attack_technique(sample_cve)
    print("Model response:", response)
    print("Cost:", token_usage["total_cost"])