import ollama

sample_cve = '''
{"cve_id": "CVE-2016-20008", "description": "The REST/JSON project 7.x-1.x for Drupal allows session enumeration, aka SA-CONTRIB-2016-033. NOTE: This project is not covered by Drupal's security advisory policy.", "exploit_instruction": "Not implemented", "cwe_name": "NVD-CWE-Other", "cwe_description": "N/A", "label_attack": "Mitre ATT&CK mapping not implemented", "severity": "7.5", "known_vulnerable_software": "[\"{'operator': 'OR', 'negate': False, 'cpeMatch': [{'vulnerable': True, 'criteria': 'cpe:2.3:a:rest\\\\\\\\/json_project:rest\\\\\\\\/json:*:*:*:*:*:drupal:*:*', 'versionEndIncluding': '7.x-1.5', 'matchCriteriaId': '82EE33D8-72D6-49AC-8367-58A829C53152'}]}\"]", "cpe_list": "[\"cpe:2.3:a:rest\\\\/json_project:rest\\\\/json:*:*:*:*:*:drupal:*:*\"]"}
'''

def chat_with_model(model_name, user_message):
    """
    Sends a message to the specified model and returns the response.

    Args:
        model_name (str): The name of the model to interact with.
        user_message (str): The message to send to the model.

    Returns:
        str: The content of the model's response.
    """
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": user_message}]
        )
        return response.get("message", {}).get("content", "No response content found.")
    except Exception as e:
        return f"An error occurred: {e}"


def collect_attack_techniques(CVE):
    """
    Collects attack techniques for a given CVE using ollama3.1, deepseek-r1, and qwen3. Then compares their responses.

    Args:
        CVE (str): The CVE data.

    Returns:
        str: The model's response containing attack techniques.
    """
    models = ["llama3.1:8b", "deepseek-r1", "qwen3:8b"]
    prompt = f"""
    You are a cybersecurity expert. Your job is to map CVEs to MITRE ATT&CK techniques. Use the following rules to decide between multiple matching techniques:
    1. Prioritize techniques that are more specific to the CVE.
    2. When deciding between multiple techniques, select the technique that would come first in the cyber kill chain.

    Do NOT:
        provide a sub-technique.

    Do:
        Only provide the technique ID of the most suitable technique, nothing else.

    Answer format:
        Txxxx
    Here are the CVE details, it includes the id, description, cwe mapping, vulnerable software, and severity:
    {CVE}.
    """
    responses = []
    for model in models:
        response = chat_with_model(model, prompt)
        responses.append((model, response))

    return responses


# Example usage
if __name__ == "__main__":
    attack_techniques = collect_attack_techniques(sample_cve)
    for model, techniques in attack_techniques:
        print(f"Model: {model}\nTechniques: {techniques}\n")
