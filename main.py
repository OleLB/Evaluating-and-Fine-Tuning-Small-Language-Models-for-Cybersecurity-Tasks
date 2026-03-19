"""
This script is the entry point for interacting with the AI RAG system.

It will:
- Take a user query as input.
- If the prompt contains a CVE on the form CVE-YYYY-NNNN, it will search the database for relevant information.
- The prompt gets passed to an AI pipeline function, which will determine if a tool call to the vector database is needed (if the user is asking about vulnerabilities for a specific software version).
- The relevant information from the database will be passed to the AI model as context, and the model will generate a response to the user based on this information.
- Finally, it will return the answer to the user.
"""

import ollama
import re
from rag_and_lora.qdrant.startQdrant import start_qdrant, check_qdrant_available
from rag_and_lora.RAG_core import getCVEInfo, qdrant_RAG
from utils.cleanCVEdata import clean_cve_data
from utils.readFile import readFile


CVE_PATTERN = r"CVE-\d{4}-\d{4,7}"
DEFAULT_AI_MODEL = "mistral-nemo-cve2:latest"
RAG_MODEL = "mistral-nemo:latest"


# Reading prompts from files
VULNERABILITY_DISCOVERY_PROMPT_PATH = "prompts/vulnerability_discovery_prompt.txt"
VULNERABILITY_DISCOVERY_PROMPT = readFile(VULNERABILITY_DISCOVERY_PROMPT_PATH)

CVE_DESCRIPTOR_PROMPT_PATH = "prompts/cve_descriptor_prompt.txt"
CVE_DESCRIPTOR_PROMPT = readFile(CVE_DESCRIPTOR_PROMPT_PATH)

GENERALL_INFORMATION_PROMPT_PATH = "prompts/general_information_prompt.txt"
GENERALL_INFORMATION_PROMPT = readFile(GENERALL_INFORMATION_PROMPT_PATH)


def build_context_block(cve_data: dict, rag_output: str) -> str:
    parts = ["### VULNERABILITY CONTEXT DATABASE ###"]

    if cve_data:
        block = "## TARGET CVE ANALYSIS\n"

        # Scalar fields — one line each
        for field in ["CVE_ID", "Severity", "CWE Name",
                      "Associated MITRE ATT&CK Technique", "MITRE ATT&CK Technique Name"]:
            if field in cve_data:
                block += f"- **{field}**: {cve_data[field]}\n"

        # Long text fields — given breathing room
        for field in ["Description", "CWE Description"]:
            if field in cve_data:
                block += f"\n**{field}**:\n{cve_data[field]}\n"

        # List fields
        for field in ["Affected Platforms", "Mitigation Techniques"]:
            items = cve_data.get(field, [])
            if items:
                block += f"\n**{field}**:\n"
                block += "\n".join(f"  - {i}" for i in items) + "\n"

        parts.append(block)

    if rag_output:
        parts.append(f"## VULNERABILITY DISCOVERY RESULTS\n{rag_output}")

    parts.append("### END OF CONTEXT ###")
    return "\n\n".join(parts)


def AI_pipeline(user_prompt: str, cve_data: dict, model: str) -> tuple[str, dict]:
    """
    Two-model pipeline:
      1. qdrant_RAG() acts as the RAG model — it determines if a vector DB tool
         call is needed, performs it, and returns the raw result.
      2. The LoRA model (DEFAULT_AI_MODEL) receives the user prompt plus any
         retrieved context and generates the final user-facing response.
    """

    rag_output = {"used_rag": False, "rag_results": None}
    rag_results = ""

    # regex check for version numbers in the prompt to decide if RAG is needed
    version_pattern = r"\b\d+\.\d+(\.\d+)?\b"
    if re.search(version_pattern, user_prompt) and re.search(r"\b\w+\b", user_prompt):
        # --- Stage 1: RAG model ---
        # rag_results: formatted string of retrieved CVE data (or "" if no tool call)
        # rag_output:  metadata dict for logging {"used_rag": bool, "rag_results": ...}
        rag_results, rag_output = qdrant_RAG(user_prompt)

    if rag_output["used_rag"] and rag_results.strip() == "":
        rag_results = "Tried finding relevant CVEs with RAG but got no results. This may indicate that the software has no known vulnerabilities"

    # --- Stage 2: Build context for the LoRA model ---
    context = build_context_block(cve_data, rag_results)

    if rag_output["used_rag"]:
        software_name = rag_output.get("software_name", "N/A")
        version = rag_output.get("version", "N/A")
        SYSTEM_PROMPT = VULNERABILITY_DISCOVERY_PROMPT.format(
            software=software_name,
            version=version
        )
    elif cve_data:
        SYSTEM_PROMPT = CVE_DESCRIPTOR_PROMPT.format(
            cve_id=list(cve_data.keys())[0]  # Just take the first CVE for the descriptor prompt
        )
    else:
        SYSTEM_PROMPT = GENERALL_INFORMATION_PROMPT
        # If no CVE data and no RAG results, we could choose to return early or proceed with a generic prompt.


    # DEBUG: print the context being passed to the LoRA model
    # print("\n--- Context passed to LoRA model ---")
    # print(context)
    # print("--- End of context ---\n")

    # Inject retrieved context as a second system message so the LoRA model
    # never mistakes it for user input.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if context:
        messages.append({
            "role": "system",
            "content": f"Retrieved context — use this to answer the user:\n\n{context}"
        })

    messages.append({"role": "user", "content": user_prompt})

    # --- Stage 3: LoRA model generates the final response ---
    lora_response = ollama.chat(model=model, messages=messages)
    final_answer = lora_response["message"]["content"]

    # rag_usage summarises what the RAG stage retrieved, for logging/debugging
    rag_usage = {
        "used_rag": rag_output["used_rag"],
        "rag_results": rag_results,
        "cve_ids_looked_up": list(cve_data.keys()),
        "context_injected": bool(context),
    }

    return final_answer, rag_usage


def handle_user_query(query: str, model_name: str = DEFAULT_AI_MODEL) -> tuple[str, dict, dict]:
    # Check for CVE pattern in the query
    cve_matches = re.findall(CVE_PATTERN, query)
    cve_data = {}
    rag_usage = {}
    if cve_matches:
        for cve_id in cve_matches:
            cve_info_dict = getCVEInfo(cve_id)
            cve_data[cve_id] = cve_info_dict

    if cve_data:
        cve_data = clean_cve_data(cve_data)

    response, rag_usage = AI_pipeline(query, cve_data, model_name)
    return response, rag_usage, cve_data

def main():
    print('Type "exit" to quit.')
    while True:
        user_input = input("prompt> ")
        if user_input.lower() == 'exit':
            break
        response, _, _ = handle_user_query(user_input)
        print("Response:", response)


if __name__ == "__main__":
    try:
        if not check_qdrant_available():
            start_qdrant()
    except Exception as e:
        print("Error starting or connecting to Qdrant:", e)
        exit(1)
    main()