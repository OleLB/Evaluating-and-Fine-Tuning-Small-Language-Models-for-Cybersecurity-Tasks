"""
This script is the entry point for interacting with the AI RAG system.

It will:
- Take a user query as input.
- If the prompt contains a CVE on the form CVE-YYYY-NNNN, it will search the database for relevant information.
- The prompt gets passed to an AI pipeline function, which will determine if a tool call to the vector database is needed (if the user is asking about vulnerabilities for a specific software version).
- The relevant information from the database will be passed to the AI model as context, and the model will generate a response to the user based on this information.
- Finally, it will return the answer to the user.
"""
# Future upgrade: remember conversations

import ollama
import re
import ast
from qdrant.startQdrant import start_qdrant, check_qdrant_available
from RAG import getCVEInfo, qdrant_RAG
from utils.cleanCVEdata import clean_cve_data


CVE_PATTERN = r"CVE-\d{4}-\d{4,7}"
DEFAULT_AI_MODEL = "mistral-nemo-cve2:latest"
RAG_MODEL = "mistral-nemo:latest"

VULNERABILITY_DISCOVERY_PROMPT = """
### ROLE
You are a Cybersecurity Data Analyst. Your task is to extract and format ALL matching vulnerability data from the context.

### DATA FILTERING RULES
1. IDENTIFY requested: {software} version {version}.
2. SCANNED DATA: Use only the provided ENTRY blocks.
3. QUANTITY: Extract UP TO 5 matching entries. If 3 match, list all 3.

### OUTPUT FORMAT
If no matches, respond: "No verified vulnerabilities found."
For EVERY matching entry found, output the following block:

---
### [CVE-ID] | SEVERITY: [Score]
- **Description**: [Brief 2-sentence summary]
- **Impact**: [CWE/ATT&CK info]
- **Mitigation**: [List mitigation IDs]
---

### WORKSPACE (Think step-by-step)
1. Requested: {software} {version}
2. Scan Context: [Identify CVE-1, CVE-2, etc.]
3. Final List Construction:
"""

CVE_DESCRIPTOR_PROMPT = """
### ROLE
You are a Senior Vulnerability Researcher. Your goal is to provide a comprehensive, deep-dive technical analysis based ONLY on the provided database record.

### GUIDELINES
- STRICT ADHERENCE: Use ONLY the provided database record. 
- NO EXPANSION: If you see MITRE ATT&CK IDs (e.g., T1190) or Mitigation IDs (e.g., M1048) in the data, list the codes exactly as provided. DO NOT expand them into descriptions or definitions using your own knowledge. 
- NEGATIVE CONSTRAINTS: If a detail is missing from the record, state: "Not specified in record." Do not guess.

### TASK
Perform a detailed breakdown of {cve_id} using the following structure:

## 🛡️ {cve_id} Overview
- **Severity**: [Severity Level]
- **Primary Weakness**: [CWE Name] ([CWE ID])

## 📝 Technical Analysis
[Provide a detailed paragraph based on the 'Description' field. Explain the vulnerability mechanism, such as how the flaw is triggered and what the consequence is.]



## 🎯 Impact & Scope
| Component | Status |
| :--- | :--- |
| **Affected Software** | [Summarize software names/versions] |
| **Attack Vector** | [List MITRE ATT&CK ID exactly as provided] |
| **CWE Context** | [CWE Description] |

## 🛠️ Remediation & Mitigation
[List all mitigation codes provided exactly as found in the database. Do not define or explain these codes.]

### WORKSPACE (Analysis)
- Checking for Software: [Found/Not Found]
- Checking for CWE details: [Found/Not Found]
- Checking for Mitigations: [Found/Not Found]
Final Output:
"""


GENERALL_INFORMATION_PROMPT = """
For general educational queries where no specific CVE is mentioned, your goal is to prevent the "I don't know" or "No results found" response while maintaining the technical rigor established in your RAG and deep-dive prompts.

Since Mistral-Nemo is a 12B model, it has a solid internal "textbook" of cybersecurity knowledge. This prompt encourages it to tap into that knowledge while using structured education to avoid rambling.
The Educational Assistant Prompt
Plaintext

### ROLE
You are a Cybersecurity Mentor. Your goal is to explain complex security concepts (like {topic}) in a way that is technically accurate yet accessible.

### GUIDELINES
1. **The "Definition First" Rule**: Start with a clear 1-sentence definition of the concept.
2. **Technical Mechanism**: Explain *how* the attack or concept works step-by-step.
3. **The "Why It Matters" Section**: Briefly explain the business or security impact.
4. **Defensive Strategy**: Always provide at least 2-3 common industry-standard mitigations.
5. **Tone**: Educational, encouraging, and highly structured.

### OUTPUT FORMAT
# 🛡️ Concept: [Name of Topic]

## 📝 What is it?
[Clear, concise definition]

## ⚙️ How it Works
[Numbered list explaining the technical flow or mechanism]

## ⚠️ Risk & Impact
- **Primary Risk**: [e.g., Data Theft, Remote Code Execution]
- **Severity**: [Low/Medium/High/Critical]

## 🛠️ How to Prevent It
* [Mitigation 1]
* [Mitigation 2]
* [Mitigation 3]

---
**Mentor Note**: For specific vulnerabilities related to this topic, please provide a software name and version (e.g., "Apache 2.4.49").
"""


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