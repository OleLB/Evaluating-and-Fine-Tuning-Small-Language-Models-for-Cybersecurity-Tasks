"""This script is the core of the RAG pipeline. This is where the code interacts with the databases."""

import sqlite3
import ollama
import ast
from rag_and_lora.qdrant.queryQdrant import query_qdrant
from packaging import version

DATABASE = "db/cve_database.db"
DEFAULT_AI_MODEL = "mistral-nemo"

SYSTEM_PROMPT = """
### ROLE
You are a precision router for a Cybersecurity Vulnerability System. 

### CRITERIA FOR TOOL CALL
A tool call is ONLY permitted if BOTH conditions are met:
1. A specific Software Name is mentioned (e.g., "Apache", "Windows").
2. A specific Version Number is mentioned (e.g., "2.4.50", "11").

### RESTRICTIONS
- If a CVE ID (e.g., CVE-2021-44228) is provided: DO NOT call the tool.
- If the question is general (e.g., "What is a buffer overflow?"): DO NOT call the tool.
- If no version is provided: DO NOT call the tool.

### TASK
Analyze the user input. If it meets the criteria, output the tool call. If it fails ANY criteria, output "NO_TOOL_REQUIRED".

### EXAMPLES
User: "What vulnerabilities affect Nginx 1.18?"
Analysis: Software (Nginx) and Version (1.18) present.
Output: [Tool Call]

User: "Tell me about CVE-2024-1234."
Analysis: Contains CVE ID, no specific software/version lookup requested.
Output: NO_TOOL_REQUIRED

User: "Is Chrome vulnerable to hacks?"
Analysis: Software present, but no version specified.
Output: NO_TOOL_REQUIRED
"""

def clean_cpe_data(raw_software_str, target_version):
    """
    Logically compares target_version against CPE ranges to 
    provide a definitive 'MATCH' or 'NO MATCH' label for the LLM.
    """
    try:
        user_v = version.parse(target_version)
        software_list = ast.literal_eval(raw_software_str)
        
        matches = []
        for entry in software_list:
            for match in entry.get('cpeMatch', []):
                v_start = match.get('versionStartIncluding') or match.get('versionStartExcluding')
                v_end = match.get('versionEndIncluding') or match.get('versionEndExcluding')
                
                is_affected = True
                
                # Logic: If current version is BELOW the start or ABOVE the end, it's not a match
                if v_start and user_v < version.parse(v_start):
                    is_affected = False
                if v_end and user_v > version.parse(v_end):
                    is_affected = False
                
                if is_affected:
                    # Capture the range string for the LLM's context
                    range_info = f"{v_start if v_start else '0'} to {v_end if v_end else 'latest'}"
                    matches.append(range_info)

        if matches:
            return f"VULNERABILITY_CHECK: MATCH CONFIRMED. User version {target_version} falls within affected range ({matches[0]})."
        
        return "VULNERABILITY_CHECK: NO MATCH. User version appears outside of known affected ranges."

    except Exception:
        # Fallback: simple text check if version parsing fails
        if target_version in raw_software_str:
            return f"VULNERABILITY_CHECK: POTENTIAL MATCH. Version {target_version} found in raw data."
        return "VULNERABILITY_CHECK: UNKNOWN. Metadata format incompatible."
    

def getCVEInfo(cve_id: str) -> str:
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        # Fix: Select all needed columns, not just description
        cursor.execute("""
            SELECT cve_id, description, cwe_name, cwe_description, 
                label_attack, severity, known_vulnerable_software, 
                cpe_list, mitigation, attack_technique_name
            FROM cves 
            WHERE cve_id = ?
        """, (cve_id,))
        row = cursor.fetchone()
        conn.close()
    except Exception as e:
        return f"Error connecting to CVE database: {e}"
    
    if row:
        formatted_info = {
            "CVE_ID": row[0],
            "Description": row[1],
            "CWE Name": row[2],
            "CWE Description": row[3],
            "Associated MITRE ATT&CK Technique": row[4],
            "MITRE ATT&CK Technique Name": row[9],
            "Severity": row[5],
            "Known Vulnerable Software": row[6],
            "Mitigation Techniques": row[8]
        }
        return formatted_info
    else:
        return "CVE not found in the database."


def qdrantRAG(software_name: str, target_version:str, top_x: int = 5) -> str:
    """
    Queries the vector database, enriches each unique result with full CVE
    details from the SQLite database, and returns a formatted string ready
    for injection into the LoRA model's context.
    """

    query = f"vulnerabilities for {software_name} version {target_version}"
    try:
        print("Executing RAG with Qdrant with query:", query)
        points = query_qdrant(query)
    except Exception as e:
        print("Error during RAG with Qdrant:", e)
        return f"Error during RAG with Qdrant: {e}"

    if not points:
        return "No relevant vulnerabilities found."

    seen_cve_ids = set()
    results = []

    for item in points:
        cve_id = item.payload.get("CVE_ID")
        if not cve_id or cve_id in seen_cve_ids:
            continue

        # check if the software_name is in the cve description
        if not any(software_name.lower() in str(value).lower() for value in item.payload.values()):
            continue

        seen_cve_ids.add(cve_id)

        cve_info = getCVEInfo(cve_id)

        # getCVEInfo returns a string on failure — skip gracefully
        if isinstance(cve_info, str):
            results.append(f"[{cve_id}] (Score: {item.score:.4f})\n  Note: {cve_info}")
            continue

        # --- CLEANING STEP ---
        # Replace the massive JSON dump with a human-readable summary
        vuln_summary = clean_cpe_data(cve_info.get('Known Vulnerable Software', ''), target_version)

        # Build a "Low-Noise" entry
        entry = (
            f"ENTRY: {cve_id}\n"
            f"SEVERITY: {cve_info.get('Severity', 'N/A')}\n"
            f"SUMMARY: {cve_info.get('Description', 'N/A')}\n"
            f"VULNERABILITY_CHECK: {vuln_summary}\n"
            f"TECHNIQUE: {cve_info.get('Associated MITRE ATT&CK Technique', 'N/A')}\n"
            f"FIX: {cve_info.get('Mitigation Techniques', 'N/A')}"
        )
        results.append(entry)
        
        if len(results) >= top_x: break

    return "\n\n---\n\n".join(results)

def qdrant_RAG(user_prompt: str, model: str = DEFAULT_AI_MODEL) -> tuple[str, dict]:
    """
    RAG model stage of the pipeline.
    Decides whether a vector DB lookup is needed, performs it if so,
    and returns the enriched results as a string for the LoRA model.
    Never makes a second LLM call — result assembly is the LoRA model's job.

    Returns:
        rag_results (str): Formatted context string, or "" if no lookup was needed.
        rag_output  (dict): Metadata for logging/debugging.
    """
    tools = [{
        'type': 'function',
        'function': {
            'name': 'vector_database_retrieval',
            'description': (
                'Search a vector database for known vulnerabilities affecting a '
                'specific software name and version. Use ONLY when the user asks '
                'about vulnerabilities for a specific piece of software such as "is there a problem with apache 2.4.49?" '
                'Do NOT use to look up a CVE ID directly.'
            ),
            'parameters': {
                'type': 'object',
                'properties': {
                    'software name': {
                        'type': 'string',
                        'description': 'Name of the software to search for (e.g., "Apache", "Windows")',
                    },
                    'version': {
                        'type': 'string',
                        'description': 'Version of the software to search for (e.g., "2.4.49", "11")',
                    }
                },
                'required': ['software name', 'version'],
            },
        },
    }]

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': user_prompt}
    ]
    response = ollama.chat(model=model, messages=messages, tools=tools)

    rag_output = {"used_rag": False, "rag_results": None}

    # structured tool call
    if response.get('message', {}).get('tool_calls'):
        for tool in response['message']['tool_calls']:
            if tool['function']['name'] == 'vector_database_retrieval':
                software_name = tool.function.arguments['software name']
                target_version = tool.function.arguments['version']
                context = qdrantRAG(software_name, target_version)
                rag_output["used_rag"] = True
                rag_output["rag_results"] = context
                rag_output["software_name"] = software_name
                rag_output["version"] = target_version
                return context, rag_output

    # --- No tool call needed ---
    return "", rag_output