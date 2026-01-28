"""
This script is the entry point for interacting with the AI RAG system.

It will:
- Take a user query as input.
- If the prompt contains a CVE on the form CVE-YYYY-NNNN, it will search the database for relevant information.
- If the AI determines that the user is searching for a CVE, it will return the relevant information from the vector database.
- Finally, it will return the answer to the user.
"""
# Future upgrade: remember conversations

import sqlite3
import ollama
import re
from qdrant.queryQdrant import query_qdrant
from qdrant.startQdrant import start_qdrant, check_qdrant_available

DATABASE = "../db/cve_database.db"
CVE_PATTERN = r"CVE-\d{4}-\d{4,7}"
DEFAULT_AI_MODEL = "mistral-nemo:latest"

SYSTEM_PROMPT = """
You are a cybersecurity expert assistant.
Your task is to help users find information about software vulnerabilities and explain CVEs.
If a user wants to know about a specific CVE you will be provided with the relevant information.
You have access to a vector database containing vulnerability data.
Use the vector database tool if the user is asking for vulnerabilities for a piece of software.

DO NOT:
Provide URLs to the user.

Examples:
If a user enters only a CVE ID (e.g., CVE-2023-12345), provide information about it. Do not use the vector database tool.
If a user enters a software name and version, use the vector database tool to find relevant vulnerabilities.
"""


def getCVEInfo(cve_id: str) -> str:
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        # Fix: Select all needed columns, not just description
        cursor.execute("""
            SELECT cve_id, description, cwe_name, cwe_description, 
                label_attack, severity, known_vulnerable_software, 
                cpe_list, mitigation 
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
            "Severity": row[5],
            "Known Vulnerable Software": row[6],
            "Mitigation Techniques": row[8]
        }
        return formatted_info
    else:
        return "CVE not found in the database."
    

# def query_ai(prompt: str, model: str = DEFAULT_AI_MODEL) -> str:
#     response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
#     """
#     Example response:
#     model='mistral-nemo:latest' created_at='2026-01-26T15:39:56.693192Z' done=True done_reason='stop' total_duration=889303500 load_duration=68903200 prompt_eval_count=9 prompt_eval_duration=16153600 eval_count=51 eval_duration=774916100 message=Message(role='assistant', content='You are talking to an assistant powered by artificial intelligence, designed to understand and generate human-like text based on the input it receives. You can ask me questions, discuss topics, or just chat casually. How can I assist you today? 😊', thinking=None, images=None, tool_name=None, tool_calls=None) logprobs=None
#     """
#     # parse response
#     content = response.message.content
#     return content


def qdrantRAG(tool_call, top_x=10) -> str:
    try:
        # 1. Access 'function' as an attribute
        # 2. Access 'arguments' as an attribute
        # 3. 'arguments' is a dictionary, so use ['query']
        query = tool_call.function.arguments['query']
        
        print("Executing RAG with Qdrant with query:", query)
        points = query_qdrant(query)
    except Exception as e:
        print("Error during RAG with Qdrant:", e)
        return f"Error during RAG with Qdrant: {e}"

    if not points:
        return "No relevant vulnerabilities found."

    filtered_results = []
    unique_cve_ids = set()

    for item in points:
        # Match the key case used in your payload ('CVE_ID')
        cve_id = item.payload.get('CVE_ID')
        cpe = item.payload.get('CPE')
        
        if cve_id and cve_id not in unique_cve_ids:
            unique_cve_ids.add(cve_id)
            # Create the string for the AI here
            text = item.payload.get("text", "")
            # use "getCVEInfo" to get the CVE description
            cve_info = getCVEInfo(cve_id)
            description = cve_info["Description"]
            filtered_results.append(f"[{cve_id}],\n{text},\nDescription: {description},\nKnown Vulnerable Software: {cpe},\n(Score: {item.score:.4f})")

    top_x_results = filtered_results[:top_x]

    # Show the top x results to the user
    # for res in top_x_results:
    #     print(res)
    #     print("\n---\n")

    # Join the top x unique results into the final context string
    return "\n\n---\n\n".join(top_x_results)

def dynamic_rag(user_prompt: str, cve_data: dict, model: str = DEFAULT_AI_MODEL):
    # 2. Define the 'tool' schema so the model knows what the function does
    tools = [{
        'type': 'function',
        'function': {
            'name': 'vector_database_retrieval',
            'description': 'Attempt to discover relevant CVEs using a vector database. Use if the user is asking you to find a vulnerability for a piece of software. Do NOT use this tool to learn more about a specific CVE. The results may or may not be relevant. The most promising results should be presented to the user in a structured format, show a maximum of 5 results. Prioritize vulnerabilities that are rated as CRITICAL or HIGH severity.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'query': {'type': 'string', 'description': 'A search query on the format: "vulnerability for <software name> version <version number>"'},
                },
                'required': ['query'],
            },
        },
    }]

    # 3. Initial Chat Call
    system_prompt = SYSTEM_PROMPT
    if cve_data:
        system_prompt += "\n\nThe user has mentioned a CVE, the following CVE information is available:\n"
        for cve_id, cve_info in cve_data.items():
            system_prompt += f"\n{cve_info}\n"
            
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
    response = ollama.chat(model=model, messages=messages, tools=tools)

    # 4. Check if the model wants to use the tool
    if response.get('message', {}).get('tool_calls'):
        for tool in response['message']['tool_calls']:
            if tool['function']['name'] == 'vector_database_retrieval':
                # Execute the function
                context = qdrantRAG(tool)
                
                # Add the tool's result to the conversation
                messages.append(response['message'])
                messages.append({
                    'role': 'tool',
                    'content': context,
                })
        
        # Final call to generate answer based on the retrieved context
        final_response = ollama.chat(model=model, messages=messages)
        return final_response['message']['content']
    
    # If no tool was needed, just return the direct response
    return response['message']['content']



def handle_user_query(query: str) -> str:
    # Check for CVE pattern in the query
    cve_matches = re.findall(CVE_PATTERN, query)
    cve_data = {}
    if cve_matches:
        for cve_id in cve_matches:
            cve_info_dict = getCVEInfo(cve_id)
            cve_data[cve_id] = cve_info_dict

    response = dynamic_rag(query, cve_data)
    return response

def main():
    print('Type "exit" to quit.')
    while True:
        user_input = input("prompt> ")
        if user_input.lower() == 'exit':
            break
        response = handle_user_query(user_input)
        print("Response:", response)


if __name__ == "__main__":
    try:
        if not check_qdrant_available():
            start_qdrant()
    except Exception as e:
        print("Error starting or connecting to Qdrant:", e)
        exit(1)
    main()