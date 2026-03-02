"""
This script is the entry point for interacting with the AI RAG system.

It will:
- Take a user query as input.
- If the prompt contains a CVE on the form CVE-YYYY-NNNN, it will search the database for relevant information.
- If the AI determines that the user is searching for a CVE, it will return the relevant information from the vector database.
- Finally, it will return the answer to the user.
"""
# Future upgrade: remember conversations

import json
import sqlite3
from types import SimpleNamespace
import ollama
import re
from qdrant.queryQdrant import query_qdrant
from qdrant.startQdrant import start_qdrant, check_qdrant_available

DATABASE = "../db/cve_database.db"
CVE_PATTERN = r"CVE-\d{4}-\d{4,7}"
DEFAULT_AI_MODEL = "mistral-nemo-cve2:latest"

SYSTEM_PROMPT = """
You are a cybersecurity assistant specialized in CVE and vulnerability lookup.
Your purpose is to help users find and understand software vulnerabilities.

You have access to a vector database tool that contains software vulnerability information.
Use the vector database tool ONLY when the user asks about vulnerabilities affecting a specific software name and version (e.g. "Are there any vulnerabilities in nginx 1.18.0?").
Do NOT use the vector database tool to look up details about a specific CVE ID.
If you use the tool, and dont recive results that are relevant for the requested software, they should not be presented to the user.

When a user asks about a specific CVE ID:
- You will be provided with the CVE description, affected software, and other details from the database. Use this information to answer the user's question.
- Report only what is available in the database record
- If specific details (such as authentication requirements, affected components, or exploit conditions) are not present in the data, say so explicitly
- Do NOT infer or fabricate details that are not in the data — an honest "the NVD record does not specify this" is always preferable to a plausible-sounding guess

For general cybersecurity questions (not about a specific CVE or software version):
- Answer from your training knowledge without using any tools
- Make clear you are speaking from general knowledge, not from the CVE database

NEVER provide URLs.


"""

'''
To call a tool, use the following format in your response:
{
    "function": {
        "name": "vector_database_retrieval",
        "arguments": "{\"query\": \"vulnerability for <software_name> version x.y.z\"}"
      }
    }
}
'''

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
    

def qdrantRAG(query, top_x=5) -> str:
    try:
        # 1. Access 'function' as an attribute
        # 2. Access 'arguments' as an attribute
        # 3. 'arguments' is a dictionary, so use ['query']

        
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


def dynamic_rag(user_prompt: str, cve_data: dict, model: str):
    # 2. Define the 'tool' schema so the model knows what the function does
    tools = [{
        'type': 'function',
        'function': {
            'name': 'vector_database_retrieval',
            'description': 'Attempt to discover weaknesses for a software version using a vector database. Use the tool if the user is asking you to find a vulnerability for a piece of software. Do NOT use this tool to learn more about a CVE. The results may or may not be relevant. The most promising results should be presented to the user in a structured format, show a maximum of 5 results. Prioritize vulnerabilities that are rated as CRITICAL or HIGH severity. Do not present irrelevant results to the user.',
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
        system_prompt += "\n\nThe user mentioned a CVE, the following CVE information is available:\n"
        for cve_id, cve_info in cve_data.items():
            system_prompt += f"\n{cve_info}\n"
            
    messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': user_prompt}]
    response = ollama.chat(model=model, messages=messages, tools=tools)

    rag_output = {"used_rag": False, "rag_results": None}
    
    # 4a. Check for standard tool calls (normal models)
    if response.get('message', {}).get('tool_calls'):
        for tool in response['message']['tool_calls']:
            if tool['function']['name'] == 'vector_database_retrieval':
                # Execute the function
                query = tool.function.arguments['query']
                context = qdrantRAG(query)
                rag_output["used_rag"] = True
                rag_output["rag_results"] = context
                # Add the tool's result to the conversation
                messages.append(response['message'])
                messages.append({
                    'role': 'tool',
                    'content': context,
                })
        
        # Final call to generate answer based on the retrieved context
        final_response = ollama.chat(model=model, messages=messages)
        return final_response['message']['content'], rag_output
    
    # 4b. Check for custom text-based tool calls (fine-tuned model)
    response_text = response.get('message', {}).get('content', '')
    if "vector_database_retrieval" in response_text:
        try:
            tool_call = json.loads(response_text)
            if tool_call['function']['name'] == 'vector_database_retrieval':
                query = tool_call['function']['arguments']['query']
                context = qdrantRAG(query)
                rag_output["used_rag"] = True
                rag_output["rag_results"] = context
                # Add the tool's result to the conversation
                messages.append({'role': 'assistant', 'content': response_text})
                messages.append({'role': 'tool', 'content': context})
                
                # Final call to generate answer based on the retrieved context
                final_response = ollama.chat(model=model, messages=messages)
                return final_response['message']['content'], rag_output
        except json.JSONDecodeError:
            print("Failed to parse tool call as JSON. Response was:", response_text)
        except Exception as e:
            print("Error processing tool call:", e)
            print("Response text was:", response_text)

    # If no tool calls, return the original response
    return response_text, rag_output


def handle_user_query(query: str, model_name: str = DEFAULT_AI_MODEL) -> str:
    # Check for CVE pattern in the query
    cve_matches = re.findall(CVE_PATTERN, query)
    cve_data = {}
    rag_usage = {}
    if cve_matches:
        for cve_id in cve_matches:
            cve_info_dict = getCVEInfo(cve_id)
            cve_data[cve_id] = cve_info_dict

    response, rag_usage = dynamic_rag(query, cve_data, model_name)
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