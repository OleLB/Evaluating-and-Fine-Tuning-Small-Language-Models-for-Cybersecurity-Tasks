#!/usr/bin/env python3
"""
Script to process training data for tool call examples.
Takes incomplete tool call examples, executes them, and uses an LLM to generate
complete conversational responses.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any

# Import your existing utilities
from utils.queryGPT import queryGPT
from qdrant.queryQdrant import query_qdrant

# Configuration
INPUT_DIR = "LoRA/training_data_tool_calls"
OUTPUT_DIR = "LoRA/training_data_tool_calls_complete"
DATABASE = "db/cve_database.db"  # Update this path
PROMPT_PATH = "prompts/filter_and_explain.txt"  # We'll create this


def readFile(filepath: str) -> str:
    """Read a file and return its contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def getCVEInfo(cve_id: str) -> Dict[str, Any]:
    """
    Retrieve CVE information from SQLite database.
    
    Args:
        cve_id: The CVE identifier (e.g., 'CVE-2024-1234')
    
    Returns:
        Dictionary containing CVE details or error message
    """
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
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
        return {"error": f"Error connecting to CVE database: {e}"}
    
    if row:
        formatted_info = {
            "CVE_ID": row[0],
            "Description": row[1],
            "CWE Name": row[2],
            "CWE Description": row[3],
            "Associated MITRE ATT&CK Technique": row[4],
            "Severity": row[5],
            "Known Vulnerable Software": row[6],
            "CPE List": row[7],
            "Mitigation Techniques": row[8]
        }
        return formatted_info
    else:
        return {"error": "CVE not found in the database."}


def executeToolCall(tool_call_data: Dict[str, Any], top_x: int = 5) -> str:
    """
    Execute the vector database tool call and retrieve CVE details.
    
    Args:
        tool_call_data: The tool call data from the training example
        top_x: Number of top results to return
    
    Returns:
        Formatted string with CVE information
    """
    try:
        # Extract the query from the tool call arguments
        arguments = tool_call_data['function']['arguments']
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        
        query = arguments['query']
        print(f"Executing RAG with query: {query}")
        
        # Query Qdrant vector database
        points = query_qdrant(query)
        
    except Exception as e:
        print(f"Error during RAG with Qdrant: {e}")
        return f"Error during RAG with Qdrant: {e}"
    
    if not points:
        return "No relevant vulnerabilities found."
    
    # Process results and get CVE details
    filtered_results = []
    unique_cve_ids = set()
    
    for item in points:
        cve_id = item.payload.get('CVE_ID')
        cpe = item.payload.get('CPE')
        
        if cve_id and cve_id not in unique_cve_ids:
            unique_cve_ids.add(cve_id)
            
            # Get full CVE details from SQLite
            cve_info = getCVEInfo(cve_id)
            
            if "error" in cve_info:
                continue
            
            text = item.payload.get("text", "")
            description = cve_info.get("Description", "N/A")
            severity = cve_info.get("Severity", "N/A")
            cwe_name = cve_info.get("CWE Name", "N/A")
            mitigation = cve_info.get("Mitigation Techniques", "N/A")
            
            result_entry = (
                f"[{cve_id}]\n"
                f"Severity: {severity}\n"
                f"CWE: {cwe_name}\n"
                f"Description: {description}\n"
                f"Known Vulnerable Software: {cpe}\n"
                f"Mitigation: {mitigation}\n"
                f"Vector DB Text: {text}\n"
                f"(Relevance Score: {item.score:.4f})"
            )
            filtered_results.append(result_entry)
    
    # Return top X results
    top_x_results = filtered_results[:top_x]
    return "\n\n---\n\n".join(top_x_results)


def prepareFilteringPrompt(user_query: str, tool_results: str) -> str:
    """
    Prepare the prompt for the LLM to filter and explain vulnerabilities.
    
    Args:
        user_query: The original user question
        tool_results: The raw results from the vector database + SQLite
    
    Returns:
        Formatted prompt for the LLM
    """
    prompt = f"""You are a cybersecurity expert analyzing vulnerability scan results. Your task is to:

1. Review the vulnerability results returned from a vector database search
2. Filter out any irrelevant or low-quality matches
3. Identify the most promising/relevant vulnerabilities based on the user's query
4. Provide a clear, professional explanation of the relevant findings

USER QUERY:
{user_query}

TOOL RESULTS (Vector Database + CVE Details):
{tool_results}

INSTRUCTIONS:
- Analyze each CVE in the context of the user's specific query
- Filter out CVEs that are not relevant to what the user asked about
- For relevant CVEs, explain WHY they matter for this specific query
- Focus on severity, exploitability, and relevance to the queried software/version
- If no results are truly relevant, state that clearly
- Use a professional, security-focused tone
- Be concise but thorough

Provide your analysis as a natural conversational response that directly addresses the user's query."""

    return prompt


def processTrainingExample(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single training example through the complete pipeline.
    
    Args:
        example: A single training example with user message and tool call
    
    Returns:
        Complete training example with tool results and LLM response
    """
    messages = example['messages']
    user_message = messages[0]['content']
    tool_call = messages[1]['tool_calls'][0]
    
    # print(f"\nProcessing: {user_message}")
    
    # Step 1: Execute the tool call
    tool_results = executeToolCall(tool_call)
    
    if not tool_results:
        print(f"  Warning: Tool call returned no useful results")
    
    # Step 2: Get LLM to filter and explain results
    prompt = prepareFilteringPrompt(user_message, tool_results)
    
    try:
        llm_response, usage_info = queryGPT(prompt, "gpt-4o")
        # print(f"  LLM response generated (tokens: {usage_info.get('total_tokens', 'N/A')})")
    except Exception as e:
        print(f"  Error querying LLM: {e}")
        llm_response = "Error generating response. Please try again."
    
    # Step 3: Format as complete training example
    complete_example = {
        "messages": [
            {
                "role": "user",
                "content": user_message
            },
            {
                "role": "assistant",
                "tool_calls": [tool_call]
            },
            {
                "role": "tool",
                "tool_call_id": tool_call['id'],
                "name": tool_call['function']['name'],
                "content": tool_results
            },
            {
                "role": "assistant",
                "content": llm_response
            }
        ]
    }
    
    return complete_example


def processJsonlFile(input_path: Path, output_path: Path) -> None:
    """
    Process a single JSONL file and save the complete training examples.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
    """
    print(f"\n{'='*60}")
    print(f"Processing file: {input_path.name}")
    print(f"{'='*60}")
    
    complete_examples = []
    
    # Read and process each line
    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                example = json.loads(line)
                complete_example = processTrainingExample(example)
                complete_examples.append(complete_example)
                
            except json.JSONDecodeError as e:
                print(f"  Error decoding JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"  Error processing line {line_num}: {e}")
    
    # Write complete examples to output file
    # with open(output_path, 'w', encoding='utf-8') as f:
    #     for example in complete_examples:
    #         f.write(json.dumps(example, ensure_ascii=False) + '\n')
    base_output_path = output_path.parent / output_path.name
    for line_num, example in  enumerate(complete_examples, 1):
        # remove .jsonl from filename
        output_path = output_path.parent / f"{output_path.name}{line_num}.json"
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
        output_path = base_output_path  # reset to base for next iteration
    
    # print(f"\nCompleted: {len(complete_examples)} examples written to {output_path.name}")


def main():
    """Main function to process all training data files."""
    # Create output directory if it doesn't exist
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # Get all JSONL files from input directory
    input_dir = Path(INPUT_DIR)
    jsonl_files = list(input_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files to process")
    
    # Process each file
    for input_path in jsonl_files:
        output_path = (output_dir / input_path.name).with_suffix('')
        
        try:
            processJsonlFile(input_path, output_path)
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"All files processed! Output saved to: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()