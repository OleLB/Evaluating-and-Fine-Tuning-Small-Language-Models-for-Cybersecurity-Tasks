"""This script generates answers to questions that already exist in the database, but are missing outputs"""

from judge.score_db_utils import get_missing_output_entries, insert_output_by_id
from main import handle_user_query

max_count = 250

def generate_output_data(model_name: str):
    missing_entries = get_missing_output_entries(model_name)
    print(f"Found {len(missing_entries)} entries with missing output for model '{model_name}'.")

    if max_count is not None:
        missing_entries = missing_entries[:max_count]
        print(f"Processing only the first {max_count} entries.")

    for entry in missing_entries:
        input_query = entry['input']
        print(f"\nProcessing entry ID {entry['id']} with input: {input_query}")
        rag_result = None
        cve_data = None
        response, rag_usage, cve_data = handle_user_query(input_query, model_name)
        rag_result = rag_usage["rag_results"]
        # print(f"Generated output for entry ID {entry['id']}:\n{response}\n")
        insert_output_by_id(entry['id'], response, rag_result, cve_data)


if __name__ == "__main__":
    model_name = "mistral_nemo_cve"
    generate_output_data(model_name)