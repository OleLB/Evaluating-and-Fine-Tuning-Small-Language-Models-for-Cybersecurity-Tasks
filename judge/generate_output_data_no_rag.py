"""
0. Open and read database db/scores.db
1. Select * FROM table "Scores" where column "model" is "mistral_nemo_cve"
2. For each row, extract the "input_id", "cve_data" and "rag_output" columns.
3. Using the "input_id", look up the corresponding "input" value from the "Inputs" table.
4. use a specified AI model (MODEL_TO_TEST) and pass the "question" value using ollama (first check if the model has already answered that question, if so, skip to the next row) and get the output.
5. Add a new row to the database with the following values:
    - "model": MODEL_TO_TEST
    - "question": the original "question" value from the row
    - "cve_data": the original "cve_data" value from the row
    - "rag_output": the original "rag_output" value from the row
    - "model_output": the output from the AI model when given the "question" value


from judge.score_db_utils import add_entry, get_connection

# def add_entry(input_text: str, output_text: str, model_name: str, rag_output: dict, cve_data: dict = None) -> int:              # Use this function to add a new entry to the database with the specified values.
"""


import json
import sqlite3
import ollama

from judge.score_db_utils import add_entry, get_connection


DEFAULT_MODEL_TO_TEST = "mistral-nemo:12b-instruct-2407-q8_0"
SOURCE_MODEL          = "mistral_nemo_cve"  # Only process rows where model = SOURCE_MODEL


def fetch_source_rows(conn: sqlite3.Connection) -> list[dict]:
    """Return all Scores rows where model = SOURCE_MODEL."""
    cursor = conn.execute(
        "SELECT input_id, cve_data, rag_output FROM Scores WHERE model_name = ?",
        (SOURCE_MODEL,),
    )
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def fetch_input_text(conn: sqlite3.Connection, input_id: int) -> str | None:
    """Return the 'input' value from the Inputs table for the given input_id."""
    cursor = conn.execute(
        "SELECT input FROM Inputs WHERE id = ?",
        (input_id,),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def already_answered(conn: sqlite3.Connection, model: str, input_id: int) -> bool:
    """Return True if the model has already produced an entry for this input_id."""
    cursor = conn.execute(
        "SELECT 1 FROM Scores WHERE model_name = ? AND input_id = ? LIMIT 1",
        (model, input_id),
    )
    return cursor.fetchone() is not None


def query_ollama(model: str, question: str) -> str:
    """Send the question to the specified Ollama model and return its response."""
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": question}],
    )
    return response["message"]["content"]


def parse_json_field(value):
    """Parse a field that may already be a dict or a JSON string."""
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value   # return raw string if it isn't valid JSON
    return value
 
 

def main(model_to_test: str) -> None:
    print(f"Source model : {SOURCE_MODEL}")
    print(f"Model to test: {model_to_test}")

    # Use get_connection from the judge utility so any setup logic is reused.
    conn = get_connection()

    source_rows = fetch_source_rows(conn)
    print(f"Found {len(source_rows)} row(s) from '{SOURCE_MODEL}' to process.\n")

    skipped = 0
    processed = 0
    errors = 0

    for row in source_rows:
        input_id  = row["input_id"]
        cve_data  = parse_json_field(row["cve_data"])
        rag_output = parse_json_field(row["rag_output"])
        rag_usage = {
            "used_rag": rag_output is not None,
            "rag_results": rag_output,
        }

        # ── 1. Skip if the model already has an answer for this input ──────
        if already_answered(conn, model_to_test, input_id):
            print(f"  [SKIP] input_id={input_id} — '{model_to_test}' already answered.")
            skipped += 1
            continue

        # ── 2. Retrieve the question text from the Inputs table ────────────
        question = fetch_input_text(conn, input_id)
        if question is None:
            print(f"  [WARN] input_id={input_id} — not found in Inputs table. Skipping.")
            skipped += 1
            continue

        # ── 3. Query the model via Ollama ──────────────────────────────────
        print(f"  [RUN ] input_id={input_id} — querying '{model_to_test}' …", end=" ", flush=True)
        try:
            model_output = query_ollama(model_to_test, question)
        except Exception as exc:
            print(f"ERROR\n         {exc}")
            errors += 1
            continue
        print("done.")

        # ── 4. Store  ──────────────────────────────────────────
        try:
            add_entry(
                question,
                model_output,
                model_to_test,
                rag_usage,
                cve_data,
            )
            processed += 1
        except Exception as exc:
            print(f"  [ERROR] Failed to add entry for input_id={input_id}: {exc}")
            errors += 1

    conn.close()

    print(f"\nDone. processed={processed}, skipped={skipped}, errors={errors}")




if __name__ == "__main__":
    main(model_to_test=DEFAULT_MODEL_TO_TEST)
