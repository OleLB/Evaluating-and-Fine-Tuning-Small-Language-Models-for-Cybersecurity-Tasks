"""
This script generates answers for every entry in the Inputs table
and stores the results as new rows in the Scores table.
"""
import sqlite3
from judge.score_db_utils import add_entry, DB_PATH
from main import handle_user_query

MAX_COUNT = None  # Set to an int to cap how many inputs are processed


def get_all_inputs() -> list[dict]:
    """Fetch all entries from the Inputs table."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT id, input FROM Inputs")
        return [dict(row) for row in cursor.fetchall()]


def input_already_scored(input_id: int, model_name: str) -> bool:
    """Return True if a non-placeholder Scores row already exists for this input + model."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*) FROM Scores
            WHERE input_id = ? AND model_name = ? AND output != 'tmp'
            """,
            (input_id, model_name),
        )
        return cursor.fetchone()[0] > 0


def generate_all_outputs(model_name: str):
    all_inputs = get_all_inputs()
    print(f"Found {len(all_inputs)} entries in the Inputs table.")

    # Filter out inputs that already have a scored row for this model
    pending = [e for e in all_inputs if not input_already_scored(e["id"], model_name)]
    print(f"{len(pending)} inputs have no existing output for model '{model_name}'.")

    if MAX_COUNT is not None:
        pending = pending[:MAX_COUNT]
        print(f"Processing only the first {MAX_COUNT} entries.")

    for entry in pending:
        input_query = entry["input"]
        print(f"\nProcessing input ID {entry['id']}: {input_query}")

        response, rag_usage, cve_data = handle_user_query(input_query, model_name)

        row_id = add_entry(input_query, response, model_name, rag_usage, cve_data)
        print(f"Stored result as Scores row ID {row_id}.")

    print(f"\nDone. Processed {len(pending)} inputs.")


if __name__ == "__main__":
    MODEL_NAME = "mistral-nemo-cve2"
    generate_all_outputs(MODEL_NAME)