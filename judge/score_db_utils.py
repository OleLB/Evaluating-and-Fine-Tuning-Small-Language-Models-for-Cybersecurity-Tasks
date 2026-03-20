import sqlite3
from typing import List, Optional, Tuple, Dict, Any

DB_PATH = "db/scores_new.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def _get_or_create_input_id(cursor, input_text: str) -> int:
    """Insert input_text into Inputs if not present, return its id."""
    cursor.execute("INSERT OR IGNORE INTO Inputs (input) VALUES (?)", (input_text,))
    cursor.execute("SELECT id FROM Inputs WHERE input = ?", (input_text,))
    return cursor.fetchone()[0]


# 1. Add input/output pair, return created row id
def add_entry(input_text: str, output_text: str, model_name: str, rag_usage: dict, cve_data: dict = None) -> int:
    rag_output = rag_usage["rag_results"] if rag_usage["used_rag"] else None
    cve_data_str = str(cve_data) if cve_data else None

    with get_connection() as conn:
        cursor = conn.cursor()
        input_id = _get_or_create_input_id(cursor, input_text)
        cursor.execute(
            """
            INSERT INTO Scores (input_id, output, model_name, rag_output, cve_data)
            VALUES (?, ?, ?, ?, ?)
            """,
            (input_id, output_text, model_name, rag_output, cve_data_str),
        )
        conn.commit()
        return cursor.lastrowid


# 2. Get row by id
def get_row_by_id(row_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.model_name, i.input, s.output, s.rag_output, s.cve_data, s.LLM_score, s.Human_score
            FROM Scores s
            JOIN Inputs i ON s.input_id = i.id
            WHERE s.id = ?
            """,
            (row_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


# 3. Delete entry by id
def delete_entry(row_id: int) -> bool:
    print(f"Attempting to delete entry with id: {row_id}")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM Scores WHERE id = ?", (row_id,))
        conn.commit()
        return cursor.rowcount > 0


# 4. Get random set of rows where Human_score IS NULL
def get_random_unscored_by_human(limit: int, model: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.model_name, i.input, s.output, s.rag_output, s.cve_data, s.LLM_score, s.Human_score
            FROM Scores s
            JOIN Inputs i ON s.input_id = i.id
            WHERE s.Human_score IS NULL AND s.model_name = ?
            ORDER BY RANDOM()
            LIMIT ?
            """,
            (model, limit),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# 5. Get all rows where LLM_score IS NULL
def get_all_unscored_by_llm(model: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.model_name, i.input, s.output, s.rag_output, s.cve_data, s.LLM_score, s.Human_score
            FROM Scores s
            JOIN Inputs i ON s.input_id = i.id
            WHERE s.LLM_score IS NULL AND s.model_name = ? AND s.output != 'tmp'
            """,
            (model,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# 6. Insert/update LLM_score for a given id
def update_llm_score(row_id: int, score: float, reasoning: str = None) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE Scores
            SET LLM_score = ?, reasoning = ?
            WHERE id = ?
            """,
            (score, reasoning, row_id),
        )
        conn.commit()
        return cursor.rowcount > 0


# 7. Insert/update Human_score for a given id
def update_human_score(row_id: int, score: float) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE Scores
            SET Human_score = ?
            WHERE id = ?
            """,
            (score, row_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_LLM_scores_by_model(model: str) -> int:
    # Sets all LLM_score values to NULL for the specified model
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE Scores
            SET LLM_score = NULL
            WHERE model_name = ?
            """,
            (model,),
        )
        conn.commit()
        return cursor.rowcount


# 8. Get average Human_score and LLM_score
def get_average_scores(model: str) -> Tuple[Optional[float], Optional[float]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                AVG(Human_score),
                AVG(LLM_score)
            FROM Scores
            WHERE (Human_score IS NOT NULL AND model_name = ?)
               OR (LLM_score IS NOT NULL AND model_name = ?)
            """,
            (model, model),
        )
        result = cursor.fetchone()
        return result if result else (None, None)


def delete_all_entries_by_model(model: str) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            DELETE FROM Scores
            WHERE model_name = ?
            """,
            (model,),
        )
        conn.commit()
        return cursor.rowcount


def reset():
    # delete the data in "output", "rag_output", "cve_data", "Human_score", and "LLM_score" for all entries in the database
    verify = input("Are you sure you want to reset the database? This will delete all output, rag_output, cve_data, Human_score, and LLM_score values. Type 'RESET' to confirm: ")
    if verify != "RESET":
        print("Reset cancelled.")
        exit(0)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE Scores
            SET output = 'tmp',
                rag_output = NULL,
                cve_data = NULL,
                Human_score = NULL,
                LLM_score = NULL,
                reasoning = NULL
            """,
        )
        conn.commit()
        return cursor.rowcount


def get_missing_output_entries(model: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.model_name, i.input, s.output, s.rag_output, s.cve_data, s.LLM_score, s.Human_score
            FROM Scores s
            JOIN Inputs i ON s.input_id = i.id
            WHERE s.output = 'tmp' AND s.model_name = ?
            """,
            (model,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def insert_output_by_id(row_id: int, output_text: str, rag_result: dict, cve_data: dict) -> bool:
    rag_output = rag_result if rag_result else None
    cve_data = str(cve_data) if cve_data else None
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE Scores
            SET output = ?,
                rag_output = ?,
                cve_data = ?
            WHERE id = ?
            """,
            (output_text, rag_output, cve_data, row_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def find_average_scores_by_model(model: str) -> Tuple[Optional[float], Optional[float]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                AVG(Human_score),
                AVG(LLM_score)
            FROM Scores
            WHERE (Human_score IS NOT NULL AND model_name = ?)
               OR (LLM_score IS NOT NULL AND model_name = ?)
            """,
            (model, model),
        )
        result = cursor.fetchone()
        return result if result else (None, None)


if __name__ == "__main__":
    model = "mistral-nemo-cve2"
    avg_human, avg_llm = find_average_scores_by_model(model)
    print(f"Average scores for model '{model}': Human_score = {avg_human}, LLM_score = {avg_llm}")