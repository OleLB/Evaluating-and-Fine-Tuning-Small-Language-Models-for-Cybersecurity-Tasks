import sqlite3
from typing import List, Optional, Tuple, Dict, Any

DB_PATH = "../db/scores.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# 1. Add input/output pair, return created row id
def add_entry(input_text: str, output_text: str, model_name: str) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO Scores (input, output, model_name)

            VALUES (?, ?, ?)
            """,
            (input_text, output_text, model_name),
        )
        conn.commit()
        return cursor.lastrowid


# 2. Get row by id
def get_row_by_id(row_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Scores WHERE id = ?", (row_id,))
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
            SELECT * FROM Scores
            WHERE Human_score IS NULL AND model_name = ?
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
            "SELECT * FROM Scores WHERE LLM_score IS NULL AND model_name = ?", (model,)
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


# 6. Insert/update LLM_score for a given id
def update_llm_score(row_id: int, score: float) -> bool:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE Scores
            SET LLM_score = ?
            WHERE id = ?
            """,
            (score, row_id),
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
            (model, model)
        )
        result = cursor.fetchone()
        return result if result else (None, None)


if __name__ == "__main__":
    # print("Database interaction module ready.")
    score = get_average_scores("mistral-nemo-cve2")
    print(f"Average scores for model 'mistral-nemo-cve2': {score}")