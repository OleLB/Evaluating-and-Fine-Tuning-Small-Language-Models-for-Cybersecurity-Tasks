import sqlite3

db_folder_path = "db"

def initialize_new_database(db_name="scores.db"):
    conn = sqlite3.connect(f"{db_folder_path}/{db_name}")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input TEXT NOT NULL UNIQUE
        );
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            input_id INTEGER NOT NULL,
            output TEXT NOT NULL,
            rag_output TEXT,
            cve_data TEXT,
            LLM_score REAL,
            Human_score REAL,
            reasoning TEXT,
            FOREIGN KEY (input_id) REFERENCES Inputs(id)
        );
    """)

    conn.commit()
    conn.close()
    print(f"New database '{db_name}' initialized successfully.")

if __name__ == "__main__":
    initialize_new_database()