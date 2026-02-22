import sqlite3

db_folder_path = "../db"

def initialize_database(db_name="scores.db"):
    conn = sqlite3.connect(db_folder_path + "/" + db_name)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            input TEXT NOT NULL,
            output TEXT NOT NULL,
            LLM_score REAL,
            Human_score REAL
        );
    """)

    conn.commit()
    conn.close()
    print(f"Database '{db_name}' initialized successfully.")

if __name__ == "__main__":
    initialize_database()
