import sqlite3

DB_PATH = "data-collection/cve_database.db"

def init_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cves (
            cve_id TEXT PRIMARY KEY,
            description TEXT,
            cwe_name TEXT,
            cwe_description TEXT,
            label_attack TEXT,
            severity TEXT,
            known_vulnerable_software TEXT,
            cpe_list TEXT
        )
    """)

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print("SQLite database initialized.")
