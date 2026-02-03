"""Python script to modify an existing database"""

add_table_query = '''
    CREATE TABLE IF NOT EXISTS attack_technique (
        cve_id TEXT PRIMARY KEY,
        llama3_1 TEXT,
        deepseek_r1 TEXT,
        qwen3 TEXT,
        FOREIGN KEY (cve_id) REFERENCES cves (cve_id)
    )
'''

expand_table_query = """
ALTER TABLE cves
ADD COLUMN mitigation TEXT;
"""

import sqlite3

DB_PATH = "data_collection/db/cve_database.db"

def execute_query(query: str, params: tuple = ()) -> None:
    """Execute a given SQL query with optional parameters."""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(query, params)

    conn.commit()
    conn.close()


def reset_tables():
    """Reset tables by using DELETE statements."""
    execute_query("DELETE FROM cves")
    execute_query("DELETE FROM attack_technique")


def expand_cve_table():
    """Expand the 'cves' table to include a 'mitigation' column."""
    execute_query(expand_table_query)


if __name__ == "__main__":
    # reset_tables()
    # expand_cve_table()
    execute_query(add_table_query)
    # print("Database modified: 'attack_technique' table added.")