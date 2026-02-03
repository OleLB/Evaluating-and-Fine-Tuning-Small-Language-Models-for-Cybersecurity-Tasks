import sqlite3

DATABASE = "../db/cve_database.db"

def getDescription(cve_id: str, database = DATABASE) -> str:
    """Retrieve the description of a CVE from the database."""
    try:
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute("SELECT description FROM cves WHERE cve_id = ?", (cve_id,))
        row = cursor.fetchone()
        conn.close()
    except Exception as e:
        return f"Error connecting to CVE database: {e}"
    
    if row:
        return row[0]
    else:
        return "CVE not found in the database."
    

def getRowCount(table_name: str, database = DATABASE) -> int:
    """Get the number of rows in a specified table."""
    try:
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return -1
    

def getRandomCVEs():
    """Retrieve 200 random CVE IDs from the database."""
    query = "SELECT * FROM cves ORDER BY RANDOM() LIMIT 200;"
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return []