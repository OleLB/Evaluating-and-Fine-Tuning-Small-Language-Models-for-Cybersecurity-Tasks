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
    

def getCVEInfo(cve_id: str, database = DATABASE) -> str:
    """Retrieve detailed information about a CVE from the database."""
    try:
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        # Fix: Select all needed columns, not just description
        cursor.execute("""
            SELECT cve_id, description, cwe_name, cwe_description, 
                label_attack, severity, known_vulnerable_software, 
                cpe_list, mitigation 
            FROM cves 
            WHERE cve_id = ?
        """, (cve_id,))
        row = cursor.fetchone()
        conn.close()
    except Exception as e:
        return f"Error connecting to CVE database: {e}"
    
    if row:
        formatted_info = {
            "CVE_ID": row[0],
            "Description": row[1],
            "CWE_Name": row[2],
            "CWE_Description": row[3],
            "Attack_Technique": row[4],
            "Severity": row[5],
            "Known_Vulnerable_Software": row[6],
            "CPE_List": row[7],
            "Mitigation": row[8]
        }
        return formatted_info
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
    

def getRandomCVEs(limit=200):
    """Retrieve 200 random CVE IDs from the database."""
    query = f"SELECT * FROM cves ORDER BY RANDOM() LIMIT {limit};"
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
    
def addColumn():
    # Add a column "attack_technique_name" to the cves table
    try:
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute("ALTER TABLE cves ADD COLUMN attack_technique_name TEXT")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return
    
def getAllCVEs(database = DATABASE):
    """Get all rows from a specified table."""
    try:
        conn = sqlite3.connect(database)
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM cves")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return []
    

def getConnection(database = DATABASE):
    """Get a connection to the SQLite database."""
    try:
        conn = sqlite3.connect(database)
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None
    
if __name__ == "__main__":
    addColumn()