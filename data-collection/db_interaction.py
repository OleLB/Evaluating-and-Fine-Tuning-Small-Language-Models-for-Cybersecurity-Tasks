
import sqlite3
import json
from typing import Dict, Any, List, Optional

DB_PATH = "data-collection/cve_database.db"

def add_cve(cve_data: Dict[str, Any], db_path: str = DB_PATH) -> None:
    """
    Insert or update a CVE object in the SQLite database.
    Safely serializes complex NVD / nvdlib objects.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT OR REPLACE INTO cves (
            cve_id,
            description,
            exploit_instruction,
            label_cwe,
            label_attack,
            severity,
            known_vulnerable_software,
            cpe_list
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cve_data.get("cve_id"),
            cve_data.get("description"),
            cve_data.get("exploit_instruction"),
            cve_data.get("label_cwe"),
            cve_data.get("label_attack"),
            str(cve_data.get("severity")),
            json.dumps(
                cve_data.get("known_vulnerable_software", []),
                default=str,
                ensure_ascii=False,
            ),
            json.dumps(
                cve_data.get("cpe_list", []),
                default=str,
                ensure_ascii=False,
            ),
        ),
    )

    conn.commit()
    conn.close()


def delete_cve(cve_id: str, db_path: str = DB_PATH) -> None:
    """Delete a CVE entry by CVE ID (e.g. CVE-2021-26855)."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM cves WHERE cve_id = ?",
        (cve_id,)
    )

    conn.commit()
    conn.close()



def get_cve_by_id(cve_id: str, db_path: str = DB_PATH) -> Optional[Dict[str, Any]]:
    """Retrieve a single CVE by its CVE ID."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM cves WHERE cve_id = ?",
        (cve_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return {
        "cve_id": row[0],
        "description": row[1],
        "exploit_instruction": row[2],
        "label_cwe": row[3],
        "label_attack": row[4],
        "severity": row[5],
        "known_vulnerable_software": json.loads(row[6]),
        "cpe_list": json.loads(row[7]),
    }



def get_all_cves(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieve all CVE entries from the database."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM cves")
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "cve_id": row[0],
            "description": row[1],
            "exploit_instruction": row[2],
            "label_cwe": row[3],
            "label_attack": row[4],
            "severity": row[5],
            "known_vulnerable_software": json.loads(row[6]),
            "cpe_list": json.loads(row[7]),
        })

    return results