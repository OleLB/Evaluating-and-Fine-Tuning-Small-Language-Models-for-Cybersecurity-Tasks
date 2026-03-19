"""
This script collects CWE data from a CWE id using the cwe library.
It is automatically called by record_collector.py
"""

from cwe import Database

def collect_cwe(cwe_id: int):
    """
    Collect CWE data for a given CWE ID.
    Args:
        cwe_id (int): The CWE ID to collect data for.
    Returns:
        A CWE object
    """
    db = Database()
    cwe = db.get(cwe_id)()
    return cwe
# Output: "Server-Side Request Forgery (SSRF)"


if __name__ == "__main__":
    cwe = collect_cwe(918)  # Example CWE ID
    if cwe:
        print(f"Name: {cwe.name}")
        print(f"Description: {cwe.description}")