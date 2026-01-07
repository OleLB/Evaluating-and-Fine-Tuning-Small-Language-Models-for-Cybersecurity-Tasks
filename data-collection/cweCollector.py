from cwe import Database

def collect_cwe(cwe_id: int):
    db = Database()
    cwe = db.get(cwe_id)
    return cwe
# Output: "Server-Side Request Forgery (SSRF)"


if __name__ == "__main__":
    cwe = collect_cwe(918)  # Example CWE ID
    if cwe:
        print(f"Name: {cwe.name}")
        print(f"Description: {cwe.description}")