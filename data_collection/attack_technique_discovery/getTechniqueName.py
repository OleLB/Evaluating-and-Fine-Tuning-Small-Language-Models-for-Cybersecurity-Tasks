"""This script is used to fetch technique names based on their IDs."""

from mitreattack.stix20 import MitreAttackData
from utils.querySQLite import getConnection, getAllCVEs
from pathlib import Path

DATABASE_PATH = "db/cve_database.db"

# 1. Define the path relative to THIS script file
script_dir = Path(__file__).resolve().parent
json_path = script_dir / "enterprise-attack.json"

# 2. Check if it exists before trying to load
if not json_path.exists():
    print(f"Error: Could not find the file at {json_path}")
    print("Please download enterprise-attack.json from MITRE and place it in the script directory.")
else:
    # 3. Initialize with the absolute path
    mitre_data = MitreAttackData(str(json_path))
    print("Successfully loaded MITRE ATT&CK data.")


def getTechniqueName(t_id: str):
    # Fetch the object by its ATT&CK ID
    technique = mitre_data.get_object_by_attack_id(t_id, "attack-pattern")
    if technique:
        return technique.name
    return "Technique ID not found."

def updateDatabaseWithTechniqueNames():
    # 1. Get all rows from database
    cves = getAllCVEs(DATABASE_PATH)

    # 2. For each row, fetch the technique name using getTechniqueName (technique id is in the "label_attack" column)
    for cve in cves:
        cve_id = cve[0]  # Assuming the CVE ID is in the first column
        attack_vector = cve[4]  # Assuming the attack vector (MITRE ATT&CK ID) is in the sixth column
        # print(f"Processing CVE ID: {cve_id}, Attack Vector: {attack_vector}")
        technique_name = getTechniqueName(attack_vector)
        # print(f"CVE ID: {cve_id}, Attack Vector: {attack_vector}, Technique Name: {technique_name}")

        # 3. Insert the technique name into a new column "attack_technique_name" in the cves table, matching on the CVE ID
        try:
            conn = getConnection(DATABASE_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE cves SET attack_technique_name = ? WHERE cve_id = ?", (technique_name, cve_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error updating database for CVE ID {cve_id}: {e}")
        


if __name__ == "__main__":
    updateDatabaseWithTechniqueNames()
    # test = getTechniqueName("T1190")
    # print(test)