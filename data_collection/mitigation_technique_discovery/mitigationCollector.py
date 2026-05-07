"""
Collects mitigation techniques for CVEs based on the MITRE ATT&CK technique ID associated with each CVE
Script must be started manually
Automatically enriches all entries in database.
python -m data_collection.mitigation_technique_discovery.mitigationCollector
"""

from mitreattack.stix20 import MitreAttackData
from db.db_interaction import get_all_cves
import sqlite3

DATABASE_PATH = "data_collection/db/cve_database.db"
ATTACK_STIX_PATH = "data_collection/db/enterprise-attack.json"

def get_mitigation_description(mitigation_entries):
    mitigation_descriptions = []
    for entry in mitigation_entries:
        # entry is a dict with keys 'object' and 'relationships'
        # entry['object'] contains the mitigation (course-of-action) object
        mitigation = entry['object']
        description = mitigation.get("description", "")
        if description:
            mitigation_descriptions.append(description)
    return "\n\n".join(mitigation_descriptions)


def load_attack_data():
    return MitreAttackData(ATTACK_STIX_PATH)

def find_mitigation_for_all_cves(range=None):
    """
    For all CVEs in the database, find related mitigations (by using MITRE ATT&CK technique from 'label_attack' column in database)
    based on associated techniques. 
    Store the mitigation IDs in the 'mitigation' column of the 'cves' table.
    """
    attack_data = load_attack_data()
    
    # Get all techniques and build lookup by external ID
    all_techniques = attack_data.get_techniques(remove_revoked_deprecated=True)
    technique_by_external_id = {
        ref["external_id"]: t
        for t in all_techniques
        for ref in t.get("external_references", [])
        if ref.get("source_name") == "mitre-attack" and "external_id" in ref
    }
    
    cves = get_all_cves()
    
    if range:
        cves = cves[range[0]:range[1]]
    
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    for cve in cves:
        cve_id = cve["cve_id"]
        technique_id = cve["label_attack"]
        
        try:
            # Get technique object by external ID (e.g., "T1234")
            technique = technique_by_external_id.get(technique_id)
            if not technique:
                print(f"Technique {technique_id} not found for CVE {cve_id}")
                continue
            
            # Use the built-in method to get mitigations for this technique
            # This returns a list of dictionaries with 'object' and 'relationships' keys
            mitigation_entries = attack_data.get_mitigations_mitigating_technique(technique["id"])
            
            mitigation_ids = []
            for entry in mitigation_entries:
                # entry is a dict with keys 'object' and 'relationships'
                # entry['object'] contains the mitigation (course-of-action) object
                mitigation = entry['object']
                
                # Extract the external ID (e.g., "M1234") from external_references
                for ref in mitigation.get("external_references", []):
                    if ref.get("source_name") == "mitre-attack" and "external_id" in ref:
                        mitigation_ids.append(ref["external_id"])
                        break
            
            # Join multiple mitigation IDs with comma separation
            mitigation_text = ", ".join(mitigation_ids)
            
            cursor.execute(
                "UPDATE cves SET mitigation = ? WHERE cve_id = ?",
                (mitigation_text, cve_id)
            )
            
            print(f"Updated CVE {cve_id} with {len(mitigation_ids)} mitigations: {mitigation_text}")
            
        except Exception as e:
            print(f"Error processing CVE {cve_id}: {e}")
            import traceback
            traceback.print_exc()
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    find_mitigation_for_all_cves()