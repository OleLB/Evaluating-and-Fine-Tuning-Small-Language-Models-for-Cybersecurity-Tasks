import nvdlib

from db_interaction import add_cve, delete_cve, get_cve_by_id, get_all_cves

"""
This file uses the CVE API to collect CVE data and stores it in an sqlite database.
"""

if __name__ == "__main__":
    """
    Make a sample request to the CVE API and print the results.
    """

    try:
        # delete existing CVE entries in the database
        delete_cve("CVE-2021-26855")
    except Exception as e:
        print(f"Error deleting CVE: {e}")

    r = nvdlib.searchCVE(cveId='CVE-2021-26855')[0]

    """
    Save the following info:
        {
        "cve_id": "CVE-2024-1234",
        "instruction": "Summarize this vulnerability and identify the attack technique.",
        "context": "A remote code execution vulnerability exists in [Software] via the [Parameter]...",
        "label_cwe": "CWE-94",
        "label_attack": "T1210 (Exploitation of Remote Services)",
        "severity": "Critical"
        }    
    """
    # print(r)

    # exit()

    cve_data = {
        "cve_id": r.id,
        "description": next((desc.value for desc in r.descriptions if desc.lang == "en"), "N/A"),
        "exploit_instruction": "Not implemented",
        "label_cwe": next((cwe.value for cwe in r.cwe if cwe.lang == "en"), "N/A") if r.cwe else "N/A",
        "label_attack": "Mitre ATT&CK mapping not implemented",
        #! Severity extraction not working
        "severity": next((score.cvssData.baseScore for score in r.metrics.cvssMetricV31 if score.source == "nvd@nist.gov"), "N/A") if r.metrics.cvssMetricV31 else "N/A",
        "known_vulnerable_software": [
            node
            for x in r.configurations
            for node in x.nodes
        ],
        "cpe_list": [cpe.criteria for cpe in r.cpe]
    }

    # Add the CVE to the database
    try:
        add_cve(cve_data)
        print(f"CVE {cve_data['cve_id']} added to the database.")
    except Exception as e:
        print(f"Error adding CVE: {e}")

    # Retrieve the CVE from the database
    try:
        retrieved_cve = get_cve_by_id(cve_data['cve_id'])
        if retrieved_cve:
            print(f"Retrieved CVE from database: {retrieved_cve['cve_id']}")
    except Exception as e:
        print(f"Error retrieving CVE: {e}")

    # Retrieve all CVEs from the database
    try:
        all_cves = get_all_cves()
        print(f"Total CVEs in database: {len(all_cves)}")
    except Exception as e:
        print(f"Error retrieving all CVEs: {e}")

    
    