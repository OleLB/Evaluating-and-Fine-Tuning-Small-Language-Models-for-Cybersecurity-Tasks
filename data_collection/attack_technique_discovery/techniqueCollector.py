from db.db_interaction import get_all_cves
from data_collection.attack_technique_discovery.gpt_interact import get_attack_technique

# Collect techniques for all CVEs in the database
# python -m data_collection.technique_labelling.techniqueCollector

def update_cve_with_technique(cve_id, technique, db_path="data_collection/db/cve_database.db"):
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE cves SET label_attack = ? WHERE cve_id = ?",
        (technique, cve_id)
    )

    conn.commit()
    conn.close()


def collect_techniques_for_all_cves(range=None):
    cves = get_all_cves()
    if range:
        cves = cves[range[0]:range[1]]
    cost_sum = 0.0
    for cve in cves:
        cve_id = cve['cve_id']
        print(f"Processing {cve_id}...")
        technique, usage = get_attack_technique(cve)
        # print(f"CVE {cve_id} mapped to technique: {technique}")
        cost_sum += usage["total_cost"]
        update_cve_with_technique(cve_id, technique)
    print(f"Total cost for processing all CVEs: {cost_sum}")


def test():
    cves = get_all_cves()
    sample_cve = cves[0]
    technique, usage = get_attack_technique(sample_cve)
    print(f"Sample CVE {sample_cve['cve_id']} mapped to technique: {technique}")
    print(f"Token usage: {usage}")


if __name__ == "__main__":
    collect_techniques_for_all_cves((4000, 5000))
    # test()
