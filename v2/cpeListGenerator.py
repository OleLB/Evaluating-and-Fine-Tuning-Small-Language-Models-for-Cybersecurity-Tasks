from cpeParser import cpe_to_human
"""
This file reads all CPEs from the CVE database (column 'cpe_list' in table 'cves') and stores them to a csv file ../db/cpe_list.csv', one CPE per line with the corresponding CVE ID. 
CVE-2021-12345,cpe:2.3:a:example:software:1.0:*:*:*:*:*:*:*
"""

"""
database data structure example:
cpe_list column example:
[
  "cpe:2.3:a:acronis:cyber_protect:*:*:*:*:*:*:*:*",
  "cpe:2.3:a:acronis:cyber_protect:16:-:*:*:*:*:*:*",
  "cpe:2.3:a:acronis:cyber_protect:16:update1:*:*:*:*:*:*",
  "cpe:2.3:a:acronis:cyber_protect:16:update2:*:*:*:*:*:*"
]
"""

from cpeParser import cpe_to_human
import sqlite3
import csv
import json

DATABASE = "../db/cve_database.db"

def extract_cpe_list(cursor):
    cursor.execute("SELECT cve_id, cpe_list FROM cves")
    rows = cursor.fetchall()
    
    cpe_entries = []
    for row in rows:
        cve_id = row[0]
        cpe_list_str = row[1]
        
        # Parse as JSON instead of manually splitting by comma
        try:
            cpe_list = json.loads(cpe_list_str)
        except json.JSONDecodeError:
            print(f"Failed to parse CPE list for {cve_id}: {cpe_list_str}")
            continue
        
        for cpe in cpe_list:
            try:
                human_readable = cpe_to_human(cpe)
            except ValueError as e:
                print(f"malformed CPE: {cpe}")
                print(f"Error: {e}")
                exit()
            cpe_entries.append((cve_id, cpe, human_readable))
    
    return cpe_entries

def save_cpe_to_csv(cpe_entries, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['CVE_ID', 'CPE', 'HUMAN_READABLE'])
        writer.writerows(cpe_entries)

if __name__ == "__main__":
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    cpe_entries = extract_cpe_list(cursor)
    save_cpe_to_csv(cpe_entries, '../db/cpe_list.csv')
    
    conn.close()
    print(f"Saved {len(cpe_entries)} CPE entries to ../db/cpe_list.csv")