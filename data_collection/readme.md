CVEs with: vulnStatus = "Rejected" are ignored by this system.

!!Data should already be collected and stored in db/cve_database.db, so there is no need to run anything in this folder!!

# Instructions


### Get CVE entries
Run the record collector "record_collector.py", this will collect CVE entries and store them in a local database "/db/cve_database.db"
    - Specify a year range


### Get attack techniques
Collect MITRE ATT&CK attack techniques with:
python -m data_collection.technique_labelling.techniqueCollector      # Will collect attack technique id's
python -m data_collection.attack_technique_discovery.getTechniqueName    # Will find technique common names by their id's


### Get mitigation techniques
Get MITRE ATT&CK mitigation techniqus with:
python -m data_collection.mitigation_technique_discovery.mitigationCollector