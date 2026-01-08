from pdb import main
import nvdlib
from datetime import datetime, timedelta
from requests.exceptions import HTTPError
import random
import time
from db_interaction import add_cve, delete_cve, get_cve_by_id, get_all_cves
from cweCollector import collect_cwe

"""
This file uses the CVE API to collect CVE data and stores it in an sqlite database.
"""

API_KEY = "8c14429b-01ef-420f-aa7c-2b0ea95c853a"

def get_highest_nvd_severity(r):
    best_score = None
    best_severity = "N/A"

    metric_sets = [
        getattr(r.metrics, "cvssMetricV31", None),
        getattr(r.metrics, "cvssMetricV30", None),
        getattr(r.metrics, "cvssMetricV2", None),
    ]

    for metrics in metric_sets:
        if not metrics:
            continue

        for m in metrics:
            if m.source != "nvd@nist.gov":
                continue

            score = getattr(m.cvssData, "baseScore", None)
            severity = getattr(m.cvssData, "baseSeverity", None)

            if score is None or severity is None:
                continue

            if best_score is None or score > best_score:
                best_score = score
                best_severity = severity

    return best_severity


def safe_searchCVE(**kwargs):
    while True:
        try:
            return nvdlib.searchCVE(**kwargs)

        except HTTPError as e:
            status = e.response.status_code if e.response else None

            # Rate limit or temporary server error
            if status in (429, 500, 502, 503, 504):
                wait = random.uniform(5, 15)
                print(f"[!] Rate limited ({status}), sleeping {wait:.1f}s")
                time.sleep(wait)
                continue

            # Anything else is a real error
            raise


def get_cves_multi_year(year, per_year=2000, instant_process=True):
    all_cves = []

    collected = 0
    start = datetime(year, 1, 1, 0, 0)
    end = datetime(year, 12, 31, 23, 59)

    while start < end and collected < per_year:
        window_end = min(start + timedelta(days=120), end)

        for cve in safe_searchCVE(
            pubStartDate=start.strftime("%Y-%m-%d %H:%M"),
            pubEndDate=window_end.strftime("%Y-%m-%d %H:%M"),
            key=API_KEY,
            noRejected=True
        ):
            all_cves.append(cve)
            collected += 1
            if instant_process:
                process_cve(cve)
            time.sleep(0.2)  # To avoid hitting rate limits
            if collected >= per_year:
                break

        start = window_end + timedelta(minutes=1)

    print(f"{year}: collected {collected} CVEs")

    return all_cves


def format_cve_data(r):
    """
    Args:
    r: nvdlib CVE object.

    Returns:
    A dictionary with CVE data.
    """
    cwe_label = "N/A"
    try:
        cwe_label = next((cwe.value for cwe in r.cwe if cwe.lang == "en"), "N/A")
    except Exception:
        pass

    cwe_number = int(cwe_label.split("-")[1]) if cwe_label.startswith("CWE-") else None

    cwe_data = collect_cwe(cwe_number)

    cve_severity = get_highest_nvd_severity(r)

    criteria = []
    try:
        criteria = [
            node
            for x in r.configurations
            for node in x.nodes
        ]
    except Exception:
        print("Error extracting configurations for CVE:", r.id)

    
    cpe = []
    try:
        cpe = [cpe.criteria for cpe in r.cpe]
    except Exception:
        print("Error extracting CPEs for CVE:", r.id)

    cve_data = {
        "cve_id": r.id,
        "description": next((desc.value for desc in r.descriptions if desc.lang == "en"), "N/A"),
        "cwe_name": cwe_label,
        "cwe_description": cwe_data.description if cwe_data else "N/A",
        "label_attack": "Mitre ATT&CK mapping not implemented",
        "severity": cve_severity,
        "known_vulnerable_software": criteria,
        "cpe_list": cpe
    }

    return cve_data


def test():
    """
    Make a sample request to the CVE API and print the results.
    """
    CVE_ID = 'CVE-2021-26855'

    try:
        # delete existing CVE entries in the database
        delete_cve(CVE_ID)
    except Exception as e:
        print(f"Error deleting CVE: {e}")

    cve_data = format_cve_data(CVE_ID)

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


def process_cve(cve):
    cve_data = format_cve_data(cve)
    try:
        already_exists = get_cve_by_id(cve_data['cve_id'])
        if already_exists:
            print(f"CVE {cve_data['cve_id']} already exists in the database. Skipping.")
        else:
            add_cve(cve_data)
    except Exception as e:
        print(f"Error checking existence of CVE: {e}")


if __name__ == "__main__":
    years = [2023]
    for year in years:
        CVE_list = get_cves_multi_year(year, per_year=1000, instant_process=True)

    print("CVE data collection complete.")