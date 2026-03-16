import re


def clean_cve_data(cve_data: dict) -> dict:
    """
    Entrypoint. Cleans and simplifies raw CVE data for consumption by a small LLM.
    
    Usage:
        raw   = getCVEInfo(cve_id)
        clean = clean_cve_data(raw)
        ctx   = build_context_block(clean, rag_output)
    """
    if not cve_data or isinstance(cve_data, str):
        return {}

    # Unwrap outer {cve_id: {data}} layer if present
    if len(cve_data) == 1 and "CVE_ID" not in cve_data:
        cve_data = next(iter(cve_data.values()))

    cleaned = {}

    for field in [
        "CVE_ID",
        "Description",
        "CWE Name",
        "CWE Description",
        "Associated MITRE ATT&CK Technique",
        "Severity",
        "MITRE ATT&CK Technique Name",
    ]:
        if field in cve_data:
            cleaned[field] = cve_data[field]

    cleaned["Affected Platforms"]   = _parse_vulnerable_software(cve_data.get("Known Vulnerable Software", ""))
    cleaned["Mitigation Techniques"] = _parse_mitigations(cve_data.get("Mitigation Techniques", ""))

    return cleaned


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_vulnerable_software(raw: str) -> list[str]:
    """
    Extracts human-readable platform names from the raw CPE blob.
    Prefers versioned labels (e.g. 'Redhat Enterprise Linux (up to 9.6)')
    and falls back to unversioned ones when no version bound is present.
    """
    versioned:   dict[str, str] = {}   # key -> "Vendor Product (up to X)"
    unversioned: dict[str, str] = {}   # key -> "Vendor Product"

    # CPE format: cpe:2.3:<type>:<vendor>:<product>:<version>:...
    cpe_re = re.compile(r"cpe:2\.3:[ao]:([^:]+):([^:]+):([^:,\'\"]+)")

    # Look for a versionEnd qualifier sitting near each CPE string
    versioned_re = re.compile(
        r"cpe:2\.3:[ao]:([^:]+):([^:]+):([^:,\'\"]+).*?"
        r"'versionEnd(?:Including|Excluding)':\s*'([^']+)'",
        re.DOTALL,
    )

    for match in versioned_re.finditer(raw):
        vendor, product, _, version_end = match.groups()
        label = _fmt_platform(vendor, product)
        versioned[label] = f"{label} (up to {version_end})"

    for match in cpe_re.finditer(raw):
        vendor, product, _ = match.groups()
        label = _fmt_platform(vendor, product)
        if label not in versioned:
            unversioned[label] = label

    platforms = list(versioned.values()) + [
        v for k, v in unversioned.items() if k not in versioned
    ]

    return sorted(set(platforms)) if platforms else ["Not specified"]


def _fmt_platform(vendor: str, product: str) -> str:
    return f"{vendor.replace('_', ' ').title()} {product.replace('_', ' ').title()}".strip()


def _parse_mitigations(raw: str) -> list[str]:
    """
    Returns mitigation codes as a clean list.
    No mapping is applied — codes are presented as-is.
    """
    if not raw:
        return ["None specified"]
    return [c.strip() for c in raw.split(",") if c.strip()]