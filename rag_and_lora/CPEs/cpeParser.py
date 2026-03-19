from typing import Dict

CPE_ESCAPE_MAP = {
    r"\*": "*",
    r"\?": "?",
    r"\!": "!",
    r"\,": ",",
    r"\\": "\\",
    r"\_": "_",
    r"\&": "&",
}

CPE_2_3_FIELDS = [
    "part",
    "vendor",
    "product",
    "version",
    "update",
    "edition",
    "language",
    "sw_edition",
    "target_sw",
    "target_hw",
    "other"
]

PART_MAP = {
    "a": "Application",
    "o": "Operating System",
    "h": "Hardware"
}

def split_cpe_components(cpe: str) -> list:
    """
    Split CPE string by colons, but not when the colon is escaped with a backslash.
    Handles cases like \\: properly.
    """
    if not cpe.startswith("cpe:2.3:"):
        raise ValueError("Not a valid CPE 2.3 string")
    
    # Remove the 'cpe:2.3:' prefix
    remainder = cpe[8:]
    
    components = []
    current = []
    i = 0
    
    while i < len(remainder):
        if remainder[i] == '\\' and i + 1 < len(remainder):
            # Escaped character - add both the backslash and next char
            current.append(remainder[i:i+2])
            i += 2
        elif remainder[i] == ':':
            # Unescaped colon - this is a field separator
            components.append(''.join(current))
            current = []
            i += 1
        else:
            current.append(remainder[i])
            i += 1
    
    # Don't forget the last component
    if current:
        components.append(''.join(current))
    
    return components

def unescape_cpe(value: str) -> str:
    """
    Unescape CPE special characters.
    """
    for esc, char in CPE_ESCAPE_MAP.items():
        value = value.replace(esc, char)
    return value

def parse_cpe(cpe: str) -> Dict[str, str]:
    """
    Parse a CPE 2.3 string into a dictionary, handling escaped characters (\\, \\, & etc).
    """
    # Split by colons while respecting escapes
    parts = split_cpe_components(cpe)
    
    # Validate length
    if len(parts) != len(CPE_2_3_FIELDS):
        raise ValueError(f"Malformed CPE 2.3 string: expected {len(CPE_2_3_FIELDS)} fields, got {len(parts)}")
    
    # Unescape each field
    parts = [unescape_cpe(p) for p in parts]
    
    # Return as dictionary
    return dict(zip(CPE_2_3_FIELDS, parts))

def humanize_value(value: str) -> str:
    """
    Convert CPE wildcards and escaped values into human-friendly text.
    """
    if value == "*":
        return "any"
    if value == "-":
        return "not specified"
    return value.replace("_", " ")

def cpe_to_human(cpe: str) -> str:
    parsed = parse_cpe(cpe)
    
    part = PART_MAP.get(parsed["part"], parsed["part"])
    vendor = humanize_value(parsed["vendor"])
    product = humanize_value(parsed["product"])
    version = humanize_value(parsed["version"])
    
    description = [
        f"{part}",
        f"'{product}'",
        f"by {vendor}",
    ]
    
    # Handle target_sw specially for readability
    target_sw = parsed.get("target_sw", "*")
    if target_sw not in ("*", "-"):
        description.append(f"on {humanize_value(target_sw)}")
    
    if version not in ("any", "not specified"):
        description.append(f"(version {version})")
    else:
        description.append("(all versions)")
    
    constraints = []
    for field in ["update", "edition", "language", "sw_edition", "target_hw", "other"]:
        value = parsed[field]
        if value not in ("*", "-"):
            constraints.append(f"{field.replace('_', ' ')} = {humanize_value(value)}")
    
    result = " ".join(description)
    
    if constraints:
        result += " with constraints: " + ", ".join(constraints)
    
    return result

if __name__ == "__main__":
    # Test with original
    cpe2 = "cpe:2.3:a:sonaar:mp3_audio_player_for_music\\,_radio_\\&_podcast:*:*:*:*:*:wordpress:*:*"
    print(cpe_to_human(cpe2))
