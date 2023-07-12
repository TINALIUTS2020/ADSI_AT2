import re
from unicodedata import normalize

def clean_string(string: str) -> str:
    """Clean a string.

    Args:
        string (str): String to be cleaned.

    Returns:
        str: Cleaned string.
    """
    # tempting to make a wrapper
    string = normalize('NFKD', string).encode('ascii', errors='ignore').decode('utf-8')
    string = string.lower().strip()
    string = re.sub(r"[^\w\s\d/]", "", string)
    string = re.sub(r"\s+", " ", string).strip()

    return string