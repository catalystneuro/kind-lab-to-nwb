import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path


def extract_session_datetime_from_xml(session_dir: str) -> datetime:
    """
    Search for 'settings.xml' under session_dir, parse SETTINGS/INFO/DATE,
    and return a datetime object.
    """
    base = Path(session_dir)
    xml_path = sorted(base.rglob("settings.xml"))[0]
    root = ET.parse(xml_path).getroot()

    date_elem = root.find("./INFO/DATE")
    date_text = date_elem.text

    fmt = "%d %b %Y %H:%M:%S"

    return datetime.strptime(date_text, fmt)
