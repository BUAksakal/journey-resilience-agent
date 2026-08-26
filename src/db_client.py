"""DB Timetables API uzerine ince bir sarmalayici."""

import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

import requests

from src import config

TIME_FMT = "%y%m%d%H%M"


def _delay_minutes(planned: Optional[str], changed: Optional[str]) -> float:
    if not planned or not changed:
        return 0.0
    return (datetime.strptime(changed, TIME_FMT) - datetime.strptime(planned, TIME_FMT)).total_seconds() / 60


def find_station(pattern: str):
    r = requests.get(f"{config.DB_BASE_URL}/station/{pattern}",
                     headers=config.DB_HEADERS, timeout=10)
    r.raise_for_status()
    return [s.attrib for s in ET.fromstring(r.content).findall("station")]


def fetch_changes(eva: str) -> ET.Element:
    r = requests.get(f"{config.DB_BASE_URL}/fchg/{eva}",
                     headers=config.DB_HEADERS, timeout=10)
    r.raise_for_status()
    return ET.fromstring(r.content)


def _extract(stop, tl) -> Optional[dict]:
    for tag in ("ar", "dp"):
        event = stop.find(tag)
        if event is None:
            continue
        cause_code = None
        for m in event.findall("m"):
            if m.get("t") == "d" and m.get("c"):
                cause_code = int(m.get("c"))
        return {
            "delay_min": _delay_minutes(event.get("pt"), event.get("ct")),
            "cause_code": cause_code,
            "train_type": tl.get("c") if tl is not None else None,
            "train_number": tl.get("n") if tl is not None else None,
        }
    return None


def find_train(eva: str, train_number: str) -> Optional[dict]:
    """Belirli bir tren numarasini istasyonun canli akisinda arar."""
    for stop in fetch_changes(eva).findall("s"):
        tl = stop.find("tl")
        if tl is None or tl.get("n") != str(train_number):
            continue
        result = _extract(stop, tl)
        if result:
            return result
    return None


def first_delayed_train(eva: str) -> Optional[dict]:
    """Aktif gecikme mesaji (t='d') tasiyan ilk treni dondurur."""
    for stop in fetch_changes(eva).findall("s"):
        tl = stop.find("tl")
        for tag in ("ar", "dp"):
            event = stop.find(tag)
            if event is None or not event.get("pt") or not event.get("ct"):
                continue
            for m in event.findall("m"):
                if m.get("t") == "d" and m.get("c"):
                    return {
                        "delay_min": _delay_minutes(event.get("pt"), event.get("ct")),
                        "cause_code": int(m.get("c")),
                        "train_type": tl.get("c") if tl is not None else "ICE",
                        "train_number": tl.get("n") if tl is not None else None,
                    }
    return None
