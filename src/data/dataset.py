"""Dataset scanner for DCASE Task 2 machine sound data.

Inspects file paths to infer machine, section, domain, split, and label.
No external metadata files needed. No anomaly labels used for selection.
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

MACHINE_PATTERNS = [
    (r"\bgearbox\b", "gearbox"),
    (r"\bbearing\b", "bearing"),
    (r"\bfan\b", "fan"),
    (r"\bslider\b", "slider"),
    (r"\bvalve\b", "valve"),
    (r"\btoycar\b", "toycar"),
    (r"\btoytrain\b", "toytrain"),
    (r"\btoyswitch\b", "toyswitch"),
    (r"\bautotrash\b", "autotrash"),
    (r"\bbandsealer\b", "bandsealer"),
    (r"\bcoffeegrinder\b", "coffeegrinder"),
    (r"\bhomecamera\b", "homecamera"),
    (r"\bpolisher\b", "polisher"),
    (r"\bscrewfeeder\b", "screwfeeder"),
    (r"\btoypet\b", "toypet"),
    (r"\btoyrccar\b", "toyrccar"),
]

SECTION_RE = re.compile(r"section[_]?(\d+)", re.IGNORECASE)


@dataclass
class AudioItem:
    path: Path
    machine: str
    section: str
    domain: str       # "source" or "target"
    split: str         # "train" or "test"
    label: Optional[int]  # 0 = normal, 1 = anomaly, None = unlabeled/unknown


def _infer_machine(pparts: List[str]) -> Optional[str]:
    joined = "/".join(p.lower() for p in pparts)
    for pattern, name in MACHINE_PATTERNS:
        if re.search(pattern, joined):
            return name
    return None


def _infer_section(pparts: List[str]) -> str:
    joined = "/".join(pparts)
    m = SECTION_RE.search(joined)
    if m:
        return f"{int(m.group(1)):02d}"
    return "00"


def _infer_domain(name: str) -> str:
    n = name.lower()
    if "source" in n:
        return "source"
    if "target" in n:
        return "target"
    return "source"  # default


def _infer_split(pparts: List[str]) -> str:
    lowered = [p.lower() for p in pparts]
    joined = "/".join(lowered)
    if "/train/" in joined or "train" in lowered:
        return "train"
    if "/test/" in joined or "test" in lowered:
        return "test"
    return "test"


def _infer_label(name: str) -> Optional[int]:
    n = name.lower()
    if "normal" in n:
        return 0
    if "anomaly" in n:
        return 1
    return None


def scan_dataset(root: Path) -> List[AudioItem]:
    """Scan a dataset root and return all audio items with inferred metadata."""
    items = []
    for wav_path in sorted(root.rglob("*.wav")):
        pparts = wav_path.parts
        fname = wav_path.name

        machine = _infer_machine(list(pparts))
        if machine is None:
            continue

        items.append(AudioItem(
            path=wav_path,
            machine=machine,
            section=_infer_section(list(pparts)),
            domain=_infer_domain(fname),
            split=_infer_split(list(pparts)),
            label=_infer_label(fname),
        ))
    return items


def filter_items(items: List[AudioItem], machine: str, section: Optional[str] = None) -> List[AudioItem]:
    """Filter items by machine and optional section."""
    result = [x for x in items if x.machine == machine]
    if section and section.strip():
        result = [x for x in result if x.section == section]
    return result


def split_train_test(items: List[AudioItem], use_target_train: bool = False) -> tuple:
    """Split items into train (normal only) and test (all labeled).

    Training: only normal files. By default only source-domain normal.
    Testing: all files with non-None label (normal + anomaly, source + target).
    """
    train_domains = {"source"}
    if use_target_train:
        train_domains.add("target")

    train = []
    for x in items:
        if x.split != "train" or x.domain not in train_domains:
            continue
        # DCASE train files are normal-only; accept missing label for train only.
        if x.label in (0, None):
            train.append(x)
    test = [
        x for x in items
        if x.split == "test" and x.label is not None and x.domain in ("source", "target")
    ]
    return train, test
