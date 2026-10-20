"""Split the contracts markdown into heading-based chunks with metadata.

Every '#' heading starts a parent document (Schedule A, or one merchant
agreement). Every '##' heading under it becomes one chunk whose `parent_key`
is taken from the '#' heading, e.g.

    # Schedule A — Standard Rate Card ...      -> parent_key "Schedule A"
    ## A.1 Forward freight (per shipment)       -> section_id "A.1"

    # Merchant Service Agreement — Megh Linen (M01)  -> parent_key "M01"
    ## M01-§1 Services                               -> section_id "M01-§1"

The text between a '#' heading and its first '##' heading (the preamble:
parties, effective date, pickup city) is kept as its own chunk.
"""

import re
from pathlib import Path

MAX_CHUNK_CHARS = 1500

MERCHANT_ID_RE = re.compile(r"\((M\d+)\)\s*$")
MERCHANT_NAME_RE = re.compile(r"—\s*(.+?)\s*\(M\d+\)\s*$")
EFFECTIVE_DATE_RE = re.compile(r"effective\s+(\d{1,2}\s+\w+\s+\d{4})", re.IGNORECASE)
PICKUP_CITY_RE = re.compile(r"warehouse in\s+([A-Z][A-Za-z ]+?)\.", re.IGNORECASE)

# References to other sections, used to pull related chunks in at query time.
SCHEDULE_REF_RE = re.compile(r"section\s+(A\.\d+)")
CLAUSE_REF_RE = re.compile(r"\b(M\d+-§\d+)(?:\.\d+)?")


def parent_key_from_heading(heading):
    """'Merchant Service Agreement — Megh Linen (M01)' -> 'M01';
    'Schedule A — Standard Rate Card ...' -> 'Schedule A'."""
    match = MERCHANT_ID_RE.search(heading)
    if match:
        return match.group(1)
    return heading.split("—")[0].strip()


def split_section_heading(heading):
    """Return (section_id, section_title) for a '##' heading.

    'A.1 Forward freight (per shipment)'          -> ('A.1', 'Forward freight (per shipment)')
    'M01-§1 Services'                              -> ('M01-§1', 'Services')
    'M02 Annexure 1 — Festive launch COD fee waiver' -> ('M02 Annexure 1', 'Festive launch COD fee waiver')
    """
    if "—" in heading:
        section_id, title = heading.split("—", 1)
        return section_id.strip(), title.strip()
    section_id, _, title = heading.partition(" ")
    return section_id.strip(), title.strip()


def find_references(text, own_parent, own_section):
    refs = set(SCHEDULE_REF_RE.findall(text))
    refs.update(CLAUSE_REF_RE.findall(text))
    if "Schedule A" in text and own_parent != "Schedule A":
        refs.add("Schedule A")
    refs.discard(own_section)
    return sorted(refs)


def split_long_body(body):
    """Split an oversized section on paragraph boundaries."""
    if len(body) <= MAX_CHUNK_CHARS:
        return [body]
    parts, current = [], ""
    for paragraph in body.split("\n\n"):
        if current and len(current) + len(paragraph) > MAX_CHUNK_CHARS:
            parts.append(current.strip())
            current = ""
        current += paragraph + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return parts


def parse_markdown(markdown):
    """Group the document into parents, each holding a preamble and sections."""
    parents = []
    parent = None
    section = None

    for line in markdown.splitlines():
        if line.startswith("# "):
            parent = {"heading": line[2:].strip(), "preamble": [], "sections": []}
            parents.append(parent)
            section = None
        elif line.startswith("## "):
            if parent is None:
                raise ValueError(f"'##' heading before any '#' heading: {line}")
            section = {"heading": line[3:].strip(), "body": []}
            parent["sections"].append(section)
        elif line.strip() == "---":
            continue
        elif section is not None:
            section["body"].append(line)
        elif parent is not None:
            parent["preamble"].append(line)

    return parents


def build_chunks(markdown_path):
    markdown = Path(markdown_path).read_text(encoding="utf-8")
    source = Path(markdown_path).name
    chunks = []

    for parent in parse_markdown(markdown):
        heading = parent["heading"]
        parent_key = parent_key_from_heading(heading)
        preamble = "\n".join(parent["preamble"]).strip()

        is_contract = parent_key != "Schedule A"
        name_match = MERCHANT_NAME_RE.search(heading)
        date_match = EFFECTIVE_DATE_RE.search(preamble)
        city_match = PICKUP_CITY_RE.search(preamble)

        parent_meta = {
            "source": source,
            "parent_key": parent_key,
            "parent_title": heading,
            "doc_type": "merchant_agreement" if is_contract else "rate_card",
            "merchant_id": parent_key if is_contract else None,
            "merchant_name": name_match.group(1) if name_match else None,
            "effective_date": date_match.group(1) if date_match else None,
            "pickup_city": city_match.group(1).strip() if city_match else None,
        }

        sections = []
        if preamble:
            sections.append((f"{parent_key} Preamble", "Preamble", preamble, "preamble"))
        for section in parent["sections"]:
            section_id, title = split_section_heading(section["heading"])
            body = "\n".join(section["body"]).strip()
            kind = "annexure" if "Annexure" in section_id else "section"
            sections.append((section_id, title, body, kind))

        for section_id, title, body, kind in sections:
            parts = split_long_body(body)
            for part_index, part in enumerate(parts):
                header = f"# {heading}\n## {section_id} {title}".rstrip()
                chunk_id = section_id if len(parts) == 1 else f"{section_id}#{part_index + 1}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "section_id": section_id,
                    "section_title": title,
                    "section_kind": kind,
                    "part": part_index + 1,
                    "references": find_references(part, parent_key, section_id),
                    **parent_meta,
                    "text": f"{header}\n\n{part}",
                })

    ids = [chunk["chunk_id"] for chunk in chunks]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate chunk ids: {sorted(duplicates)}")
    return chunks
