"""Reading and writing the files in data/ and outputs/. No logic.

The dashboard carries its own copy of this module (CONTRACTS.md rule 4):
the FORMAT is shared, the code is not, so either component can be lifted
out whole. check_boundaries.py compares the copies -- deliberate
divergence is allowed, silent drift is not.
"""
import hashlib
import json
import os
import time
from pathlib import Path

#: top-level keys of a run record written one item per line, so that a
#: grep for a case id or an entity id returns the whole thing
LINE_PER_ITEM = ("relations", "entities", "cases")


def load_triples(path):
    """A TSV of head/relation/tail strings, order preserved.

    Blank lines are skipped; any other line must hold exactly three
    tab-separated fields, or this raises ValueError naming the line.
    """
    triples = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{path}:{number}: expected 3 tab-separated "
                                 f"fields, found {len(parts)}")
            triples.append(tuple(parts))
    return triples


def write_tsv(path, rows):
    """Rows as tab-separated lines: UTF-8, LF, newline-terminated."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write("\t".join(str(field) for field in row) + "\n")


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload, compact=False):
    """JSON, atomically. Small files (manifests) indented, a view model
    compact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    else:
        text = json.dumps(payload, ensure_ascii=False, indent=1)
    _replace_atomically(path, text + "\n")


def dump_record(record):
    """A run record as text: header fields indented, and every item of the
    LINE_PER_ITEM keys compact on a line of its own. Valid JSON --
    json.loads reads it back unchanged."""
    def compact(value):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    lines = ["{"]
    items = list(record.items())
    for position, (key, value) in enumerate(items):
        comma = "," if position < len(items) - 1 else ""
        name = json.dumps(key)
        if key in LINE_PER_ITEM and isinstance(value, list) and value:
            lines.append(f" {name}: [")
            lines.append(",\n".join(compact(item) for item in value))
            lines.append(f" ]{comma}")
        elif key in LINE_PER_ITEM and isinstance(value, dict) and value:
            lines.append(f" {name}: {{")
            lines.append(",\n".join(f"{json.dumps(k)}:{compact(v)}"
                                    for k, v in value.items()))
            lines.append(f" }}{comma}")
        else:
            body = json.dumps(value, ensure_ascii=False, indent=1)
            lines.append(f" {name}: " + body.replace("\n", "\n ") + comma)
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_record(path, record):
    """Write a run record atomically: a reader never sees half a file, and
    a crash mid-write leaves the previous version in place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_atomically(path, dump_record(record))


def _replace_atomically(path, text):
    """Temp file beside the target, then one rename over it.

    On Windows a rename onto a file another process has open fails with
    PermissionError (the dashboard exporter reading the record at that
    moment), so the rename is retried for a few seconds before giving up.
    """
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    for attempt in range(12):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            time.sleep(0.25 * (attempt + 1))
    os.replace(temporary, path)


def sha256_of(path):
    """A file's fingerprint -- what binds a record to the files it names."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
