"""JSON input, output and streaming SHA256 checks."""

import hashlib
import json


def read(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def public_entries(path, is_mapping=False):
    """Stream one item at a time from the audited, two-space-indented archive."""
    expected_start, expected_end = ("{", "}") if is_mapping else ("[", "]")
    with path.open(encoding="utf-8") as stream:
        if stream.readline().strip() != expected_start:
            raise ValueError(f"Unexpected archive layout in {path}")
        item = []
        for line in stream:
            if line.strip() == expected_end and not line.startswith(" "):
                if item or stream.read().strip():
                    raise ValueError(f"Unexpected trailing data in {path}")
                return
            item.append(line)
            if line.rstrip() not in ("  }", "  },"):
                continue
            value = "".join(item).rstrip().removesuffix(",")
            item.clear()
            if is_mapping:
                yield next(iter(json.loads("{" + value + "}").items()))
            else:
                yield json.loads(value)
    raise ValueError(f"Unexpected end of {path}")
