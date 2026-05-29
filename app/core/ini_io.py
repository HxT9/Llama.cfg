"""Custom INI reader/writer for the llama-server models-preset file.

We do NOT use configparser for writing because the values are Windows paths
containing ':' and '\\' (and possibly '%'), which trip configparser's delimiter
and interpolation handling. Our reader splits on the first '=' only and never
treats ':' as a delimiter.

Format::

    [section-name]
    model = G:\\path\\to\\model.gguf
    c = 16384
    no-mmap = true

Booleans are written as ``key = true`` (false-valued flags are simply omitted).
Section order and key order are preserved on round-trip.
"""
from __future__ import annotations

import os
import tempfile
from collections import OrderedDict

# section -> OrderedDict[key, value]
IniData = "OrderedDict[str, OrderedDict[str, str]]"


def parse_ini(text: str) -> "OrderedDict[str, OrderedDict[str, str]]":
    sections: OrderedDict[str, OrderedDict[str, str]] = OrderedDict()
    current: OrderedDict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in ";#":
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            current = sections.setdefault(name, OrderedDict())
            continue
        if current is None:
            continue  # key before any section header; ignore
        if "=" not in line:
            # valueless key -> treat as boolean true
            current[line] = "true"
            continue
        key, _, value = line.partition("=")
        current[key.strip()] = value.strip()
    return sections


def _section_lines(name: str, kv: "OrderedDict[str, str]") -> list[str]:
    lines = [f"[{name}]"]
    # model first if present, then the rest in insertion order
    ordered_keys = list(kv.keys())
    if "model" in kv:
        ordered_keys = ["model"] + [k for k in ordered_keys if k != "model"]
    for k in ordered_keys:
        lines.append(f"{k} = {kv[k]}")
    return lines


def render_ini(sections: "OrderedDict[str, OrderedDict[str, str]]") -> str:
    blocks = []
    for name, kv in sections.items():
        blocks.append("\n".join(_section_lines(name, kv)))
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def write_ini_atomic(path: str, text: str) -> None:
    """Write atomically: temp file in the same dir, then os.replace."""
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
