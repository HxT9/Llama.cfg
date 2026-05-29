"""Scan and parse llama-server CLI flags from `--help`.

The parsed result is cached to data/flags_cache.json keyed by the exe's
(path, mtime, size) so we only shell out when the binary changes.

Help format (observed, llama.cpp build b7xxx)::

    ----- common params -----                         <- section banner

    -t,    --threads N                      number of CPU threads ...
                                            (env: LLAMA_ARG_THREADS)
    -ctk,  --cache-type-k TYPE              KV cache data type for K
                                            allowed values: f32, f16, ...
                                            (default: f16)
    --mmap, --no-mmap                       whether to memory-map model ...
    -fa,   --flash-attn [on|off|auto]       set Flash Attention use ...
    --rope-scaling {none,linear,yarn}       RoPE frequency scaling ...
    --spec-type none,draft-simple,ngram...  (bare comma-list enum)

Signature vs description split: the description starts at the first run of 2+
spaces that is *not* immediately preceded by a comma (commas+padding separate
aliases, so those gaps are skipped).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from app.config import FLAGS_CACHE_PATH, ensure_data_dir
from app.models import FlagSpec

_BANNER_RE = re.compile(r"^-{3,}\s*(.+?)\s*-{3,}\s*$")
# 2+ spaces preceded by a non-comma, non-space char => end of signature
_DESC_SPLIT_RE = re.compile(r"(?<=[^,\s])\s{2,}")
_DEFAULT_RE = re.compile(r"\(default:\s*(.*?)\)\s*$")
_ENV_RE = re.compile(r"\(env:\s*([A-Z0-9_]+)\)")
_ALLOWED_RE = re.compile(r"allowed values:\s*(.+)$", re.IGNORECASE)


def _infer_type(placeholder: str | None) -> tuple[str, list[str]]:
    """Return (value_type, enum_values) for a placeholder token string."""
    if placeholder is None:
        return "bool", []
    p = placeholder.strip()
    # [on|off|auto] / [on|off]
    m = re.fullmatch(r"\[(.+)\]", p)
    if m and "|" in m.group(1):
        return "tristate", [v.strip() for v in m.group(1).split("|")]
    # {a,b,c}
    m = re.fullmatch(r"\{(.+)\}", p)
    if m:
        return "enum", [v.strip() for v in m.group(1).split(",")]
    # <0|1>  -> enum
    m = re.fullmatch(r"<([^<>]*\|[^<>]*)>", p)
    if m:
        return "enum", [v.strip() for v in m.group(1).split("|")]
    # <0...100> or N or any single placeholder token in angle brackets
    if p.startswith("<") and p.endswith(">"):
        return "string", []
    # bare comma-list enum (e.g. --spec-type none,draft-simple,...). Exclude
    # bracketed device lists (<dev1,dev2,..>) and numeric-margin lists
    # (MiB0,MiB1,...) which carry an ellipsis.
    if "," in p and "..." not in p and "<" not in p and "[" not in p:
        return "enum", [v.strip() for v in p.split(",") if v.strip()]
    # path-ish placeholders
    if p.upper() in {"FNAME", "FILE", "PATH", "URL"}:
        return "path", []
    if p == "N":
        return "int", []
    if p in {"P"}:
        return "number", []
    return "string", []


def _parse_signature(sig: str) -> tuple[list[str], str | None]:
    """Split a signature into (aliases, placeholder)."""
    tokens = sig.split()
    aliases: list[str] = []
    i = 0
    while i < len(tokens) and tokens[i].lstrip(",").startswith("-"):
        aliases.append(tokens[i].rstrip(","))
        i += 1
    placeholder = " ".join(tokens[i:]).strip() or None
    return aliases, placeholder


def _canonical(aliases: list[str]) -> str:
    longs = [a for a in aliases if a.startswith("--")]
    base = longs[0] if longs else (aliases[0] if aliases else "")
    return base.lstrip("-")


def parse_help_text(text: str) -> list[FlagSpec]:
    flags: list[FlagSpec] = []
    group: str | None = None
    cur: dict | None = None

    def finalize(spec: dict | None):
        if spec is None:
            return
        value_type, enum_from_ph = _infer_type(spec["placeholder"])
        aliases = spec["aliases"]
        # combined enable/disable toggles => boolean, no value
        if any(a.startswith("--no-") for a in aliases) and spec["placeholder"] is None:
            value_type = "bool"
        enum_values = spec["enum_allowed"] or enum_from_ph
        # an "allowed values:" continuation makes a TYPE/string placeholder an enum
        if spec["enum_allowed"] and value_type in ("string", "path"):
            value_type = "enum"
        takes_value = value_type not in ("bool",)
        flags.append(
            FlagSpec(
                canonical_key=_canonical(aliases),
                aliases=aliases,
                value_type=value_type,
                value_placeholder=spec["placeholder"],
                enum_values=enum_values,
                default=spec["default"],
                env=spec["env"],
                group=spec["group"],
                description=" ".join(spec["desc"]).strip(),
                takes_value=takes_value,
            )
        )

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        banner = _BANNER_RE.match(line.strip())
        if banner and not line.startswith("- "):
            finalize(cur)
            cur = None
            group = banner.group(1)
            continue
        is_header = line[:1] == "-" and not line.startswith("- ")
        if is_header:
            finalize(cur)
            m = _DESC_SPLIT_RE.search(line)
            if m:
                sig, desc = line[: m.start()], line[m.end():]
            else:
                sig, desc = line, ""
            aliases, placeholder = _parse_signature(sig)
            cur = {
                "aliases": aliases,
                "placeholder": placeholder,
                "group": group,
                "desc": [],
                "default": None,
                "env": None,
                "enum_allowed": [],
            }
            _absorb_desc_line(cur, desc)
        elif cur is not None:
            _absorb_desc_line(cur, line.strip())
    finalize(cur)
    return flags


def _absorb_desc_line(spec: dict, text: str) -> None:
    if not text:
        return
    env = _ENV_RE.search(text)
    if env:
        spec["env"] = env.group(1)
        text = _ENV_RE.sub("", text).strip()
    dft = _DEFAULT_RE.search(text)
    if dft:
        spec["default"] = dft.group(1).strip()
    allowed = _ALLOWED_RE.search(text)
    if allowed:
        spec["enum_allowed"] = [v.strip() for v in allowed.group(1).split(",") if v.strip()]
        return
    if text:
        spec["desc"].append(text)


# --- exe invocation + caching ------------------------------------------------
def _exe_key(exe_path: str) -> dict:
    st = os.stat(exe_path)
    return {"path": os.path.abspath(exe_path), "mtime": st.st_mtime, "size": st.st_size}


def _load_cache(key: dict) -> list[FlagSpec] | None:
    if not FLAGS_CACHE_PATH.exists():
        return None
    try:
        data = json.loads(FLAGS_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("key") != key:
        return None
    return [FlagSpec(**f) for f in data.get("flags", [])]


def _save_cache(key: dict, flags: list[FlagSpec]) -> None:
    ensure_data_dir()
    payload = {"key": key, "flags": [f.model_dump() for f in flags]}
    FLAGS_CACHE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_help(exe_path: str, timeout: int = 30) -> str:
    proc = subprocess.run(
        [exe_path, "--help"],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    # llama-server prints help to stdout; fall back to stderr just in case
    return proc.stdout if proc.stdout.strip() else proc.stderr


def get_flags(exe_path: str, force_refresh: bool = False) -> list[FlagSpec]:
    if not exe_path or not Path(exe_path).exists():
        raise FileNotFoundError(f"llama-server exe not found: {exe_path}")
    key = _exe_key(exe_path)
    if not force_refresh:
        cached = _load_cache(key)
        if cached is not None:
            return cached
    flags = parse_help_text(run_help(exe_path))
    _save_cache(key, flags)
    return flags
