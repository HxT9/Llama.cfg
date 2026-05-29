"""Custom GGUF header/metadata parser (stdlib only).

Reads only the header + metadata KV block of a .gguf file; it never touches the
tensor info table or tensor data, so it works in O(metadata) regardless of the
model size (8GB+ files are read in a few KB).

GGUF layout (little-endian):
    u32  magic == 'GGUF'
    u32  version (2 or 3)
    u64  tensor_count
    u64  metadata_kv_count
    metadata_kv_count * KV pairs:
        string key            (u64 length + bytes)
        u32    value_type
        value                 (per type below)

We stop reading after metadata_kv_count pairs.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path

from app.models import GgufMetadata

GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian as u32

# GGUF metadata value types
(
    GGUF_U8, GGUF_I8, GGUF_U16, GGUF_I16, GGUF_U32, GGUF_I32,
    GGUF_F32, GGUF_BOOL, GGUF_STRING, GGUF_ARRAY, GGUF_U64, GGUF_I64, GGUF_F64,
) = range(13)

_SCALAR_FMT = {
    GGUF_U8: ("<B", 1), GGUF_I8: ("<b", 1),
    GGUF_U16: ("<H", 2), GGUF_I16: ("<h", 2),
    GGUF_U32: ("<I", 4), GGUF_I32: ("<i", 4),
    GGUF_F32: ("<f", 4), GGUF_BOOL: ("<?", 1),
    GGUF_U64: ("<Q", 8), GGUF_I64: ("<q", 8), GGUF_F64: ("<d", 8),
}

_MAX_STRING = 64 * 1024 * 1024   # sanity cap; longer => treat as corrupt
_MAX_ARRAY = 16 * 1024 * 1024


class GgufError(Exception):
    pass


class _Cursor:
    """Sequential reader over an open binary file with struct helpers."""

    def __init__(self, fh):
        self.fh = fh

    def read(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise GgufError("unexpected end of file")
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def scalar(self, vtype: int):
        fmt, size = _SCALAR_FMT[vtype]
        return struct.unpack(fmt, self.read(size))[0]

    def string(self) -> str:
        n = self.u64()
        if n > _MAX_STRING:
            raise GgufError(f"string length {n} too large (corrupt file?)")
        return self.read(n).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype in _SCALAR_FMT:
            return self.scalar(vtype)
        if vtype == GGUF_STRING:
            return self.string()
        if vtype == GGUF_ARRAY:
            elem_type = self.u32()
            count = self.u64()
            if count > _MAX_ARRAY:
                raise GgufError(f"array count {count} too large (corrupt file?)")
            # We don't need full array contents for our metadata; consume them.
            out = []
            for _ in range(count):
                out.append(self.value(elem_type))
            return out
        raise GgufError(f"unknown value type {vtype}")


def _read_kv(path: Path) -> tuple[int, dict]:
    """Return (version, {key: value}) reading only the metadata block."""
    with open(path, "rb") as fh:
        cur = _Cursor(fh)
        magic = cur.u32()
        if magic != GGUF_MAGIC:
            raise GgufError(f"not a GGUF file (magic={magic:#x})")
        version = cur.u32()
        if version not in (2, 3):
            # keep going but flag; layout for v2/v3 is identical for our needs
            pass
        _tensor_count = cur.u64()
        kv_count = cur.u64()
        kv: dict = {}
        for _ in range(kv_count):
            key = cur.string()
            vtype = cur.u32()
            kv[key] = cur.value(vtype)
        return version, kv


def _as_int(v):
    """Coerce a metadata value to int. Some models store per-layer head counts
    as arrays; take the max (conservative for VRAM budgeting)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        nums = [x for x in v if isinstance(x, (int, float))]
        return int(max(nums)) if nums else None
    if isinstance(v, (int, float)):
        return int(v)
    return None


def read_gguf_metadata(path: str | os.PathLike) -> GgufMetadata:
    """Parse a GGUF file into GgufMetadata. Never raises on a bad file —
    returns a metadata object with `error` set instead."""
    p = Path(path)
    try:
        resolved = p.resolve()
        size = os.path.getsize(resolved)
    except OSError as e:
        return GgufMetadata(error=f"cannot stat file: {e}", file_size_bytes=0)

    try:
        version, kv = _read_kv(resolved)
    except (GgufError, OSError, struct.error) as e:
        return GgufMetadata(error=str(e), file_size_bytes=size)

    arch = kv.get("general.architecture")

    def a(suffix: str):
        if arch is None:
            return None
        return kv.get(f"{arch}.{suffix}")

    n_head = _as_int(a("attention.head_count"))
    n_head_kv = _as_int(a("attention.head_count_kv"))
    if n_head_kv is None:
        n_head_kv = n_head
    n_embd = _as_int(a("embedding_length"))
    head_dim = _as_int(a("attention.key_length"))
    if head_dim is None and n_embd and n_head:
        head_dim = int(n_embd // n_head)

    expert_count = _as_int(a("expert_count")) or 0
    expert_used = _as_int(a("expert_used_count")) or 0

    return GgufMetadata(
        architecture=arch,
        name=kv.get("general.name"),
        version=version,
        n_layers=_as_int(a("block_count")),
        context_length=_as_int(a("context_length")),
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=n_head_kv,
        head_dim=head_dim,
        expert_count=int(expert_count),
        expert_used_count=int(expert_used),
        file_size_bytes=size,
        is_moe=int(expert_count) > 1,
    )
