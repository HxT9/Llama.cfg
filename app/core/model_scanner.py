"""Recursive *.gguf scanner with symlink resolution and dedupe by blob.

HuggingFace stores the gguf under snapshots/<sha>/name.gguf as a symlink to
blobs/<sha256>. We display the snapshot path (what the existing INI uses) but
dedupe by the resolved blob so the same model isn't listed twice.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from app.core.gguf_reader import read_gguf_metadata
from app.models import ModelInfo, ScanResult

# multipart: name-00001-of-00003.gguf
_MULTIPART_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)
_MMPROJ_RE = re.compile(r"^mmproj.*\.gguf$", re.I)


def _resolve(p: Path) -> Path:
    try:
        return p.resolve()
    except OSError:
        return p


def scan(roots: list[str], with_metadata: bool = True) -> ScanResult:
    warnings: list[str] = []
    by_blob: dict[str, ModelInfo] = {}
    # group multipart files: key = (dir, stem) -> list of (idx, path)
    multipart: dict[tuple[str, str], list[tuple[int, Path]]] = {}

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            warnings.append(f"scan root does not exist: {root}")
            continue
        try:
            gguf_files = list(root_path.rglob("*.gguf"))
        except OSError as e:
            warnings.append(f"error walking {root}: {e}")
            continue

        for f in gguf_files:
            name = f.name
            if _MMPROJ_RE.match(name):
                continue  # projectors are attached to their sibling model, not listed
            mp = _MULTIPART_RE.match(name)
            if mp:
                key = (str(f.parent), mp.group("stem"))
                multipart.setdefault(key, []).append((int(mp.group("idx")), f))
                continue
            _add_model(f, by_blob, warnings, single_part=True)

    # collapse multipart groups into one entry (use part 1 as the model path)
    for (dirname, stem), parts in multipart.items():
        parts.sort()
        first = parts[0][1]
        _add_model(
            first,
            by_blob,
            warnings,
            single_part=False,
            part_paths=[str(p) for _, p in parts],
        )

    models = list(by_blob.values())
    if with_metadata:
        for m in models:
            m.metadata = read_gguf_metadata(m.blob_path)

    models.sort(key=lambda m: m.name.lower())
    return ScanResult(models=models, warnings=warnings)


def _add_model(
    f: Path,
    by_blob: dict[str, ModelInfo],
    warnings: list[str],
    single_part: bool,
    part_paths: list[str] | None = None,
) -> None:
    resolved = _resolve(f)
    blob_key = str(resolved)
    try:
        if part_paths:
            size = sum(_safe_size(Path(p)) for p in part_paths)
        else:
            size = _safe_size(resolved)
    except OSError as e:
        warnings.append(f"cannot stat {f}: {e}")
        return

    if blob_key in by_blob:
        return  # dedupe

    mmproj = _find_mmproj(f.parent)
    by_blob[blob_key] = ModelInfo(
        display_path=str(f),
        blob_path=blob_key,
        size_bytes=size,
        name=f.name,
        dir=str(f.parent),
        mmproj_candidates=mmproj,
        parts=part_paths or [],
    )


def _safe_size(p: Path) -> int:
    return os.path.getsize(_resolve(p))


def _find_mmproj(directory: Path) -> list[str]:
    out: list[str] = []
    try:
        for entry in directory.iterdir():
            if _MMPROJ_RE.match(entry.name):
                out.append(str(entry))
    except OSError:
        pass
    return sorted(out)
