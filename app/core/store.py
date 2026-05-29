"""JSON working store: config entries + settings, and INI import/export.

The JSON store (data/configs.json) is the source of truth while editing. The
INI is an export/import artifact produced from / seeded into the store.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from collections import OrderedDict
from datetime import datetime, timezone

from app import config as cfg
from app.core.ini_io import parse_ini, render_ini, write_ini_atomic
from app.models import ConfigEntry, Settings

# keys handled by dedicated entry fields, not free-form flags
_RESERVED_KEYS = {"model", "mmproj"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path, data) -> None:
    cfg.ensure_data_dir()
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# --- Settings ----------------------------------------------------------------
def default_settings() -> Settings:
    return Settings(
        scan_roots=list(cfg.DEFAULT_SCAN_ROOTS),
        llama_server_exe=cfg.DEFAULT_LLAMA_SERVER_EXE,
        output_ini_path=cfg.DEFAULT_OUTPUT_INI,
        manual_vram_mib=None,
        headroom_frac=cfg.DEFAULT_HEADROOM_FRAC,
        compute_reserve_mib=cfg.DEFAULT_COMPUTE_RESERVE_MIB,
    )


def load_settings() -> Settings:
    if not cfg.SETTINGS_PATH.exists():
        s = default_settings()
        save_settings(s)
        return s
    try:
        data = json.loads(cfg.SETTINGS_PATH.read_text(encoding="utf-8"))
        return Settings(**data)
    except (OSError, json.JSONDecodeError, ValueError):
        return default_settings()


def save_settings(s: Settings) -> Settings:
    _atomic_write_json(cfg.SETTINGS_PATH, s.model_dump())
    return s


# --- Config entries ----------------------------------------------------------
def load_entries() -> list[ConfigEntry]:
    if not cfg.CONFIGS_PATH.exists():
        return []
    try:
        data = json.loads(cfg.CONFIGS_PATH.read_text(encoding="utf-8"))
        return [ConfigEntry(**e) for e in data]
    except (OSError, json.JSONDecodeError, ValueError):
        return []


def save_entries(entries: list[ConfigEntry]) -> None:
    _atomic_write_json(cfg.CONFIGS_PATH, [e.model_dump() for e in entries])


def create_entry(
    name: str,
    model_display_path: str = "",
    flags: dict | None = None,
    mmproj: str | None = None,
    notes: str = "",
) -> ConfigEntry:
    entries = load_entries()
    entry = ConfigEntry(
        id=uuid.uuid4().hex,
        name=name,
        model_display_path=model_display_path,
        flags=flags or {},
        mmproj=mmproj,
        notes=notes,
        created=_now(),
        updated=_now(),
    )
    entries.append(entry)
    save_entries(entries)
    return entry


def get_entry(entry_id: str) -> ConfigEntry | None:
    return next((e for e in load_entries() if e.id == entry_id), None)


def update_entry(entry_id: str, patch: dict) -> ConfigEntry | None:
    entries = load_entries()
    for i, e in enumerate(entries):
        if e.id == entry_id:
            data = e.model_dump()
            for k, v in patch.items():
                if v is not None:
                    data[k] = v
            data["updated"] = _now()
            entries[i] = ConfigEntry(**data)
            save_entries(entries)
            return entries[i]
    return None


def delete_entry(entry_id: str) -> bool:
    entries = load_entries()
    new = [e for e in entries if e.id != entry_id]
    if len(new) == len(entries):
        return False
    save_entries(new)
    return True


# --- INI mapping -------------------------------------------------------------
def entries_to_ini_text(entries: list[ConfigEntry]) -> str:
    """Validate and render entries to an INI string."""
    seen: set[str] = set()
    sections: "OrderedDict[str, OrderedDict[str, str]]" = OrderedDict()
    for e in entries:
        if not e.name:
            raise ValueError("a config has an empty section name")
        if e.name in seen:
            raise ValueError(f"duplicate section name: {e.name}")
        seen.add(e.name)
        if not e.model_display_path and "model" not in e.flags:
            raise ValueError(f"config '{e.name}' is missing required 'model' path")
        kv: "OrderedDict[str, str]" = OrderedDict()
        if e.model_display_path:
            kv["model"] = e.model_display_path
        for k, v in e.flags.items():
            if k in _RESERVED_KEYS and k == "model" and e.model_display_path:
                continue
            kv[k] = v
        if e.mmproj:
            kv["mmproj"] = e.mmproj
        sections[e.name] = kv
    return render_ini(sections)


def export_store_to_ini(path: str) -> str:
    text = entries_to_ini_text(load_entries())
    write_ini_atomic(path, text)
    return text


def ini_text_to_entries(text: str) -> list[ConfigEntry]:
    sections = parse_ini(text)
    entries: list[ConfigEntry] = []
    for name, kv in sections.items():
        flags = {k: v for k, v in kv.items() if k not in _RESERVED_KEYS}
        entries.append(
            ConfigEntry(
                id=uuid.uuid4().hex,
                name=name,
                model_display_path=kv.get("model", ""),
                mmproj=kv.get("mmproj"),
                flags=flags,
                created=_now(),
                updated=_now(),
            )
        )
    return entries


def import_ini_to_store(text: str, replace: bool = True) -> list[ConfigEntry]:
    new_entries = ini_text_to_entries(text)
    if replace:
        save_entries(new_entries)
        return new_entries
    existing = load_entries()
    by_name = {e.name: e for e in existing}
    for e in new_entries:
        by_name[e.name] = e  # imported wins on name collision
    merged = list(by_name.values())
    save_entries(merged)
    return merged
