"""Pydantic data models shared across the backend + API layer."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# --- GGUF / models -----------------------------------------------------------
class GgufMetadata(BaseModel):
    architecture: Optional[str] = None
    name: Optional[str] = None
    version: Optional[int] = None
    n_layers: Optional[int] = None          # <arch>.block_count
    context_length: Optional[int] = None    # <arch>.context_length
    n_embd: Optional[int] = None            # <arch>.embedding_length
    n_head: Optional[int] = None            # <arch>.attention.head_count
    n_head_kv: Optional[int] = None         # <arch>.attention.head_count_kv
    head_dim: Optional[int] = None          # key_length (fallback n_embd/n_head)
    expert_count: int = 0                   # <arch>.expert_count
    expert_used_count: int = 0              # <arch>.expert_used_count
    expert_fraction: Optional[float] = None  # fraction of params in MoE experts
    file_size_bytes: int = 0
    is_moe: bool = False
    error: Optional[str] = None


class ModelInfo(BaseModel):
    display_path: str                       # snapshot symlink path (written to INI)
    blob_path: str                          # resolved real file
    size_bytes: int = 0
    name: str = ""
    dir: str = ""
    mmproj_candidates: list[str] = Field(default_factory=list)
    parts: list[str] = Field(default_factory=list)   # multipart members (if any)
    metadata: Optional[GgufMetadata] = None


class ScanResult(BaseModel):
    models: list[ModelInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# --- Flags -------------------------------------------------------------------
class FlagSpec(BaseModel):
    canonical_key: str                      # long flag, dashes stripped (ini key)
    aliases: list[str] = Field(default_factory=list)   # raw alias tokens incl. dashes
    value_type: str = "string"              # bool|int|number|enum|tristate|path|string
    value_placeholder: Optional[str] = None
    enum_values: list[str] = Field(default_factory=list)
    default: Optional[str] = None
    env: Optional[str] = None
    group: Optional[str] = None
    description: str = ""
    takes_value: bool = True


# --- Hardware ----------------------------------------------------------------
class GpuInfo(BaseModel):
    index: int
    name: Optional[str] = None
    total_mib: int = 0
    free_mib: int = 0


class HardwareInfo(BaseModel):
    gpus: list[GpuInfo] = Field(default_factory=list)
    vram_source: str = "nvidia-smi"         # nvidia-smi | pynvml | manual
    ram_total_mib: int = 0
    ram_available_mib: int = 0


# --- Suggestion --------------------------------------------------------------
class SuggestRequest(BaseModel):
    model_path: str
    mmproj_path: Optional[str] = None       # counted against VRAM if offloaded
    context: Optional[int] = None           # desired ctx; None = auto-pick max
    ctk: str = "f16"
    ctv: str = "f16"
    vram_budget_mode: str = "free"          # free | total | manual
    manual_vram_mib: Optional[int] = None
    headroom_frac: Optional[float] = None
    compute_reserve_mib: Optional[int] = None


class Suggestion(BaseModel):
    explicit: dict[str, Any] = Field(default_factory=dict)   # {ngl, c, ctk, ctv}
    fit: dict[str, Any] = Field(default_factory=dict)        # {fit, fitc, fitt}
    breakdown: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


# --- Config store ------------------------------------------------------------
class ConfigEntry(BaseModel):
    id: str
    name: str                               # = INI section, unique on export
    model_display_path: str = ""
    flags: dict[str, str] = Field(default_factory=dict)
    mmproj: Optional[str] = None
    suggestion_snapshot: Optional[dict[str, Any]] = None
    notes: str = ""
    created: Optional[str] = None
    updated: Optional[str] = None


class ConfigCreate(BaseModel):
    name: str
    model_display_path: str = ""
    flags: dict[str, str] = Field(default_factory=dict)
    mmproj: Optional[str] = None
    notes: str = ""


class ConfigUpdate(BaseModel):
    name: Optional[str] = None
    model_display_path: Optional[str] = None
    flags: Optional[dict[str, str]] = None
    mmproj: Optional[str] = None
    suggestion_snapshot: Optional[dict[str, Any]] = None
    notes: Optional[str] = None


class Settings(BaseModel):
    scan_roots: list[str] = Field(default_factory=list)
    llama_server_exe: str = ""
    output_ini_path: str = ""
    manual_vram_mib: Optional[int] = None
    headroom_frac: float = 0.10
    compute_reserve_mib: int = 1024
    # remembered values for editable-combobox flags, keyed by flag key
    value_presets: dict[str, list[str]] = Field(default_factory=dict)
    # last-used suggestion panel inputs (context, ctk, ctv, vram mode, manual vram)
    suggest_defaults: dict[str, Any] = Field(default_factory=dict)


class ExportRequest(BaseModel):
    output_path: Optional[str] = None       # override settings.output_ini_path


class ImportRequest(BaseModel):
    text: Optional[str] = None              # raw INI; if None, read output_ini_path
    replace: bool = True                    # replace store vs merge
