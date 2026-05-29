"""Paths, constants and default settings."""
from __future__ import annotations

from pathlib import Path

# --- Filesystem layout -------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
STATIC_DIR = APP_DIR / "static"

SETTINGS_PATH = DATA_DIR / "settings.json"
CONFIGS_PATH = DATA_DIR / "configs.json"
FLAGS_CACHE_PATH = DATA_DIR / "flags_cache.json"

# --- Defaults for first run --------------------------------------------------
DEFAULT_SCAN_ROOTS = [r"G:\Home\AI\HuggingFace\hub"]
DEFAULT_LLAMA_SERVER_EXE = (
    r"G:\Home\AI\Backend\llamaCPP\build\bin\Release\llama-server.exe"
)
DEFAULT_OUTPUT_INI = r"G:\Home\AI\llama-server-config.ini"

# --- Suggester tunables (overridable via Settings) ---------------------------
DEFAULT_HEADROOM_FRAC = 0.10          # fraction of VRAM reserved as headroom
DEFAULT_COMPUTE_RESERVE_MIB = 1024    # fixed VRAM reserve for compute buffers/OS

# Context sizes the suggester will snap to (ascending).
CONTEXT_STEPS = [4096, 8192, 16384, 32768, 65536, 131072, 262144]

# Approximate bytes-per-element for KV cache quantizations.
# (quant block formats carry small scale overhead, hence the > nominal values)
KV_BYTES_PER_ELEM = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.06,
    "q5_1": 0.75,
    "q5_0": 0.69,
    "q4_1": 0.63,
    "q4_0": 0.56,
    "iq4_nl": 0.56,
}


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
