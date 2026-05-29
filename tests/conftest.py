"""Shared fixtures. Local-machine paths are probed and tests that need them
are skipped if absent (so the suite still runs on other machines / CI)."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"

LLAMA_EXE = r"G:\Home\AI\Backend\llamaCPP\build\bin\Release\llama-server.exe"
HF_HUB = r"G:\Home\AI\HuggingFace\hub"


@pytest.fixture
def help_text() -> str:
    return (FIXTURES / "llama_server_help.txt").read_text(encoding="utf-8")


@pytest.fixture
def sample_ini() -> str:
    return (FIXTURES / "sample_config.ini").read_text(encoding="utf-8")


@pytest.fixture
def llama_exe() -> str:
    if not os.path.exists(LLAMA_EXE):
        pytest.skip("local llama-server.exe not present")
    return LLAMA_EXE


@pytest.fixture
def small_gguf() -> str:
    matches = glob.glob(os.path.join(HF_HUB, "**", "*E4B*Q8_0*.gguf"), recursive=True)
    matches = [m for m in matches if "mmproj" not in os.path.basename(m).lower()]
    if not matches:
        pytest.skip("no local small gguf present")
    return matches[0]


@pytest.fixture
def moe_gguf() -> str:
    matches = glob.glob(os.path.join(HF_HUB, "**", "*A3B*.gguf"), recursive=True)
    matches = [m for m in matches if "mmproj" not in os.path.basename(m).lower()]
    if not matches:
        pytest.skip("no local MoE gguf present")
    return matches[0]


@pytest.fixture
def isolated_data(tmp_path, monkeypatch):
    """Point the store/config at a temp data dir so tests don't touch real data."""
    from app import config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(cfg, "CONFIGS_PATH", tmp_path / "configs.json")
    monkeypatch.setattr(cfg, "FLAGS_CACHE_PATH", tmp_path / "flags_cache.json")
    return tmp_path
