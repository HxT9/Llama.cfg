"""Detect available VRAM (NVIDIA) and system RAM.

VRAM detection tries pynvml first (if installed), then falls back to parsing
`nvidia-smi`. If neither works, the GPU list is empty and vram_source="manual"
so the UI can prompt for a manual VRAM figure.
"""
from __future__ import annotations

import subprocess

from app.models import GpuInfo, HardwareInfo


def _ram_mib() -> tuple[int, int]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return vm.total // (1024 * 1024), vm.available // (1024 * 1024)
    except Exception:
        return 0, 0


def _gpus_pynvml() -> list[GpuInfo] | None:
    try:
        import pynvml  # type: ignore
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        out: list[GpuInfo] = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            out.append(
                GpuInfo(
                    index=i,
                    name=name,
                    total_mib=mem.total // (1024 * 1024),
                    free_mib=mem.free // (1024 * 1024),
                )
            )
        pynvml.nvmlShutdown()
        return out
    except Exception:
        return None


def _gpus_nvidia_smi() -> list[GpuInfo] | None:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    out: list[GpuInfo] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            out.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mib=int(float(parts[2])),
                    free_mib=int(float(parts[3])),
                )
            )
        except ValueError:
            continue
    return out or None


def detect() -> HardwareInfo:
    ram_total, ram_avail = _ram_mib()
    gpus = _gpus_pynvml()
    source = "pynvml"
    if gpus is None:
        gpus = _gpus_nvidia_smi()
        source = "nvidia-smi"
    if gpus is None:
        gpus = []
        source = "manual"
    return HardwareInfo(
        gpus=gpus,
        vram_source=source,
        ram_total_mib=ram_total,
        ram_available_mib=ram_avail,
    )
