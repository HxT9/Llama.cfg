"""Hardware detection endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app.core.hardware import detect
from app.models import HardwareInfo

router = APIRouter(prefix="/api/hardware", tags=["hardware"])


@router.get("", response_model=HardwareInfo)
def hardware() -> HardwareInfo:
    return detect()
