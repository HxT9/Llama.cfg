"""llama-server flag scanning endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core import store
from app.core.flag_scanner import get_flags
from app.models import FlagSpec

router = APIRouter(prefix="/api/flags", tags=["flags"])


def _get(force: bool) -> list[FlagSpec]:
    exe = store.load_settings().llama_server_exe
    try:
        return get_flags(exe, force_refresh=force)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:  # subprocess / parse errors
        raise HTTPException(status_code=500, detail=f"flag scan failed: {e}")


@router.get("", response_model=list[FlagSpec])
def list_flags() -> list[FlagSpec]:
    return _get(force=False)


@router.post("/refresh", response_model=list[FlagSpec])
def refresh_flags() -> list[FlagSpec]:
    return _get(force=True)
