"""Model scanning + per-file GGUF metadata endpoints."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.core import store
from app.core.gguf_reader import read_gguf_metadata
from app.core.model_scanner import scan
from app.models import GgufMetadata, ScanResult

router = APIRouter(prefix="/api/models", tags=["models"])

# simple in-process cache of the last scan
_last: ScanResult | None = None


@router.get("", response_model=ScanResult)
def list_models() -> ScanResult:
    global _last
    if _last is None:
        _last = scan(store.load_settings().scan_roots)
    return _last


@router.post("/scan", response_model=ScanResult)
def rescan() -> ScanResult:
    global _last
    _last = scan(store.load_settings().scan_roots)
    return _last


@router.get("/metadata", response_model=GgufMetadata)
def metadata(path: str = Query(...)) -> GgufMetadata:
    meta = read_gguf_metadata(path)
    if meta.error and not meta.file_size_bytes:
        raise HTTPException(status_code=404, detail=meta.error)
    return meta
