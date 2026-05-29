"""Suggestion endpoint: combine hardware + GGUF metadata -> recommendation."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core import store
from app.core.gguf_reader import read_gguf_metadata
from app.core.hardware import detect
from app.models import Suggestion, SuggestRequest
from app.core.suggester import suggest as run_suggest

router = APIRouter(prefix="/api/suggest", tags=["suggest"])


def _vram_budget(req: SuggestRequest, settings) -> tuple[int, list[str]]:
    warnings: list[str] = []
    if req.vram_budget_mode == "manual":
        mib = req.manual_vram_mib or settings.manual_vram_mib
        if not mib:
            raise HTTPException(status_code=400, detail="manual VRAM not provided")
        return int(mib), warnings
    hw = detect()
    if not hw.gpus:
        mib = req.manual_vram_mib or settings.manual_vram_mib
        if not mib:
            raise HTTPException(
                status_code=400,
                detail="no GPU detected and no manual VRAM provided",
            )
        warnings.append("no GPU detected; using manual VRAM value")
        return int(mib), warnings
    gpu = hw.gpus[0]
    return (gpu.free_mib if req.vram_budget_mode == "free" else gpu.total_mib), warnings


@router.post("", response_model=Suggestion)
def suggest(req: SuggestRequest) -> Suggestion:
    settings = store.load_settings()
    meta = read_gguf_metadata(req.model_path)
    if meta.error and not meta.n_layers:
        raise HTTPException(status_code=400, detail=f"cannot read GGUF: {meta.error}")
    vram_mib, warnings = _vram_budget(req, settings)
    mmproj_bytes = 0
    if req.mmproj_path:
        mm = read_gguf_metadata(req.mmproj_path)
        mmproj_bytes = mm.file_size_bytes or 0
    result = run_suggest(
        meta,
        vram_mib,
        context=req.context,
        ctk=req.ctk,
        ctv=req.ctv,
        mmproj_bytes=mmproj_bytes,
        headroom_frac=req.headroom_frac
        if req.headroom_frac is not None
        else settings.headroom_frac,
        compute_reserve_mib=req.compute_reserve_mib
        if req.compute_reserve_mib is not None
        else settings.compute_reserve_mib,
    )
    result.warnings = warnings + result.warnings
    return result
