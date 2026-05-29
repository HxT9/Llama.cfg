"""CRUD for config entries + INI preview/export/import."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from app.core import store
from app.models import (
    ConfigCreate,
    ConfigEntry,
    ConfigUpdate,
    ExportRequest,
    ImportRequest,
)

router = APIRouter(prefix="/api/configs", tags=["configs"])


@router.get("", response_model=list[ConfigEntry])
def list_configs() -> list[ConfigEntry]:
    return store.load_entries()


@router.post("", response_model=ConfigEntry)
def create_config(body: ConfigCreate) -> ConfigEntry:
    return store.create_entry(
        name=body.name,
        model_display_path=body.model_display_path,
        flags=body.flags,
        mmproj=body.mmproj,
        notes=body.notes,
    )


# NOTE: declare static routes before the dynamic /{entry_id} ones
@router.get("/preview", response_class=PlainTextResponse)
def preview() -> str:
    try:
        return store.entries_to_ini_text(store.load_entries())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/export")
def export(body: ExportRequest) -> dict:
    settings = store.load_settings()
    path = body.output_path or settings.output_ini_path
    if not path:
        raise HTTPException(status_code=400, detail="no output path configured")
    try:
        text = store.export_store_to_ini(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"write failed: {e}")
    return {"path": path, "bytes": len(text.encode("utf-8")), "text": text}


@router.post("/import", response_model=list[ConfigEntry])
def import_configs(body: ImportRequest) -> list[ConfigEntry]:
    text = body.text
    if text is None:
        path = store.load_settings().output_ini_path
        try:
            with open(path, "r", encoding="utf-8") as fh:
                text = fh.read()
        except OSError as e:
            raise HTTPException(status_code=404, detail=f"cannot read {path}: {e}")
    return store.import_ini_to_store(text, replace=body.replace)


@router.get("/{entry_id}", response_model=ConfigEntry)
def get_config(entry_id: str) -> ConfigEntry:
    e = store.get_entry(entry_id)
    if e is None:
        raise HTTPException(status_code=404, detail="config not found")
    return e


@router.put("/{entry_id}", response_model=ConfigEntry)
def update_config(entry_id: str, body: ConfigUpdate) -> ConfigEntry:
    e = store.update_entry(entry_id, body.model_dump(exclude_unset=True))
    if e is None:
        raise HTTPException(status_code=404, detail="config not found")
    return e


@router.delete("/{entry_id}")
def delete_config(entry_id: str) -> dict:
    if not store.delete_entry(entry_id):
        raise HTTPException(status_code=404, detail="config not found")
    return {"deleted": entry_id}
