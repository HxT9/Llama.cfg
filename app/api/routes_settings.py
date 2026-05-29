"""Settings get/update endpoints."""
from __future__ import annotations

from fastapi import APIRouter

from app.core import store
from app.models import Settings

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("", response_model=Settings)
def get_settings() -> Settings:
    return store.load_settings()


@router.put("", response_model=Settings)
def update_settings(body: Settings) -> Settings:
    return store.save_settings(body)
