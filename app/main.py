"""FastAPI app factory + uvicorn entry point."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import config as cfg
from app.api import (
    routes_configs,
    routes_flags,
    routes_hardware,
    routes_models,
    routes_settings,
    routes_suggest,
)


def create_app() -> FastAPI:
    cfg.ensure_data_dir()
    app = FastAPI(title="llamacfg", version="0.1.0")

    app.include_router(routes_models.router)
    app.include_router(routes_flags.router)
    app.include_router(routes_configs.router)
    app.include_router(routes_hardware.router)
    app.include_router(routes_suggest.router)
    app.include_router(routes_settings.router)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(cfg.STATIC_DIR / "index.html")

    if cfg.STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(cfg.STATIC_DIR)), name="static")

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8080)
