@echo off
REM Launch the llamacfg web app and open it in the browser.
cd /d "%~dp0"

REM Ensure dependencies are installed (no-op if already synced).
call uv sync

REM Open the UI once the server has had a moment to start.
start "" http://127.0.0.1:8080

REM Run the server (blocks; close this window or Ctrl+C to stop).
REM --reload auto-restarts the server when source files change.
uv run uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
