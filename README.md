# llamacfg — llama.cpp config manager

A local web app to manage a `llama-server --models-preset` INI router config.

It scans GGUF models (recursively, configurable roots), scans every `llama-server`
CLI flag from `--help`, lets you build multiple named configs per model with a flag
editor, suggests `n-gpu-layers` / context from your VRAM+RAM and the GGUF metadata
(MoE-aware), and composes the final INI file.

## Setup

```powershell
uv sync
```

## Run

```powershell
uv run llamacfg
# or, for dev with autoreload:
uv run uvicorn app.main:app --host 127.0.0.1 --port 8080 --reload
```

Then open http://127.0.0.1:8080

## Configuration

All paths are editable in the **Settings** tab and persisted to `data/settings.json`:

- **scan_roots** — folders scanned recursively for `*.gguf` (default `G:\Home\AI\HuggingFace\hub`)
- **llama_server_exe** — used to scan flags via `--help`
- **output_ini_path** — where the composed INI is written on Export

## Tests

```powershell
uv run pytest                    # offline unit + integration tests
uv run pytest -m requires_local  # tests needing the real exe / gguf files
```

## Notes

- The JSON store (`data/configs.json`) is the working source of truth; the INI is an
  export/import artifact. Export writes atomically; diff against your live INI before
  pointing `output_ini_path` at it.
- The suggestion math is an approximation (per-layer-via-filesize + KV estimate); the
  `fit` variant lets llama.cpp do exact final placement.
