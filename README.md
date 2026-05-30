# llama.cfg

A small local web app that helps you build the `llama-server` config file
(the `--models-preset` INI) without hand-editing it.

It scans your GGUF models, knows every `llama-server` flag, and — given your
GPU's VRAM — suggests how many layers/experts to put on the GPU and how much
context you can afford. Then it writes the INI for you.

## Why

Maintaining the preset INI by hand is fiddly:

- model paths are buried deep in the HuggingFace cache
- `llama-server` has ~250 flags
- picking `n-gpu-layers` / context / `n-cpu-moe` for a given GPU is guesswork,
  especially for MoE and sliding-window models

This tool does the bookkeeping and the math.

## Quick start

1. Double-click **`start.bat`** (or run it from a terminal).
2. It opens **http://127.0.0.1:8080** in your browser.

That's it. The window stays open while the app runs; close it or press Ctrl+C
to stop.

> First run installs dependencies automatically (via [uv](https://docs.astral.sh/uv/)).
> Make sure `uv` is installed.

## How to use it

The app has four tabs:

- **Models** — lists every `.gguf` it found (name, architecture, layers, size,
  MoE badge). Click **New config** on a model to start a config for it.
- **Configs** — your saved configs, grouped by model (you can have several per
  model). Each config has:
  - a flag editor (a curated *common* set plus a searchable *all flags* list)
  - a **live VRAM estimate** that updates as you change context / ngl / cache
    type / n-cpu-moe
  - a **Suggest** button that reads your VRAM and proposes settings. You get two
    one-click options: **explicit** (`-ngl` / `-c` / `n-cpu-moe`) or **fit**
    (let llama.cpp auto-size).
- **INI Preview** — see the generated file, then **Export** it to disk (or
  **Import** an existing one).
- **Settings** — where to scan for models, the `llama-server.exe` path, the
  output INI path, and VRAM headroom.

## A couple of things to know

- The app keeps its own working copy of your configs; the INI is only written
  when you click **Export**. It's safe to experiment.
- The VRAM suggestion is a smart estimate (it understands MoE experts,
  sliding-window attention, and hybrid models), not a guarantee — treat it as a
  strong starting point and nudge if needed.
- Settings, remembered values, and your last suggestion inputs persist between
  sessions.

## For developers

```powershell
uv sync                 # install
uv run uvicorn app.main:app --reload   # dev server
uv run pytest           # tests
```

Backend is FastAPI; the frontend is plain HTML/JS (no build step). GGUF parsing,
flag scanning, and the suggestion math live in `app/core/`.
