# AI_FOR_CAVEMAN — Diamond Protocol V3.1 Automation Factory

Production-ready pipeline for the **AI FOR CAVEMAN** funnel:
1. Generate lyrics (local LLM)
2. Generate music audio (local model hook, mock supported)
3. Generate background video clips (ComfyUI/LTX2 hook, mock supported)
4. Assemble YouTube-ready MP4
5. Store provenance metadata and embedding BLOBs in SQLite (WAL mode)

> Safety rule: **no auto publishing**. Human approval is required before uploading to YouTube.

## Features

- Cross-platform Python 3.10+ (Windows + Linux)
- Per-job IPC directories under `_temp/job_<uuid>/`
- Atomic worker manifest signaling (`.tmp` + `os.replace`)
- WAL-enabled SQLite with BLOB embedding storage
- Optional ANN index path with fallback full-scan nearest-neighbor search
- Targeted process-tree termination via `psutil`
- GPU polling and baseline restoration checks via `nvidia-smi`
- Mock mode for CI/local verification (sine WAV + synthetic MP4)
- Structured logs per job in `_temp/job_<uuid>/logs/`

## Repository layout

```
AI_FOR_CAVEMAN/
  README.md
  requirements.txt
  dev-requirements.txt
  Dockerfile
  config.yaml
  setup_project.py
  models.json
  LICENSE
  _database/
  _scripts/
  _assets/
  _input/
  _output/
  _temp/
  tests/
  monitoring/
  docker/
  .github/workflows/ci.yml
```

## Quickstart (local, mock mode)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -r dev-requirements.txt
python setup_project.py --root .
python _scripts/controller.py --mode=mock --topic "Caveman synth trailer"
```

Expected result:
- Final video in `_output/`
- Job manifests/logs in `_temp/job_<uuid>/`
- DB row in `_database/mission_log.db` with embedding BLOB

## Docker (NVIDIA)

Build:
```bash
docker build -t ai_for_caveman .
```

Run with GPU:
```bash
docker run --rm --gpus all -v ${PWD}:/app ai_for_caveman
```

Alternative Dockerfile exists at `docker/Dockerfile`.

## Configuration

Edit `config.yaml`:
- `paths.*` for DB/input/output/temp assets
- `gpu.*` VRAM requirements and restore timeouts
- `vram.*` worker-specific VRAM gates (lyrics/audio/video)
- `ollama.*` Qwen model defaults and model path
- `runtime.*` (`mock_mode` default true and preflight switch)
- `models.*` local model names
- `commands.*` per-platform command templates
- `alerts.*` optional webhook URLs

## Qwen 3 14B preflight

Run this manually on the target machine before disabling mock mode:

```bash
python tools/preflight_qwen_check.py --model qwen3:14b --out preflight_report.json
```

Report fields:
- `model_found`: whether `ollama ls` detected the requested/selected model
- `peak_memory_mb`: peak GPU memory observed during minimal probe
- `baseline_free_mb`: free VRAM before probe starts
- `status`: `OK` | `TOO_MUCH_VRAM` | `MODEL_NOT_FOUND`
- `pid_snapshot`: Ollama PID snapshot before/after the probe

Safety note: **do not disable `runtime.mock_mode` in `config.yaml` until preflight returns `status: OK` on your machine.**

## Replacing mock mode with real models

### 1) Lyrics model + embeddings
- Set `LOCAL_LLM_URL` (Ollama API endpoint) if non-default.
- `worker_lyrics.py` resolves model from `config.yaml` (`ollama.model_name`) and falls back to `ollama ls` qwen detection.
- Optional embedding service: set `EMBED_SERVICE_URL`.
- If no service URL, `worker_lyrics.py` uses `sentence-transformers` local model.

### 2) Audio model hook
- Update `_scripts/worker_audio.py` function `_real_audio_hook`.
- Load your local model in that function only.
- Return stereo float32 waveform; script handles LUFS normalization and WAV writing.

### 3) Video model hook (ComfyUI/LTX2)
- Set `COMFYUI_COMMAND` env var for your headless per-job invocation.
- Update `_scripts/worker_video.py` function `_real_video_hook` for:
  - seed injection by sampler node type/name,
  - job submit/poll,
  - artifact move into job output.

### 4) Credentials and endpoints
- Keep credentials in environment variables, not in source files.
- Use local-only endpoints for generation services.

## Human-in-loop publishing policy

- Do **not** auto-publish videos.
- Review generated content, model license constraints, and fingerprint checks manually.
- `models.json` contains `commercial_use_allowed` flags.
- If any selected model is non-commercial, block automated publishing.

## Testing

```bash
pytest -q
```

Tests include:
- Utility unit tests (`atomic_write_json`, VRAM parser, process-tree kill, DB embeddings)
- End-to-end mock run (`controller --mode=mock`) validating:
  - final MP4 exists
  - loudness approximately `-14 LUFS`
  - DB row + embedding BLOB exists
  - per-job GPU snapshot logs exist

## Observability

See `monitoring/README.md` for log structure and webhook integration guidance.

## Operational checklist for real generation

1. Place required assets (`_assets/book_cover.png`) and workflow JSON (`_input/workflow_api.json`).
2. Install external tools: `ffmpeg`, `ffprobe`, and optional `fpcalc`.
3. Configure `COMFYUI_COMMAND` and verify ComfyUI/LTX2 path.
4. Set local model endpoints/credentials (`LOCAL_LLM_URL`, etc.).
5. Confirm model licenses in `models.json` permit intended use.
6. Run `python tools/preflight_qwen_check.py --model <name> --out preflight_report.json` and confirm `status` is `OK`.
7. Set `runtime.mock_mode: false` and run `python _scripts/controller.py --mode=real --topic "..."`.
8. Perform manual review before any upload/publication.
