# PATCH_QWEN_UPDATE

## Exact edits applied

1. Updated `config.yaml`:
   - Added `ollama` defaults with `model_name: qwen3:14b` and model path.
   - Added `vram` block using conservative values.
   - Added `runtime` block with `mock_mode: true` and `preflight_on_deploy: true`.

2. Added `_scripts/ollama_utils.py`:
   - `list_models()` parses `ollama ls`.
   - `confirm_model_exists(name)` returns presence.
   - `choose_qwen_model()` fallback selection for qwen variants.

3. Added `tools/preflight_qwen_check.py`:
   - Validates model existence via `ollama ls`.
   - Performs minimal `ollama run` probe.
   - Samples GPU memory timeline and peak usage.
   - Writes status report (`OK`, `TOO_MUCH_VRAM`, `MODEL_NOT_FOUND`).
   - Captures command logs and waits for baseline VRAM recovery.

4. Updated `_scripts/worker_lyrics.py`:
   - Loads `config.yaml` using `CAVEMAN_CONFIG_PATH`.
   - Uses config-driven Ollama model choice.
   - Runs preflight in real mode when enabled.
   - Guards heavy model calls unless `--mode real` and `runtime.mock_mode=false`.

5. Updated `_scripts/controller.py`:
   - Uses config runtime default mode when `--mode` is omitted.
   - Runs preflight before real-mode lyrics worker.
   - Uses `config.vram.*` values for GPU waiting.

6. Added tests:
   - `tests/test_ollama_config.py` for config parsing and model choice logic.
   - `tests/test_preflight_skip_ci.py` to confirm preflight script presence with CI skip.

7. Updated docs:
   - Added Qwen preflight instructions and safety notes to `README.md`.
   - Added `preflight_report.json.example`.

## Recommended VRAM starting configuration (16 GB GPU)

- `lyrics_worker_mb`: **12000**
- `audio_worker_mb`: **9000**
- `video_worker_mb`: **9000**
- `safety_margin_mb`: **1500**

These are conservative defaults intended to avoid OOM during initial rollout.
