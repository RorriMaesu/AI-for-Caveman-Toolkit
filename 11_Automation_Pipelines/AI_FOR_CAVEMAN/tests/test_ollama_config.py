from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _scripts.controller import _effective_mode, _load_config  # noqa: E402
from _scripts.ollama_utils import choose_qwen_model  # noqa: E402
from _scripts.ollama_utils import list_models  # noqa: E402


def test_load_config_contains_qwen_defaults() -> None:
    config = _load_config(PROJECT_ROOT / "config.yaml")
    assert config["runtime"]["mock_mode"] is True
    assert config["ollama"]["model_name"] == "qwen3:14b"
    assert config["vram"]["lyrics_worker_mb"] == 12000


def test_effective_mode_prefers_runtime_when_cli_omitted() -> None:
    config = {"runtime": {"mock_mode": True}}
    assert _effective_mode(config, None) == "mock"

    config = {"runtime": {"mock_mode": False}}
    assert _effective_mode(config, None) == "real"


def test_choose_qwen_model_prefers_explicit() -> None:
    available = ["llama3.1:8b", "qwen3:14b", "qwen2.5:7b"]
    assert choose_qwen_model("qwen3:14b", available) == "qwen3:14b"


def test_choose_qwen_model_fallback_from_ollama_ls() -> None:
    available = ["mistral:7b", "qwen2.5:7b", "qwen3:14b-q4"]
    selected = choose_qwen_model(None, available)
    assert selected in {"qwen3:14b-q4", "qwen2.5:7b"}


def test_list_models_parses_ollama_ls(monkeypatch) -> None:
    class FakeResult:
        returncode = 0
        stdout = "NAME ID SIZE MODIFIED\nqwen3:14b abc 8 GB today\nllama3.1:8b def 5 GB today\n"
        stderr = ""

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        _ = kwargs
        return FakeResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert list_models() == ["qwen3:14b", "llama3.1:8b"]
