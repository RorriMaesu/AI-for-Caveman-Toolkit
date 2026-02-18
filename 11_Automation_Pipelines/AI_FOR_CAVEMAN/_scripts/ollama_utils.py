from __future__ import annotations

import json
import subprocess
from pathlib import Path

from _scripts.caveman_utils import write_job_log


def list_models(job_dir: Path | None = None) -> list[str]:
    """Return Ollama model names parsed from `ollama ls`."""
    command = ["ollama", "ls"]
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    if job_dir is not None:
        write_job_log(
            job_dir,
            "ollama_ls.json",
            {
                "command": command,
                "return_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            },
        )

    if proc.returncode != 0:
        return []

    names: list[str] = []
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    for idx, line in enumerate(lines):
        if idx == 0 and "NAME" in line.upper():
            continue
        first_col = line.split()[0]
        if ":" in first_col:
            names.append(first_col)

    return names


def confirm_model_exists(name: str, job_dir: Path | None = None) -> bool:
    """Return True when model exists in local Ollama model list."""
    models = list_models(job_dir=job_dir)
    return name in models


def choose_qwen_model(preferred_name: str | None, available: list[str]) -> str | None:
    """Select preferred model or nearest qwen variant from available list."""
    if preferred_name:
        normalized = preferred_name.strip().lower()
        for name in available:
            if name.lower() == normalized:
                return name

    qwen_candidates = [name for name in available if "qwen" in name.lower()]
    if not qwen_candidates:
        return None

    def rank_key(model_name: str) -> tuple[int, int]:
        lowered = model_name.lower()
        has_qwen3 = 1 if "qwen3" in lowered else 0
        has_14b = 1 if "14b" in lowered else 0
        return (has_qwen3 + has_14b, len(model_name))

    ranked = sorted(qwen_candidates, key=rank_key, reverse=True)
    return ranked[0]


def write_selection_report(job_dir: Path, payload: dict[str, object]) -> None:
    """Persist model-selection metadata for troubleshooting."""
    write_job_log(job_dir, "ollama_model_selection.json", payload)
    report_path = job_dir / "ollama_model_selection.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
