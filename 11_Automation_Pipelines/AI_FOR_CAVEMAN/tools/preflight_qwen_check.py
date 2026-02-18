from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _scripts.caveman_utils import get_free_vram, kill_process_tree, write_job_log  # noqa: E402
from _scripts.ollama_utils import choose_qwen_model, confirm_model_exists, list_models  # noqa: E402


def _safe_snapshot() -> dict[str, Any]:
    snapshot = get_free_vram()
    return {
        "available": bool(snapshot.get("available")),
        "total_mb": int(snapshot.get("total_mb", 0)),
        "free_mb": int(snapshot.get("free_mb", 0)),
        "used_mb": max(int(snapshot.get("total_mb", 0)) - int(snapshot.get("free_mb", 0)), 0),
        "timestamp": time.time(),
    }


def _find_ollama_pids() -> list[int]:
    pids: list[int] = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()
            if "ollama" in name or "ollama" in cmdline:
                pids.append(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(set(pids))


def _run_ollama_ls(log_dir: Path) -> list[str]:
    models = list_models(job_dir=log_dir)
    write_job_log(log_dir, "preflight_ollama_ls_result.json", {"models": models})
    return models


def _probe_model(
    model_name: str,
    log_dir: Path,
    use_gpu: bool,
    max_tokens: int,
) -> tuple[int, list[dict[str, Any]], str, str]:
    command = [
        "ollama",
        "run",
        model_name,
        "hello",
        "--nowordwrap",
    ]
    if not use_gpu:
        command.extend(["--verbose"])

    env = os.environ.copy()
    if not use_gpu:
        env["OLLAMA_NUM_GPU"] = "0"

    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    write_job_log(
        log_dir,
        "preflight_ollama_run_start.json",
        {
            "command": command,
            "pid": proc.pid,
            "use_gpu": use_gpu,
            "max_tokens_hint": max_tokens,
        },
    )

    timeline: list[dict[str, Any]] = []
    nvidia_invocations: list[dict[str, Any]] = []
    start = time.time()

    while proc.poll() is None:
        snap = _safe_snapshot()
        timeline.append(snap)
        nvidia_invocations.append(
            {
                "timestamp": snap["timestamp"],
                "used_mb": snap["used_mb"],
                "free_mb": snap["free_mb"],
            }
        )
        if time.time() - start > 60:
            proc.terminate()
            break
        time.sleep(0.2)

    stdout, stderr = proc.communicate(timeout=20)
    write_job_log(
        log_dir,
        "preflight_ollama_run_result.json",
        {
            "return_code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "timeline_count": len(timeline),
            "nvidia_smi_samples": nvidia_invocations,
        },
    )

    return proc.pid, timeline, stdout, stderr


def _wait_baseline_restore(log_dir: Path, baseline_free_mb: int, timeout_seconds: int = 60) -> dict[str, Any]:
    deadline = time.time() + timeout_seconds
    checks: list[dict[str, Any]] = []
    restored = False

    while time.time() < deadline:
        snap = _safe_snapshot()
        checks.append(snap)
        if not snap["available"]:
            restored = True
            break
        if snap["free_mb"] >= baseline_free_mb:
            restored = True
            break
        time.sleep(1)

    write_job_log(
        log_dir,
        "preflight_baseline_restore.json",
        {
            "baseline_free_mb": baseline_free_mb,
            "restored": restored,
            "checks": checks,
        },
    )
    return {"restored": restored, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely preflight Qwen model on Ollama")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--out", default="preflight_report.json")
    parser.add_argument("--safety-margin-mb", type=int, default=1500)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--use-gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_path.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    use_gpu = args.use_gpu and not args.no_gpu

    if args.model_dir:
        os.environ["OLLAMA_MODELS"] = args.model_dir

    baseline = _safe_snapshot()
    available_models = _run_ollama_ls(log_dir)
    selected_model = choose_qwen_model(args.model, available_models)

    model_found = bool(selected_model) and confirm_model_exists(selected_model, job_dir=log_dir)
    pid_snapshot_before = _find_ollama_pids()

    report: dict[str, Any] = {
        "model_requested": args.model,
        "model_selected": selected_model,
        "model_found": model_found,
        "status": "MODEL_NOT_FOUND",
        "peak_memory_mb": 0,
        "baseline_free_mb": baseline["free_mb"],
        "baseline_total_mb": baseline["total_mb"],
        "safety_margin_mb": args.safety_margin_mb,
        "timeline": [],
        "pid_snapshot": {
            "before": pid_snapshot_before,
            "after": [],
        },
        "ollama_model_dir": args.model_dir or os.getenv("OLLAMA_MODELS", ""),
    }

    if not model_found:
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_job_log(log_dir, "preflight_report.json", report)
        print(json.dumps(report, indent=2))
        sys.exit(1)

    run_pid = -1
    timeline: list[dict[str, Any]] = []
    stdout = ""
    stderr = ""
    try:
        run_pid, timeline, stdout, stderr = _probe_model(
            selected_model,
            log_dir=log_dir,
            use_gpu=use_gpu,
            max_tokens=args.max_tokens,
        )
    finally:
        if run_pid > 0 and psutil.pid_exists(run_pid):
            kill_process_tree(run_pid, escalate=True)

    peak_used = max((item["used_mb"] for item in timeline), default=0)
    total_mb = int(baseline["total_mb"])

    status = "OK"
    if total_mb > 0 and (peak_used + args.safety_margin_mb) > total_mb:
        status = "TOO_MUCH_VRAM"

    restore = _wait_baseline_restore(log_dir, baseline_free_mb=int(baseline["free_mb"]))
    pid_snapshot_after = _find_ollama_pids()

    report.update(
        {
            "status": status,
            "peak_memory_mb": peak_used,
            "timeline": timeline,
            "baseline_restored": restore["restored"],
            "pid_snapshot": {
                "before": pid_snapshot_before,
                "after": pid_snapshot_after,
            },
            "ollama_stdout": stdout,
            "ollama_stderr": stderr,
        }
    )

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_job_log(log_dir, "preflight_report.json", report)
    print(json.dumps(report, indent=2))

    if status != "OK":
        sys.exit(2)


if __name__ == "__main__":
    main()
