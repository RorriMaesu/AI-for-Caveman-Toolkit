from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _scripts.caveman_utils import (  # noqa: E402
    atomic_read_json,
    atomic_write_json,
    get_free_vram,
    query_nearest_embeddings,
    run_subprocess_logged,
    setup_db_index_if_missing,
    utc_now_iso,
    wait_for_gpu,
    write_job_log,
)


def _load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _effective_mode(config: dict[str, Any], cli_mode: str | None) -> str:
    if cli_mode:
        return cli_mode
    runtime_cfg = config.get("runtime", {})
    return "mock" if bool(runtime_cfg.get("mock_mode", True)) else "real"


def _insert_job(db_path: Path, job_id: str, topic: str) -> None:
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute(
            "INSERT INTO jobs(id, timestamp, topic, status) VALUES (?, ?, ?, ?)",
            (job_id, utc_now_iso(), topic, "created"),
        )
        connection.commit()
    finally:
        connection.close()


def _update_job(
    db_path: Path,
    job_id: str,
    *,
    status: str,
    model_info: dict[str, Any] | None = None,
    final_file: str | None = None,
) -> None:
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute(
            """
            UPDATE jobs
            SET status = ?,
                model_info = COALESCE(?, model_info),
                final_file = COALESCE(?, final_file)
            WHERE id = ?
            """,
            (
                status,
                json.dumps(model_info) if model_info is not None else None,
                final_file,
                job_id,
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _load_job_embedding(db_path: Path, job_id: str) -> np.ndarray | None:
    connection = sqlite3.connect(str(db_path))
    try:
        row = connection.execute(
            "SELECT embedding FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    finally:
        connection.close()

    if not row or row[0] is None:
        return None
    return np.frombuffer(row[0], dtype=np.float32)


def _run_worker(
    command: list[str],
    cwd: Path,
    job_dir: Path,
    timeout: int,
    env: dict[str, str],
    step_name: str,
    retries: int,
    backoff_seconds: int,
) -> None:
    logs_dir = job_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        stdout_path = logs_dir / f"{step_name}.attempt{attempt}.stdout.log"
        stderr_path = logs_dir / f"{step_name}.attempt{attempt}.stderr.log"
        write_job_log(
            job_dir,
            f"{step_name}.attempt{attempt}.meta.json",
            {
                "timestamp": utc_now_iso(),
                "command": command,
                "timeout": timeout,
                "attempt": attempt,
            },
        )

        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
            check=False,
        )
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")

        if proc.returncode == 0:
            return

        if attempt < retries:
            time.sleep(backoff_seconds * attempt)

    raise RuntimeError(f"Worker failed after retries: {step_name}")


def _wait_for_vram_baseline(job_dir: Path, baseline_free_mb: int, timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = get_free_vram()
        write_job_log(job_dir, "nvidia_snapshot.json", snapshot)
        if not snapshot["available"]:
            return True
        if int(snapshot["free_mb"]) >= baseline_free_mb:
            return True
        time.sleep(2)
    return False


def _call_assembler(
    python_exec: str,
    audio_path: Path,
    clips_dir: Path,
    output_path: Path,
    cover_path: Path,
    job_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    command = [
        python_exec,
        str(PROJECT_ROOT / "_scripts" / "worker_assemble.py"),
        str(audio_path),
        str(clips_dir),
        str(output_path),
        "--cover-image",
        str(cover_path),
        "--job-dir",
        str(job_dir),
    ]

    stdout_path = job_dir / "logs" / "assemble.stdout.log"
    stderr_path = job_dir / "logs" / "assemble.stderr.log"
    proc = run_subprocess_logged(
        command=command,
        cwd=PROJECT_ROOT,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError("Assembly failed. See logs.")

    return atomic_read_json(job_dir / "assemble_result.json")


def _run_preflight_controller(
    *,
    config: dict[str, Any],
    job_dir: Path,
    timeout: int,
) -> None:
    runtime_cfg = config.get("runtime", {})
    if not bool(runtime_cfg.get("preflight_on_deploy", True)):
        return

    ollama_cfg = config.get("ollama", {})
    model_name = str(ollama_cfg.get("model_name", "qwen3:14b"))
    report_path = job_dir / "preflight_report.json"

    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "preflight_qwen_check.py"),
        "--model",
        model_name,
        "--out",
        str(report_path),
        "--safety-margin-mb",
        str(config.get("vram", {}).get("safety_margin_mb", 1500)),
        "--max-tokens",
        str(ollama_cfg.get("max_tokens", 1024)),
        "--model-dir",
        str(ollama_cfg.get("model_dir", "")),
    ]
    if not bool(ollama_cfg.get("use_gpu", True)):
        command.append("--no-gpu")

    proc = run_subprocess_logged(
        command=command,
        cwd=PROJECT_ROOT,
        stdout_path=job_dir / "logs" / "preflight.controller.stdout.log",
        stderr_path=job_dir / "logs" / "preflight.controller.stderr.log",
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError("Preflight process failed before lyrics generation")

    report = atomic_read_json(report_path)
    if report.get("status") != "OK":
        raise RuntimeError(f"Preflight blocked deployment: {report.get('status')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diamond Protocol V3.1 controller")
    parser.add_argument("--topic", default="Caveman future bass")
    parser.add_argument("--mode", choices=["real", "mock"], default=None)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"))
    args = parser.parse_args()

    config = _load_config(Path(args.config).resolve())
    mode = _effective_mode(config, args.mode)
    python_exec = sys.executable

    db_path = PROJECT_ROOT / config["paths"]["database"]
    temp_dir = PROJECT_ROOT / config["paths"]["temp"]
    output_dir = PROJECT_ROOT / config["paths"]["output"]
    cover_image = PROJECT_ROOT / config["paths"]["cover_image"]
    workflow_json = PROJECT_ROOT / config["paths"]["workflow_api_json"]

    for folder in [db_path.parent, temp_dir, output_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        setup_cmd = [python_exec, str(PROJECT_ROOT / "setup_project.py"), "--root", str(PROJECT_ROOT)]
        subprocess.run(setup_cmd, check=True, timeout=120)

    job_id = str(uuid.uuid4())
    job_dir = temp_dir / f"job_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    _insert_job(db_path, job_id, args.topic)
    setup_db_index_if_missing(db_path)

    base_env = os.environ.copy()
    base_env["CAVEMAN_JOB_ID"] = job_id
    base_env["UNIQUENESS_THRESHOLD"] = str(config["uniqueness"]["threshold"])
    base_env["CAVEMAN_CONFIG_PATH"] = str(Path(args.config).resolve())

    write_job_log(
        job_dir,
        "controller_start.json",
        {
            "timestamp": utc_now_iso(),
            "job_id": job_id,
            "topic": args.topic,
            "mode": mode,
        },
    )

    retries = int(config["retries"]["max_attempts"])
    backoff = int(config["retries"]["backoff_seconds"])
    vram_cfg = config.get("vram", {})

    if mode == "real":
        _run_preflight_controller(
            config=config,
            job_dir=job_dir,
            timeout=int(config["timeouts"]["lyrics_seconds"]),
        )

    if not wait_for_gpu(
        required_mb=int(vram_cfg.get("lyrics_worker_mb", 12000)),
        safety_margin_mb=int(vram_cfg.get("safety_margin_mb", 1500)),
        timeout=int(config["gpu"]["wait_timeout_seconds"]),
    ):
        raise RuntimeError("GPU not ready for lyrics worker")

    lyrics_manifest = job_dir / "lyrics_manifest.json"
    _run_worker(
        command=[
            python_exec,
            str(PROJECT_ROOT / "_scripts" / "worker_lyrics.py"),
            args.topic,
            str(lyrics_manifest),
            str(db_path),
            "--mode",
            mode,
        ],
        cwd=PROJECT_ROOT,
        job_dir=job_dir,
        timeout=int(config["timeouts"]["lyrics_seconds"]),
        env=base_env,
        step_name="lyrics",
        retries=retries,
        backoff_seconds=backoff,
    )

    done_file = lyrics_manifest.with_suffix(".json.done")
    if not done_file.exists():
        raise RuntimeError("Lyrics worker did not emit done marker")

    _update_job(db_path, job_id, status="lyrics_complete")

    emb = _load_job_embedding(db_path, job_id)
    if emb is None:
        raise RuntimeError("Embedding not stored by lyrics worker")

    nearest = query_nearest_embeddings(db_path, emb, top_k=5)
    nearest_nonself = [n for n in nearest if n["job_id"] != job_id]
    duplicate = False
    if nearest_nonself and nearest_nonself[0]["similarity"] >= float(config["uniqueness"]["threshold"]):
        duplicate = True

    write_job_log(
        job_dir,
        "uniqueness.json",
        {
            "duplicate": duplicate,
            "neighbors": nearest_nonself,
        },
    )

    if duplicate:
        _update_job(db_path, job_id, status="duplicate_skipped")
        print(f"Job {job_id} skipped due to duplicate embedding")
        return

    baseline = get_free_vram()
    baseline_free = int(baseline.get("free_mb", 0))
    write_job_log(job_dir, "nvidia_baseline.json", baseline)

    if not wait_for_gpu(
        required_mb=int(vram_cfg.get("audio_worker_mb", 9000)),
        safety_margin_mb=int(vram_cfg.get("safety_margin_mb", 1500)),
        timeout=int(config["gpu"]["wait_timeout_seconds"]),
    ):
        raise RuntimeError("GPU not ready for audio worker")

    audio_wav = job_dir / "audio.wav"
    _run_worker(
        command=[
            python_exec,
            str(PROJECT_ROOT / "_scripts" / "worker_audio.py"),
            str(lyrics_manifest),
            str(audio_wav),
            "--mode",
            mode,
        ],
        cwd=PROJECT_ROOT,
        job_dir=job_dir,
        timeout=int(config["timeouts"]["audio_seconds"]),
        env=base_env,
        step_name="audio",
        retries=retries,
        backoff_seconds=backoff,
    )

    if not _wait_for_vram_baseline(
        job_dir,
        baseline_free_mb=baseline_free,
        timeout=int(config["gpu"]["baseline_restore_timeout_seconds"]),
    ):
        raise RuntimeError("GPU baseline was not restored after audio")

    if not wait_for_gpu(
        required_mb=int(vram_cfg.get("video_worker_mb", 9000)),
        safety_margin_mb=int(vram_cfg.get("safety_margin_mb", 1500)),
        timeout=int(config["gpu"]["wait_timeout_seconds"]),
    ):
        raise RuntimeError("GPU not ready for video worker")

    video_dir = job_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    if not workflow_json.exists():
        atomic_write_json(workflow_json, {"seed": 424242, "nodes": []})

    _run_worker(
        command=[
            python_exec,
            str(PROJECT_ROOT / "_scripts" / "worker_video.py"),
            str(workflow_json),
            str(video_dir),
            "--mode",
            mode,
        ],
        cwd=PROJECT_ROOT,
        job_dir=job_dir,
        timeout=int(config["timeouts"]["video_seconds"]),
        env=base_env,
        step_name="video",
        retries=retries,
        backoff_seconds=backoff,
    )

    if not _wait_for_vram_baseline(
        job_dir,
        baseline_free_mb=baseline_free,
        timeout=int(config["gpu"]["baseline_restore_timeout_seconds"]),
    ):
        raise RuntimeError("GPU baseline was not restored after video")

    final_path = output_dir / f"AI_FOR_CAVEMAN_{job_id}.mp4"
    assemble_result = _call_assembler(
        python_exec=python_exec,
        audio_path=audio_wav,
        clips_dir=video_dir,
        output_path=final_path,
        cover_path=cover_image,
        job_dir=job_dir,
        timeout=int(config["timeouts"]["assemble_seconds"]),
    )

    audio_meta = atomic_read_json(audio_wav.with_suffix(".audio.json"))
    lyrics_meta = atomic_read_json(lyrics_manifest)
    video_meta = atomic_read_json(video_dir / "video_manifest.json")

    model_info = {
        "mode": mode,
        "lyrics": lyrics_meta.get("model_info", {}),
        "audio": {
            "target_lufs": -14.0,
            "measured_lufs": audio_meta.get("measured_lufs"),
            "fingerprint": audio_meta.get("fingerprint"),
        },
        "video": {
            "workflow": str(workflow_json),
            "clip_count": len(video_meta.get("clips", [])),
        },
        "assemble": assemble_result,
    }

    _update_job(
        db_path,
        job_id,
        status="completed",
        model_info=model_info,
        final_file=str(final_path),
    )

    final_manifest = {
        "job_id": job_id,
        "timestamp": utc_now_iso(),
        "topic": args.topic,
        "status": "completed",
        "final_video": str(final_path),
        "job_dir": str(job_dir),
        "audio_metadata": audio_meta,
        "video_metadata": video_meta,
        "assemble": assemble_result,
    }
    atomic_write_json(job_dir / "job_manifest.json", final_manifest)

    write_job_log(job_dir, "controller_result.json", final_manifest)
    print(json.dumps(final_manifest, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"Controller failed: {exc}", file=sys.stderr)
        raise
