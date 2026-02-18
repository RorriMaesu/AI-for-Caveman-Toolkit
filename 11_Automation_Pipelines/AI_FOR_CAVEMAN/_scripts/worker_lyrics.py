from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _scripts.caveman_utils import (  # noqa: E402
    atomic_write_json,
    query_nearest_embeddings,
    store_embedding_to_db,
    utc_now_iso,
    write_job_log,
)
from _scripts.ollama_utils import choose_qwen_model, confirm_model_exists, list_models  # noqa: E402


def _load_config() -> dict[str, Any]:
    config_path = Path(os.getenv("CAVEMAN_CONFIG_PATH", PROJECT_ROOT / "config.yaml")).resolve()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _mock_embedding(text: str, dim: int = 384) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, dim).astype(np.float32)


def _service_embedding(text: str, url: str) -> np.ndarray:
    response = requests.post(url, json={"text": text}, timeout=45)
    response.raise_for_status()
    payload = response.json()
    vector = payload.get("embedding")
    if not isinstance(vector, list):
        raise ValueError("Embedding service returned invalid payload")
    return np.asarray(vector, dtype=np.float32)


def _local_embedding(text: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore

    model_name = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)
    vector = model.encode([text], normalize_embeddings=False)
    return np.asarray(vector[0], dtype=np.float32)


def _generate_lyrics_mock(topic: str, seed: int) -> str:
    lines = [
        f"{topic} in stone and neon light",
        "Fire in the cave, code in the night",
        "Drums of the mountain, synth in the sky",
        "AI for a caveman, never asking why",
        "From sparks to circuits, rhythm in clay",
        "We build tomorrow with primal sway",
    ]
    random.Random(seed).shuffle(lines)
    return "\n".join(lines)


def _resolve_ollama_model(config: dict[str, Any], job_dir: Path) -> str:
    ollama_cfg = config.get("ollama", {})
    preferred = (ollama_cfg.get("model_name") or "").strip() or None

    available = list_models(job_dir=job_dir)
    selected = choose_qwen_model(preferred, available)
    payload = {
        "preferred": preferred,
        "available": available,
        "selected": selected,
    }
    write_job_log(job_dir, "ollama_resolution.json", payload)

    if not selected:
        raise RuntimeError("No suitable Qwen model found in `ollama ls` output")
    if not confirm_model_exists(selected, job_dir=job_dir):
        raise RuntimeError(f"Selected model missing from Ollama list: {selected}")
    return selected


def _run_preflight_if_needed(
    *,
    config: dict[str, Any],
    mode: str,
    selected_model: str,
    job_dir: Path,
) -> None:
    runtime_cfg = config.get("runtime", {})
    should_preflight = bool(runtime_cfg.get("preflight_on_deploy", True))
    heavy_mode = mode == "real" and not bool(runtime_cfg.get("mock_mode", True))

    if not should_preflight or not heavy_mode:
        return

    out_path = job_dir / "preflight_report.json"
    log_out = job_dir / "logs" / "preflight.stdout.log"
    log_err = job_dir / "logs" / "preflight.stderr.log"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "preflight_qwen_check.py"),
        "--model",
        selected_model,
        "--out",
        str(out_path),
        "--safety-margin-mb",
        str(config.get("vram", {}).get("safety_margin_mb", 1500)),
        "--max-tokens",
        str(config.get("ollama", {}).get("max_tokens", 1024)),
    ]
    if not bool(config.get("ollama", {}).get("use_gpu", True)):
        command.append("--no-gpu")

    proc = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )
    log_out.parent.mkdir(parents=True, exist_ok=True)
    log_out.write_text(proc.stdout or "", encoding="utf-8")
    log_err.write_text(proc.stderr or "", encoding="utf-8")

    report: dict[str, Any] = {}
    if out_path.exists():
        report = json.loads(out_path.read_text(encoding="utf-8"))
    if proc.returncode != 0 or report.get("status") not in {"OK", None}:
        if not report:
            report = {
                "status": "FAILED",
                "reason": "preflight command failed",
                "return_code": proc.returncode,
            }
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise RuntimeError(f"Preflight check failed: {report.get('status', 'FAILED')}")


def _generate_lyrics_real(
    topic: str,
    seed: int,
    model: str,
    max_tokens: int,
    use_gpu: bool,
    job_dir: Path,
) -> str:
    url = os.getenv("LOCAL_LLM_URL", "http://127.0.0.1:11434/api/generate")
    prompt = (
        "Write song lyrics for a cinematic AI music video funnel. "
        f"Topic: {topic}. 16-20 lines, no profanity, memorable chorus."
    )
    options: dict[str, Any] = {
        "seed": seed,
        "temperature": 0.8,
        "num_predict": max_tokens,
    }
    if not use_gpu:
        options["num_gpu"] = 0

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    write_job_log(
        job_dir,
        "lyrics_model_request.json",
        {
            "url": url,
            "payload": {
                "model": model,
                "stream": False,
                "options": options,
                "prompt_preview": prompt[:180],
            },
        },
    )
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()
    write_job_log(
        job_dir,
        "lyrics_model_response.json",
        {
            "status_code": response.status_code,
            "response_preview": response.text[:500],
        },
    )
    content = response.json().get("response", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Local LLM returned empty response")
    return content.strip()


def _fallback_lyrics(topic: str) -> str:
    return (
        f"[{topic}]\n"
        "Verse: Stone hands shaping electric dreams.\n"
        "Hook: AI for a caveman, rise and sing.\n"
        "Outro: We climb from firelight to satellite."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unique lyrics + embedding manifest")
    parser.add_argument("topic")
    parser.add_argument("output_manifest_path")
    parser.add_argument("db_path")
    parser.add_argument("--mode", choices=["real", "mock"], default="mock")
    args = parser.parse_args()

    output_manifest_path = Path(args.output_manifest_path).resolve()
    job_dir = output_manifest_path.parent
    job_dir.mkdir(parents=True, exist_ok=True)

    job_id = os.getenv("CAVEMAN_JOB_ID")
    if not job_id:
        raise RuntimeError("CAVEMAN_JOB_ID must be set by controller")

    db_path = Path(args.db_path).resolve()
    config = _load_config()
    runtime_cfg = config.get("runtime", {})
    ollama_cfg = config.get("ollama", {})
    retries = 5
    uniqueness_threshold = float(os.getenv("UNIQUENESS_THRESHOLD", "0.92"))
    heavy_real_mode = args.mode == "real" and not bool(runtime_cfg.get("mock_mode", True))

    selected_model = "mock-template"
    if heavy_real_mode:
        selected_model = _resolve_ollama_model(config=config, job_dir=job_dir)
        _run_preflight_if_needed(
            config=config,
            mode=args.mode,
            selected_model=selected_model,
            job_dir=job_dir,
        )

    write_job_log(
        job_dir,
        "lyrics_start.json",
        {
            "timestamp": utc_now_iso(),
            "topic": args.topic,
            "mode": args.mode,
            "job_id": job_id,
        },
    )

    last_error = None
    lyrics = ""
    embedding = np.array([], dtype=np.float32)
    selected_seed = 0

    for attempt in range(1, retries + 1):
        selected_seed = 1000 + attempt
        try:
            if heavy_real_mode:
                lyrics = _generate_lyrics_real(
                    args.topic,
                    selected_seed,
                    selected_model,
                    int(ollama_cfg.get("max_tokens", 1024)),
                    bool(ollama_cfg.get("use_gpu", True)),
                    job_dir,
                )
                embed_service = os.getenv("EMBED_SERVICE_URL", "").strip()
                if embed_service:
                    embedding = _service_embedding(lyrics, embed_service)
                else:
                    embedding = _local_embedding(lyrics)
            else:
                lyrics = _generate_lyrics_mock(args.topic, selected_seed)
                embedding = _mock_embedding(lyrics)

            nearest = query_nearest_embeddings(db_path, embedding, top_k=3)
            too_close = any(item["similarity"] >= uniqueness_threshold for item in nearest)
            if too_close and attempt < retries:
                continue
            if too_close and attempt == retries:
                lyrics = _fallback_lyrics(args.topic)
                embedding = _mock_embedding(lyrics)

            break
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)

    if embedding.size == 0:
        raise RuntimeError(f"Failed to produce embedding: {last_error}")

    lyrics_path = job_dir / "lyrics.txt"
    lyrics_path.write_text(lyrics, encoding="utf-8")

    store_embedding_to_db(db_path, job_id, embedding)

    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute(
            "UPDATE jobs SET lyrics = ?, status = ? WHERE id = ?",
            (lyrics, "lyrics_generated", job_id),
        )
        connection.commit()
    finally:
        connection.close()

    embedding_npy_path = job_dir / "embedding.npy"
    np.save(embedding_npy_path, embedding)

    manifest = {
        "job_id": job_id,
        "timestamp": utc_now_iso(),
        "topic": args.topic,
        "lyrics_path": str(lyrics_path),
        "embedding_path": str(embedding_npy_path),
        "embedding_dim": int(embedding.shape[0]),
        "model_info": {
            "lyrics_model": selected_model,
            "embedding_model": os.getenv(
                "EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            "mode": args.mode,
            "effective_runtime": "real" if heavy_real_mode else "mock",
        },
        "seed": selected_seed,
    }
    atomic_write_json(output_manifest_path, manifest)
    done_path = output_manifest_path.with_suffix(output_manifest_path.suffix + ".done")
    done_path_tmp = done_path.with_suffix(done_path.suffix + ".tmp")
    done_path_tmp.write_text("ok", encoding="utf-8")
    os.replace(done_path_tmp, done_path)

    write_job_log(job_dir, "lyrics_result.json", manifest)


if __name__ == "__main__":
    main()
