from __future__ import annotations

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psutil


def utc_now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def write_job_log(job_dir: Path | str, name: str, data: Any) -> Path:
    """Write structured log payload under job logs directory."""
    base = Path(job_dir)
    logs_dir = base / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    target = logs_dir / name

    if isinstance(data, (dict, list)):
        payload = json.dumps(data, ensure_ascii=False, indent=2)
    else:
        payload = str(data)

    target.write_text(payload, encoding="utf-8")
    return target


def atomic_write_json(path: Path | str, data: dict[str, Any]) -> None:
    """Write JSON atomically via temporary file + replace."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path = target.with_suffix(target.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, target)


def atomic_read_json(path: Path | str) -> dict[str, Any]:
    """Read JSON file into dictionary."""
    target = Path(path)
    return json.loads(target.read_text(encoding="utf-8"))


def _parse_nvidia_smi_csv(text: str) -> tuple[int, int]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("No GPU memory rows found")

    total_values: list[int] = []
    free_values: list[int] = []
    for line in lines:
        parts = [part.strip().replace(" MiB", "") for part in line.split(",")]
        if len(parts) < 2:
            continue
        total_values.append(int(parts[0]))
        free_values.append(int(parts[1]))

    if not total_values:
        raise ValueError("Unable to parse nvidia-smi output")

    return min(total_values), min(free_values)


def get_free_vram() -> dict[str, int | bool]:
    """Return VRAM stats from nvidia-smi, robust to missing tool."""
    command = [
        "nvidia-smi",
        "--query-gpu=memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        total_mb, free_mb = _parse_nvidia_smi_csv(proc.stdout)
        return {
            "available": True,
            "total_mb": total_mb,
            "free_mb": free_mb,
        }
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return {
            "available": False,
            "total_mb": 0,
            "free_mb": 0,
        }


def wait_for_gpu(
    required_mb: int,
    safety_margin_mb: int = 1500,
    timeout: int = 600,
    poll_interval: int = 3,
) -> bool:
    """Poll GPU memory until enough free VRAM is available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        snapshot = get_free_vram()
        if not snapshot["available"]:
            return True
        free_mb = int(snapshot["free_mb"])
        if free_mb >= required_mb + safety_margin_mb:
            return True
        time.sleep(poll_interval)
    return False


def _descendants(pid: int) -> list[psutil.Process]:
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return []

    try:
        return parent.children(recursive=True)
    except psutil.Error:
        return []


def kill_process_tree(pid: int, escalate: bool = True) -> bool:
    """Terminate process tree gracefully, then force-kill if needed."""
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return True

    processes = _descendants(pid)
    processes.append(parent)

    for process in processes:
        try:
            process.terminate()
        except psutil.Error:
            continue

    _, alive = psutil.wait_procs(processes, timeout=8)

    if alive and escalate:
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                continue
        psutil.wait_procs(alive, timeout=5)

    for process in _descendants(pid):
        try:
            process.send_signal(signal.SIGTERM)
        except (psutil.Error, AttributeError):
            continue

    return not psutil.pid_exists(pid)


def _ensure_wal(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL;")


def _to_blob(np_array: np.ndarray) -> bytes:
    arr = np.asarray(np_array, dtype=np.float32)
    return arr.tobytes(order="C")


def _from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def store_embedding_to_db(db_path: Path | str, job_id: str, np_array: np.ndarray) -> None:
    """Store embedding vector as float32 BLOB in jobs table."""
    connection = sqlite3.connect(str(db_path))
    try:
        _ensure_wal(connection)
        connection.execute(
            "UPDATE jobs SET embedding = ? WHERE id = ?",
            (_to_blob(np_array), job_id),
        )
        connection.commit()
    finally:
        connection.close()


def query_nearest_embeddings(
    db_path: Path | str,
    vector: np.ndarray,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return top nearest job embeddings using cosine similarity.

    If optional hnswlib index is available and built, use it; otherwise fallback
    to full scan in SQLite-backed vectors.
    """
    base_vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(base_vector)
    if norm == 0:
        return []
    base_vector = base_vector / norm

    index_meta = setup_db_index_if_missing(db_path)
    hnsw_path = Path(index_meta["hnsw_index_path"])
    labels_path = Path(index_meta["hnsw_labels_path"])

    if hnsw_path.exists() and labels_path.exists():
        try:
            import hnswlib  # type: ignore

            labels = json.loads(labels_path.read_text(encoding="utf-8"))
            index = hnswlib.Index(space="cosine", dim=base_vector.shape[0])
            index.load_index(str(hnsw_path))
            k = min(top_k, len(labels))
            if k <= 0:
                return []
            ids, distances = index.knn_query(base_vector, k=k)
            rows = []
            for idx, distance in zip(ids[0], distances[0]):
                job_id = labels[int(idx)]
                rows.append({"job_id": job_id, "similarity": 1.0 - float(distance)})
            return rows
        except Exception:
            pass

    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            "SELECT id, embedding FROM jobs WHERE embedding IS NOT NULL"
        ).fetchall()
    finally:
        connection.close()

    scored: list[dict[str, Any]] = []
    for job_id, blob in rows:
        other = _from_blob(blob)
        other_norm = np.linalg.norm(other)
        if other_norm == 0:
            continue
        similarity = float(np.dot(base_vector, other / other_norm))
        scored.append({"job_id": job_id, "similarity": similarity})

    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[:top_k]


def setup_db_index_if_missing(db_path: Path | str) -> dict[str, str]:
    """Ensure embedding index metadata exists and optionally build hnsw index.

    For small datasets, full scan fallback is used automatically.
    """
    db = Path(db_path)
    index_dir = db.parent / "vector_index"
    index_dir.mkdir(parents=True, exist_ok=True)

    hnsw_index_path = index_dir / "embeddings.hnsw"
    labels_path = index_dir / "labels.json"

    connection = sqlite3.connect(str(db))
    try:
        _ensure_wal(connection)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS vector_index_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO vector_index_meta(key, value) VALUES(?, ?)",
            ("hnsw_index_path", str(hnsw_index_path)),
        )
        connection.execute(
            "INSERT OR REPLACE INTO vector_index_meta(key, value) VALUES(?, ?)",
            ("hnsw_labels_path", str(labels_path)),
        )
        connection.commit()

        rows = connection.execute(
            "SELECT id, embedding FROM jobs WHERE embedding IS NOT NULL"
        ).fetchall()
    finally:
        connection.close()

    if len(rows) < 50:
        return {
            "hnsw_index_path": str(hnsw_index_path),
            "hnsw_labels_path": str(labels_path),
        }

    try:
        import hnswlib  # type: ignore

        first_vec = _from_blob(rows[0][1])
        dim = int(first_vec.shape[0])
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(rows), ef_construction=200, M=16)

        labels: list[str] = []
        vectors = np.zeros((len(rows), dim), dtype=np.float32)
        for idx, (job_id, blob) in enumerate(rows):
            labels.append(job_id)
            vectors[idx] = _from_blob(blob)

        index.add_items(vectors, np.arange(len(rows)))
        index.save_index(str(hnsw_index_path))
        labels_path.write_text(json.dumps(labels), encoding="utf-8")
    except Exception:
        pass

    return {
        "hnsw_index_path": str(hnsw_index_path),
        "hnsw_labels_path": str(labels_path),
    }


def command_exists(command_name: str) -> bool:
    """Return True if command is in PATH."""
    return shutil.which(command_name) is not None


def run_subprocess_logged(
    command: list[str],
    cwd: Path | str,
    stdout_path: Path,
    stderr_path: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    """Run command with captured stdout/stderr redirected to files."""
    process = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(process.stdout or "", encoding="utf-8")
    stderr_path.write_text(process.stderr or "", encoding="utf-8")
    return process
