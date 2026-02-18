from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _scripts.caveman_utils import (  # noqa: E402
    _parse_nvidia_smi_csv,
    atomic_read_json,
    atomic_write_json,
    get_free_vram,
    kill_process_tree,
    query_nearest_embeddings,
    store_embedding_to_db,
)
from setup_project import initialize_database  # noqa: E402


def test_atomic_write_and_read_json(tmp_path: Path) -> None:
    target = tmp_path / "manifest.json"
    payload = {"k": "v", "n": 42}
    atomic_write_json(target, payload)
    loaded = atomic_read_json(target)
    assert loaded == payload


def test_parse_nvidia_output() -> None:
    text = "24576, 20200\n24576, 19980\n"
    total, free = _parse_nvidia_smi_csv(text)
    assert total == 24576
    assert free == 19980


def test_get_free_vram_mocked(monkeypatch) -> None:
    class DummyResult:
        def __init__(self) -> None:
            self.stdout = "12000, 9000\n"

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = args
        _ = kwargs
        return DummyResult()

    monkeypatch.setattr(subprocess, "run", fake_run)
    snapshot = get_free_vram()
    assert snapshot["available"] is True
    assert snapshot["total_mb"] == 12000
    assert snapshot["free_mb"] == 9000


def test_kill_process_tree() -> None:
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess,sys,time;"
                "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                "time.sleep(60)"
            ),
        ]
    )
    time.sleep(1.5)
    ok = kill_process_tree(parent.pid, escalate=True)
    assert ok is True


def test_store_and_query_embeddings(tmp_path: Path) -> None:
    db_path = tmp_path / "mission_log.db"
    initialize_database(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO jobs(id, timestamp, topic, status) VALUES (?, datetime('now'), ?, ?)",
            ("job_1", "topic", "created"),
        )
        conn.execute(
            "INSERT INTO jobs(id, timestamp, topic, status) VALUES (?, datetime('now'), ?, ?)",
            ("job_2", "topic", "created"),
        )
        conn.commit()
    finally:
        conn.close()

    v1 = np.ones(8, dtype=np.float32)
    v2 = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32)
    store_embedding_to_db(db_path, "job_1", v1)
    store_embedding_to_db(db_path, "job_2", v2)

    nearest = query_nearest_embeddings(db_path, v1, top_k=2)
    assert nearest
    assert nearest[0]["job_id"] == "job_1"
