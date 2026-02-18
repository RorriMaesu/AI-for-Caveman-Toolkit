from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _latest_job_dir(temp_dir: Path) -> Path:
    jobs = sorted(temp_dir.glob("job_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    assert jobs, "No job directory created"
    return jobs[0]


def test_end_to_end_mock_pipeline() -> None:
    topic = f"pytest_topic_{uuid.uuid4().hex[:8]}"

    setup_proc = subprocess.run(
        [sys.executable, "setup_project.py", "--root", str(PROJECT_ROOT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert setup_proc.returncode == 0, setup_proc.stderr

    run_proc = subprocess.run(
        [
            sys.executable,
            "_scripts/controller.py",
            "--mode=mock",
            "--topic",
            topic,
            "--config",
            "config.yaml",
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run_proc.returncode == 0, run_proc.stderr

    temp_dir = PROJECT_ROOT / "_temp"
    job_dir = _latest_job_dir(temp_dir)

    manifest_path = job_dir / "job_manifest.json"
    assert manifest_path.exists(), "job_manifest.json missing"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    final_video = Path(manifest["final_video"])
    assert final_video.exists(), "Final MP4 missing"

    measured_lufs = float(manifest["audio_metadata"]["measured_lufs"])
    assert -15.5 <= measured_lufs <= -12.5

    assemble = manifest["assemble"]
    assert assemble["codec"] in {"h264", "unknown"}
    assert assemble["pix_fmt"] in {"yuv420p", "unknown"}

    db_path = PROJECT_ROOT / "_database" / "mission_log.db"
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT id, embedding, status, final_file FROM jobs WHERE id = ?",
            (manifest["job_id"],),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[1] is not None and len(row[1]) > 0
    assert row[2] == "completed"
    assert row[3] == str(final_video)

    nvidia_log = job_dir / "logs" / "nvidia_snapshot.json"
    assert nvidia_log.exists(), "GPU snapshot log missing"
