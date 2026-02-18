from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DEFAULT_FOLDERS = [
    "_database",
    "_scripts",
    "tools",
    "_assets",
    "_input",
    "_output",
    "_temp",
    "monitoring",
    "docker",
    "tests",
    ".github/workflows",
]


def ensure_directories(root_dir: Path) -> None:
    """Create required project directories."""
    for rel in DEFAULT_FOLDERS:
        (root_dir / rel).mkdir(parents=True, exist_ok=True)


def initialize_database(db_path: Path) -> None:
    """Create mission database with WAL mode and jobs table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                topic TEXT NOT NULL,
                lyrics TEXT,
                embedding BLOB,
                model_info TEXT,
                status TEXT NOT NULL,
                final_file TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS job_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL,
                event_time TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize AI_FOR_CAVEMAN project structure")
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parent),
        help="Project root path",
    )
    args = parser.parse_args()

    root_dir = Path(args.root).resolve()
    ensure_directories(root_dir)
    initialize_database(root_dir / "_database" / "mission_log.db")
    print(f"Project initialized at: {root_dir}")


if __name__ == "__main__":
    main()
