from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _scripts.caveman_utils import (  # noqa: E402
    atomic_read_json,
    atomic_write_json,
    run_subprocess_logged,
    utc_now_iso,
    write_job_log,
)


def _mock_video(path: Path, seconds: int = 6, fps: int = 24, size: tuple[int, int] = (1280, 720)) -> None:
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Failed to create mock video output")

    for frame_index in range(seconds * fps):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        phase = frame_index / max(1, seconds * fps)
        frame[:, :, 0] = int(80 + 120 * phase)
        frame[:, :, 1] = int(40 + 100 * (1.0 - phase))
        frame[:, :, 2] = 160

        x_pos = int((width - 220) * phase)
        y_pos = int(height * 0.35)
        cv2.rectangle(frame, (x_pos, y_pos), (x_pos + 220, y_pos + 140), (255, 255, 255), 3)
        cv2.putText(
            frame,
            "AI FOR CAVEMAN",
            (60, height - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()


def _real_video_hook(workflow: dict[str, Any], output_dir: Path, job_dir: Path) -> Path:
    """Run per-job ComfyUI process and collect generated clip.

    TODO: Replace command and output collection with your local ComfyUI + LTX2 integration.
    """
    comfy_command = os.getenv("COMFYUI_COMMAND", "")
    if not comfy_command:
        raise RuntimeError(
            "COMFYUI_COMMAND not configured. Set command for headless per-job ComfyUI invocation."
        )

    seed = workflow.get("seed")
    write_job_log(job_dir, "video_seed.json", {"seed": seed, "workflow_keys": list(workflow.keys())})

    stdout_path = job_dir / "logs" / "video_real_stdout.log"
    stderr_path = job_dir / "logs" / "video_real_stderr.log"
    process = run_subprocess_logged(
        command=comfy_command.split(),
        cwd=PROJECT_ROOT,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=1800,
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"ComfyUI command failed with code {process.returncode}. See logs in {job_dir / 'logs'}"
        )

    candidates = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise RuntimeError("No mp4 produced by ComfyUI run")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate video clips via ComfyUI or mock mode")
    parser.add_argument("workflow_api_json")
    parser.add_argument("output_dir")
    parser.add_argument("--mode", choices=["real", "mock"], default="mock")
    args = parser.parse_args()

    workflow_path = Path(args.workflow_api_json).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    job_dir = output_dir

    write_job_log(job_dir, "video_start.json", {"timestamp": utc_now_iso(), "mode": args.mode})

    if workflow_path.exists():
        workflow = atomic_read_json(workflow_path)
    else:
        workflow = {"seed": 1234, "nodes": []}

    clip_path = output_dir / "clip_00.mp4"
    if args.mode == "real":
        generated = _real_video_hook(workflow, output_dir, job_dir)
        if generated != clip_path:
            generated.replace(clip_path)
    else:
        _mock_video(clip_path, seconds=6, fps=24)

    manifest = {
        "timestamp": utc_now_iso(),
        "mode": args.mode,
        "workflow_path": str(workflow_path),
        "clips": [str(clip_path)],
    }
    manifest_path = output_dir / "video_manifest.json"
    atomic_write_json(manifest_path, manifest)

    done_path = output_dir / "video.done"
    done_tmp = output_dir / "video.done.tmp"
    done_tmp.write_text("ok", encoding="utf-8")
    os.replace(done_tmp, done_path)

    write_job_log(job_dir, "video_result.json", manifest)


if __name__ == "__main__":
    main()
