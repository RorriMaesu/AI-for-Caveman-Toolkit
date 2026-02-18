from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _scripts.caveman_utils import atomic_write_json, utc_now_iso, write_job_log  # noqa: E402


def _audio_duration(audio_path: Path) -> float:
    data, sample_rate = sf.read(str(audio_path))
    return float(len(data) / sample_rate)


def _run(command: list[str], job_dir: Path, timeout: int = 1800) -> None:
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    write_job_log(job_dir, "assemble_stdout.log", proc.stdout or "")
    write_job_log(job_dir, "assemble_stderr.log", proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed with code {proc.returncode}")


def _ffprobe_codec_info(video_path: Path) -> dict[str, str]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"codec": "unknown", "pix_fmt": "unknown"}

    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt",
            "-of",
            "default=noprint_wrappers=1:nokey=0",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    codec = "unknown"
    pix_fmt = "unknown"
    for line in proc.stdout.splitlines():
        if line.startswith("codec_name="):
            codec = line.split("=", 1)[1].strip()
        if line.startswith("pix_fmt="):
            pix_fmt = line.split("=", 1)[1].strip()
    return {"codec": codec, "pix_fmt": pix_fmt}


def assemble_video(
    audio_path: Path,
    clips: list[Path],
    output_path: Path,
    cover_image: Path | None,
    job_dir: Path,
    cover_width_pct: float = 0.2,
    cover_margin_px: int = 24,
) -> dict[str, str | float]:
    duration = _audio_duration(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_clips = [clip for clip in clips if clip.exists()]
    if not source_clips:
        raise RuntimeError("No video clips found for assembly")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        shutil.copy2(source_clips[0], output_path)
        result = {
            "output_path": str(output_path),
            "duration": duration,
            "codec": "unknown",
            "pix_fmt": "unknown",
        }
        atomic_write_json(
            job_dir / "assemble_result.json",
            {"timestamp": utc_now_iso(), "fallback": "ffmpeg_missing", **result},
        )
        return result

    target_count = max(3, math.ceil(duration / 6.0))
    while len(source_clips) < target_count:
        source_clips.append(source_clips[len(source_clips) % len(source_clips)])

    xfade_duration = 0.45
    segment = (duration + xfade_duration * (len(source_clips) - 1)) / len(source_clips)

    cmd: list[str] = [ffmpeg, "-y"]
    for clip in source_clips:
        cmd.extend(["-stream_loop", "-1", "-t", f"{segment:.3f}", "-i", str(clip)])

    cmd.extend(["-i", str(audio_path)])

    if cover_image and cover_image.exists():
        cmd.extend(["-i", str(cover_image)])

    filter_parts: list[str] = []
    for idx in range(len(source_clips)):
        filter_parts.append(f"[{idx}:v]setpts=PTS-STARTPTS,scale=1280:720[v{idx}]")

    chain_label = "v0"
    offset = segment - xfade_duration
    for idx in range(1, len(source_clips)):
        next_label = f"x{idx}"
        filter_parts.append(
            f"[{chain_label}][v{idx}]xfade=transition=fade:duration={xfade_duration}:offset={offset:.3f}[{next_label}]"
        )
        chain_label = next_label
        offset += segment - xfade_duration

    if cover_image and cover_image.exists():
        cover_index = len(source_clips) + 1
        filter_parts.append(
            f"[{cover_index}:v]scale=iw*{cover_width_pct}:-1[cover]"
        )
        filter_parts.append(
            f"[{chain_label}][cover]overlay=W-w-{cover_margin_px}:H-h-{cover_margin_px}[vfinal]"
        )
        map_label = "vfinal"
    else:
        map_label = chain_label

    filter_complex = ";".join(filter_parts)

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            f"[{map_label}]",
            "-map",
            f"{len(source_clips)}:a:0",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-t",
            f"{duration:.3f}",
            "-shortest",
            str(output_path),
        ]
    )

    _run(cmd, job_dir)

    codec_info = _ffprobe_codec_info(output_path)
    result = {
        "output_path": str(output_path),
        "duration": duration,
        "codec": codec_info["codec"],
        "pix_fmt": codec_info["pix_fmt"],
    }
    atomic_write_json(job_dir / "assemble_result.json", {"timestamp": utc_now_iso(), **result})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble final YouTube-ready video")
    parser.add_argument("audio_file")
    parser.add_argument("clips_dir")
    parser.add_argument("output_file")
    parser.add_argument("--cover-image", default="")
    parser.add_argument("--job-dir", default="")
    args = parser.parse_args()

    audio_path = Path(args.audio_file).resolve()
    clips_dir = Path(args.clips_dir).resolve()
    output_path = Path(args.output_file).resolve()
    job_dir = Path(args.job_dir).resolve() if args.job_dir else output_path.parent

    clips = sorted(clips_dir.glob("*.mp4"))
    cover = Path(args.cover_image).resolve() if args.cover_image else None
    result = assemble_video(audio_path, clips, output_path, cover, job_dir)
    print(result)


if __name__ == "__main__":
    main()
