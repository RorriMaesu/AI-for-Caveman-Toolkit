from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _scripts.caveman_utils import (  # noqa: E402
    atomic_read_json,
    atomic_write_json,
    utc_now_iso,
    write_job_log,
)


def _synthesize_mock_audio(seconds: float = 18.0, sample_rate: int = 44100) -> np.ndarray:
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    left = 0.22 * np.sin(2 * np.pi * 220.0 * t)
    right = 0.22 * np.sin(2 * np.pi * 330.0 * t)
    stereo = np.column_stack((left, right))
    return stereo.astype(np.float32)


def _normalize_to_lufs(
    stereo_audio: np.ndarray,
    sample_rate: int,
    target_lufs: float = -14.0,
) -> tuple[np.ndarray, float]:
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(stereo_audio)
    normalized = pyln.normalize.loudness(stereo_audio, loudness, target_lufs)
    meter2 = pyln.Meter(sample_rate)
    final_loudness = meter2.integrated_loudness(normalized)
    return normalized.astype(np.float32), float(final_loudness)


def _run_fpcalc(wav_path: Path) -> dict[str, Any]:
    fpcalc = shutil.which("fpcalc")
    if not fpcalc:
        return {"available": False, "fingerprint": None, "duration": None}

    proc = subprocess.run(
        [fpcalc, str(wav_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    data: dict[str, Any] = {"available": True, "fingerprint": None, "duration": None}
    for line in proc.stdout.splitlines():
        if line.startswith("DURATION="):
            data["duration"] = line.split("=", 1)[1].strip()
        if line.startswith("FINGERPRINT="):
            data["fingerprint"] = line.split("=", 1)[1].strip()
    data["return_code"] = proc.returncode
    if proc.stderr:
        data["stderr"] = proc.stderr.strip()
    return data


def _real_audio_hook(lyrics_manifest: dict[str, Any], sample_rate: int) -> np.ndarray:
    """TODO: Replace this hook with local audio model inference.

    Expected behavior:
    1) Load local model in this subprocess only.
    2) Generate stereo audio from lyrics text.
    3) Return float32 stereo numpy array in range [-1, 1].
    """
    _ = lyrics_manifest
    _ = sample_rate
    return _synthesize_mock_audio(seconds=20.0, sample_rate=sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate normalized WAV from lyrics manifest")
    parser.add_argument("lyrics_manifest_path")
    parser.add_argument("output_wav_path")
    parser.add_argument("--mode", choices=["real", "mock"], default="mock")
    args = parser.parse_args()

    manifest_path = Path(args.lyrics_manifest_path).resolve()
    output_wav_path = Path(args.output_wav_path).resolve()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    job_dir = manifest_path.parent
    write_job_log(
        job_dir,
        "audio_start.json",
        {
            "timestamp": utc_now_iso(),
            "mode": args.mode,
            "lyrics_manifest": str(manifest_path),
        },
    )

    lyrics_manifest = atomic_read_json(manifest_path)
    sample_rate = 44100

    if args.mode == "real":
        audio = _real_audio_hook(lyrics_manifest, sample_rate)
    else:
        audio = _synthesize_mock_audio(seconds=18.0, sample_rate=sample_rate)

    normalized, measured_lufs = _normalize_to_lufs(audio, sample_rate, target_lufs=-14.0)

    sf.write(str(output_wav_path), normalized, sample_rate, subtype="PCM_16")
    fingerprint = _run_fpcalc(output_wav_path)

    metadata = {
        "timestamp": utc_now_iso(),
        "wav_path": str(output_wav_path),
        "sample_rate": sample_rate,
        "channels": 2,
        "target_lufs": -14.0,
        "measured_lufs": measured_lufs,
        "fingerprint": fingerprint,
        "mode": args.mode,
    }
    metadata_path = output_wav_path.with_suffix(".audio.json")
    atomic_write_json(metadata_path, metadata)
    write_job_log(job_dir, "audio_result.json", metadata)


if __name__ == "__main__":
    main()
