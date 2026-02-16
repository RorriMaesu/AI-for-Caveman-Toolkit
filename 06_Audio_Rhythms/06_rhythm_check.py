# 06_rhythm_check.py
# Grug Warning: This script needs 'librosa'. It is heavy.
import sys
try:
    import librosa
    import numpy as np
except ImportError:
    print("!!! MISSING TOOLS !!! Run: pip install librosa numpy")
    sys.exit(1)

AUDIO_FILE = 'ch06_tribe_loop.wav'

def check_rhythm():
    print(f"--- Listening to {AUDIO_FILE} ---")
    try:
        y, sr = librosa.load(AUDIO_FILE)
    except FileNotFoundError:
        print(f"Error: Could not find {AUDIO_FILE}")
        return

    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Duration: {duration:.2f} seconds")

    print("Analyzing beat...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    print(f"Estimated Heartbeat: {tempo[0]:.1f} BPM")

if __name__ == "__main__":
    check_rhythm()