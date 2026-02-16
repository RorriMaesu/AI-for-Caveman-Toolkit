# 10_check_voice.py
# Needs ffmpeg installed.
import sys, os
try:
    from pydub import AudioSegment
except ImportError:
    print("Missing pydub. Run: pip install pydub")
    sys.exit(1)

AUDIO_FILE = 'ch10_vo_sample.mp3'

def check_voice():
    if not os.path.exists(AUDIO_FILE):
        print(f"Where is {AUDIO_FILE}?")
        return

    try:
        sound = AudioSegment.from_file(AUDIO_FILE)
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"Duration: {len(sound) / 1000.0:.2f}s")
    print(f"Loudness: {sound.rms_dBFS:.2f} dBFS")

if __name__ == "__main__":
    check_voice()