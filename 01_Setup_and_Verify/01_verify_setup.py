# 01_verify_setup.py
# Grug say: Run this to see if Spirit is awake.
import os

TARGET_FILE = 'ch01_hello_transcript.txt'

def check_cave():
    print(f"--- Grug Inspecting Cave for {TARGET_FILE} ---")
    
    if not os.path.exists(TARGET_FILE):
        print(f"FAIL: {TARGET_FILE} is missing.")
        print("Fix: Run your AI model and save the output first.")
        return

    file_size = os.path.getsize(TARGET_FILE)
    if file_size < 10:
        print(f"FAIL: {TARGET_FILE} is too small (only {file_size} bytes).")
        print("Fix: Did the AI actually answer you?")
        return

    print("PASS: File found and looks heavy with words.")
    print("Grug happy. You are ready for Chapter 2.")

if __name__ == "__main__":
    check_cave()