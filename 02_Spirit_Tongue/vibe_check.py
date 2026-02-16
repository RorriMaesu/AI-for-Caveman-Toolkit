# 02_Spirit_Tongue/vibe_check.py
# Grug say: Machine speak many tongues. Python is sharpest spear.
# This script is the result of the "Vibe Check" challenge in Chapter 2.

import psutil
import platform
import os

def check_spirit_strength():
    print("--- 🗿 Grug's System Vibe Check 🗿 ---")
    
    # 1. Check the Rock (OS)
    print(f"Cave Wall (OS): {platform.system()} {platform.release()}")
    
    # 2. Check the Brain (CPU)
    cpu_freq = psutil.cpu_freq()
    print(f"Thinking Power (CPU): {psutil.cpu_percent(interval=1)}% utilized")
    if cpu_freq:
         print(f"  - Speed: {cpu_freq.current:.2f} Mhz")
    
    # 3. Check the Memory (RAM)
    mem = psutil.virtual_memory()
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    print(f"Head Space (RAM): {available_gb:.1f} GB free out of {total_gb:.1f} GB")

    if total_gb < 8:
        print("\nWARNING: Cave is small. AI might hit head.")
    else:
        print("\nGood. Cave is big enough for small spirits.")

if __name__ == "__main__":
    check_spirit_strength()