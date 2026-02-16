# 05_loop_perfect.py
# Grug say: Circle must be perfect. End is Start.
import os
import numpy as np
from PIL import Image

VIDEO_FRAMES_DIR = "output_frames"

def calculate_mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def check_loop():
    if not os.path.exists(VIDEO_FRAMES_DIR):
        print(f"Error: Folder '{VIDEO_FRAMES_DIR}' not found.")
        return

    frames = sorted([f for f in os.listdir(VIDEO_FRAMES_DIR) if f.endswith('.png')])
    if len(frames) < 2:
        print("Not enough frames to check loop.")
        return

    first_frame_path = os.path.join(VIDEO_FRAMES_DIR, frames[0])
    last_frame_path = os.path.join(VIDEO_FRAMES_DIR, frames[-1])
    print(f"Comparing Start: {frames[0]} and End: {frames[-1]}")

    img1 = np.array(Image.open(first_frame_path).convert('L'))
    img2 = np.array(Image.open(last_frame_path).convert('L'))

    mse = calculate_mse(img1, img2)
    print(f"\nDifference Score (MSE): {mse:.2f}")

    if mse < 500:
        print("RESULT: PASS. The loop is seamless.")
    else:
        print("RESULT: FAIL. The jump is too visible. Regenerate.")

if __name__ == "__main__":
    check_loop()