# 03_image_audit.py
# Grug say: Never lose the recipe for your mammoth.
import os
from PIL import Image 

def audit_images():
    print("--- Scanning Cave for Paintings (PNGs) ---")
    images = [f for f in os.listdir('.') if f.lower().endswith('.png')]
    
    if not images:
        print("No images found. Go make art first!")
        return

    for img_file in images:
        try:
            with Image.open(img_file) as img:
                width, height = img.size
                metadata = img.info
                print(f"\nChecking: {img_file}")
                print(f"  - Size: {width}x{height}")
                
                if metadata:
                    print("  - Metadata: FOUND (Good!)")
                else:
                    print("  - Metadata: MISSING (Bad!)")
        except Exception as e:
            print(f"  - Error reading {img_file}: {e}")

if __name__ == "__main__":
    audit_images()