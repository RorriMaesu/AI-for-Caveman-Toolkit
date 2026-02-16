# 03_vram_estimator.py
# Grug say: usage > capacity = crash.
# This script guesses if a model fits in your VRAM.

def estimate_vram(params_billions, quant_level):
    # Rough math:
    # FP16 (Raw) = 2 bytes per param
    # Q8 = 1 byte per param
    # Q4 = 0.7 bytes per param (roughly, with overhead)
    
    if quant_level == "FP16":
        size_gb = params_billions * 2
    elif quant_level == "Q8":
        size_gb = params_billions * 1
    elif quant_level == "Q4":
        size_gb = params_billions * 0.7
    else:
        return 0

    # Add 20% overhead for context window (KV Cache)
    total_needed = size_gb * 1.2
    return total_needed

if __name__ == "__main__":
    print("--- Grug's VRAM Guesser ---")
    p = float(input("How many Billion params? (e.g. 8, 14, 70): "))
    q = input("What Quant? (FP16, Q8, Q4): ")
    
    needed = estimate_vram(p, q)
    print(f"\nYou need roughly {needed:.1f} GB of VRAM.")