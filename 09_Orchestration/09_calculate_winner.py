# 09_calculate_winner.py
# Grug say: Who is strongest God? Math decides.
import json
import os

def create_dummy_data():
    if not os.path.exists('ch09_scores_A.json'):
        with open('ch09_scores_A.json', 'w') as f: json.dump({"clarity": 3, "accuracy": 4}, f)
    if not os.path.exists('ch09_scores_B.json'):
        with open('ch09_scores_B.json', 'w') as f: json.dump({"clarity": 5, "accuracy": 5}, f)

def calculate_score(filename):
    try:
        with open(filename, 'r') as f:
            scores = json.load(f)
        return sum(scores.values())
    except FileNotFoundError:
        print(f"Error: Could not find {filename}")
        return 0

if __name__ == "__main__":
    create_dummy_data()
    score_A = calculate_score('ch09_scores_A.json')
    score_B = calculate_score('ch09_scores_B.json')
    print(f"Oracle A: {score_A} | Oracle B: {score_B}")
    if score_A > score_B: print("Winner: Oracle A")
    elif score_B > score_A: print("Winner: Oracle B")
    else: print("It's a tie!")