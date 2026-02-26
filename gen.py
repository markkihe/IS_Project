import random
import itertools
import pandas as pd
import numpy as np

# ====== CONFIG ======
N_SAMPLES = 100000
MODE = "river"   # "river" หรือ "turn"

# ====== DECK ======
ranks = "23456789TJQKA"
suits = "SHDC"
deck_template = [r+s for r in ranks for s in suits]
rank_value = {r:i for i,r in enumerate(ranks)}

# ====== 5-card evaluator ======
def evaluate_5(cards):
    vals = sorted([rank_value[c[0]] for c in cards], reverse=True)
    suits_only = [c[1] for c in cards]

    # Straight (รองรับ wheel A-2-3-4-5)
    unique_vals = sorted(set(vals), reverse=True)
    is_straight = False
    if len(unique_vals) == 5:
        if unique_vals[0] - unique_vals[4] == 4:
            is_straight = True
        if set(unique_vals) == {12,3,2,1,0}:
            is_straight = True
            vals = [3,2,1,0,-1]

    is_flush = len(set(suits_only)) == 1

    counts = {v:vals.count(v) for v in set(vals)}
    count_sorted = sorted(counts.values(), reverse=True)

    if is_flush and is_straight: return (8, vals)
    if 4 in count_sorted: return (7, vals)
    if 3 in count_sorted and 2 in count_sorted: return (6, vals)
    if is_flush: return (5, vals)
    if is_straight: return (4, vals)
    if 3 in count_sorted: return (3, vals)
    if count_sorted.count(2) == 2: return (2, vals)
    if 2 in count_sorted: return (1, vals)
    return (0, vals)

# ====== 7-card evaluator ======
def evaluate_7(cards7):
    best = None
    for combo in itertools.combinations(cards7,5):
        score = evaluate_5(combo)
        if best is None or score > best:
            best = score
    return best

# ====== Exact Turn Probability ======
def exact_turn_prob(p1,p2,p3,board4):
    remaining = [c for c in deck_template if c not in p1+p2+p3+board4]
    wins = [0,0,0]
    total = 0

    for river in remaining:
        board = board4 + [river]
        scores = [
            evaluate_7(p1+board),
            evaluate_7(p2+board),
            evaluate_7(p3+board)
        ]
        max_s = max(scores)
        winners = [i for i,s in enumerate(scores) if s==max_s]
        for w in winners:
            wins[w] += 1/len(winners)
        total += 1

    return [w/total for w in wins]

# ====== Generate One Game ======
def generate_game():
    deck = deck_template.copy()
    random.shuffle(deck)

    p1 = deck[0:2]
    p2 = deck[2:4]
    p3 = deck[4:6]

    if MODE == "river":
        board = deck[6:11]
        scores = [
            evaluate_7(p1+board),
            evaluate_7(p2+board),
            evaluate_7(p3+board)
        ]
        max_s = max(scores)
        winners = [i for i,s in enumerate(scores) if s==max_s]
        probs = [0,0,0]
        for w in winners:
            probs[w] = 1/len(winners)

    else:  # TURN MODE
        board4 = deck[6:10]
        probs = exact_turn_prob(p1,p2,p3,board4)
        board = board4 + ["NA"]

    return p1+p2+p3+board, probs

# ====== Create Dataset ======
data = []
for i in range(N_SAMPLES):
    cards, probs = generate_game()
    data.append(cards+probs)

columns = [
    "P1_1","P1_2","P2_1","P2_2","P3_1","P3_2",
    "F1","F2","F3","Turn","River",
    "P1_prob","P2_prob","P3_prob"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("poker_dataset100k.csv", index=False)

print("Dataset Done!")