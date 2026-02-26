import streamlit as st
import numpy as np
import itertools
import joblib

st.set_page_config(page_title="Poker AI", page_icon="♠", layout="wide")

st.markdown("# ♠ Poker 3 Players AI System")

@st.cache_resource
def load_models():
    log_model = joblib.load("log_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    xgb_model = joblib.load("xgb_model.pkl")
    return log_model, rf_model, xgb_model

log_model, rf_model, xgb_model = load_models()

# =========================
# ===== Deck Setup ========
# =========================
ranks = "23456789TJQKA"
suits = "SHDC"

deck = [r+s for r in ranks for s in suits]
rank_value = {r:i for i,r in enumerate(ranks)}
card_to_index = {card:i for i,card in enumerate(deck)}

suit_symbol = {"S": "♠","H": "♥","D": "♦","C": "♣"}

def display_card(card):
    return f"{card[0]}{suit_symbol[card[1]]}"

display_deck = {display_card(c): c for c in deck}
display_options = list(display_deck.keys())

def encode_cards(cards):
    vec = np.zeros(52)
    for c in cards:
        vec[card_to_index[c]] = 1
    return vec.reshape(1,-1)

# =========================
# ===== Rule Engine =======
# =========================
def evaluate_5(cards):
    vals = sorted([rank_value[c[0]] for c in cards], reverse=True)
    suits_only = [c[1] for c in cards]

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

def evaluate_7(cards7):
    best = None
    for combo in itertools.combinations(cards7,5):
        score = evaluate_5(combo)
        if best is None or score > best:
            best = score
    return best

# =========================
# ===== UI Layout =========
# =========================
st.markdown("## 🎴 Players")
col1, col2, col3 = st.columns(3)

with col1:
    p1_1_disp = st.selectbox("P1 Card 1", display_options, key="p1_1")
    p1_2_disp = st.selectbox("P1 Card 2", display_options, key="p1_2")

with col2:
    p2_1_disp = st.selectbox("P2 Card 1", display_options, key="p2_1")
    p2_2_disp = st.selectbox("P2 Card 2", display_options, key="p2_2")

with col3:
    p3_1_disp = st.selectbox("P3 Card 1", display_options, key="p3_1")
    p3_2_disp = st.selectbox("P3 Card 2", display_options, key="p3_2")

st.markdown("## 🃏 Board")
b1,b2,b3,b4,b5 = st.columns(5)

with b1:
    f1_disp = st.selectbox("Flop 1", display_options, key="f1")
with b2:
    f2_disp = st.selectbox("Flop 2", display_options, key="f2")
with b3:
    f3_disp = st.selectbox("Flop 3", display_options, key="f3")
with b4:
    turn_disp = st.selectbox("Turn", display_options, key="turn")
with b5:
    river_disp = st.selectbox("River", ["🚫 Not Opened"]+display_options, key="river")

# Convert back
p1 = [display_deck[p1_1_disp], display_deck[p1_2_disp]]
p2 = [display_deck[p2_1_disp], display_deck[p2_2_disp]]
p3 = [display_deck[p3_1_disp], display_deck[p3_2_disp]]

board = [
    display_deck[f1_disp],
    display_deck[f2_disp],
    display_deck[f3_disp],
    display_deck[turn_disp]
]

river = None if river_disp=="🚫 Not Opened" else display_deck[river_disp]

# =========================
# ===== Prediction ========
# =========================
if st.button("🎯 Evaluate / Predict"):

    base_cards = p1+p2+p3+board
    all_cards = base_cards if river is None else base_cards+[river]

    if len(all_cards) != len(set(all_cards)):
        st.error("❌ ห้ามเลือกไพ่ซ้ำกัน")
    else:

        if river is not None:
            board_full = board+[river]
            scores = [
                evaluate_7(p1+board_full),
                evaluate_7(p2+board_full),
                evaluate_7(p3+board_full)
            ]
            max_score = max(scores)
            winners = [i for i,s in enumerate(scores) if s==max_score]
            probs = [0,0,0]
            for w in winners:
                probs[w]=1/len(winners)

        else:
            X = encode_cards(base_cards)

            p_log = log_model.predict_proba(X)[0]
            p_rf  = rf_model.predict_proba(X)[0]
            p_xgb = xgb_model.predict_proba(X)[0]

            probs = (p_log+p_rf+p_xgb)/3

        probs_percent = np.round(np.array(probs)*100,2)
        winner = np.argmax(probs_percent)

        st.markdown("## 🏆 Result")
        col1,col2,col3 = st.columns(3)

        for i,col in enumerate([col1,col2,col3]):
            with col:
                if i==winner:
                    st.success(f"🎉 Player {i+1}\n\n{probs_percent[i]} %")
                else:
                    st.write(f"Player {i+1}\n\n{probs_percent[i]} %")