import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# ===== Load Dataset =====
df = pd.read_csv("poker_dataset100k.csv")

# ===== Create 52-card mapping =====
ranks = "23456789TJQKA"
suits = "SHDC"
deck = [r+s for r in ranks for s in suits]
card_to_index = {card:i for i,card in enumerate(deck)}

def encode_row(row):
    vec = np.zeros(52)
    cards = row[:10]   # 6 player cards + 4 board (Turn stage)
    for c in cards:
        if c in card_to_index:
            vec[card_to_index[c]] = 1
    return vec

# ===== Encode All Rows =====
X = np.array([encode_row(row) for row in df.values])

# ===== Create Label =====
y = df[["P1_prob","P2_prob","P3_prob"]].idxmax(axis=1)
y = y.map({"P1_prob":0,"P2_prob":1,"P3_prob":2})

# ===== Train/Test Split =====
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# ===== Train Models =====
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train,y_train)

rf_model = RandomForestClassifier(
    n_estimators=50,        # ลดจาก 100 → 50
    max_depth=8,            # ลดความลึก
    min_samples_leaf=10,    # บังคับใบใหญ่ขึ้น
    max_features="sqrt",    # จำกัด feature split
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train,y_train)

xgb_model = XGBClassifier(eval_metric="mlogloss")
xgb_model.fit(X_train,y_train)

# ===== Evaluate =====
print("Logistic Accuracy:", accuracy_score(y_test,log_model.predict(X_test)))
print("RF Accuracy:", accuracy_score(y_test,rf_model.predict(X_test)))
print("XGB Accuracy:", accuracy_score(y_test,xgb_model.predict(X_test)))

# ===== Save =====
joblib.dump(log_model,"log_model.pkl")
joblib.dump(rf_model,"rf_model.pkl")
joblib.dump(xgb_model,"xgb_model.pkl")

print("Models saved successfully!")