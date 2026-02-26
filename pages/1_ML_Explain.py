import streamlit as st

st.set_page_config(
    page_title="ML Explain",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Model Explanation")
st.markdown("---")

# =====================================
# 1. Dataset Information
# =====================================
st.header("1️⃣ Dataset Information")

st.markdown("""
### 🎴 Poker Simulation Dataset

- **Type:** Structured Dataset  
- **Source:** Artificially generated using ChatGPT for simulation purposes  
- **Domain:** Texas Hold'em Poker  

The dataset was generated to simulate multiple poker game scenarios
including different player hands and community card combinations.

Each record contains:

- 2 cards per player (3 players total)
- Community cards (Flop, Turn, River)
- Target label: Winning player (Player 1, 2, or 3)

The dataset includes intentionally diverse and imperfect combinations
to simulate real-world variability.
""")

# =====================================
# 2. Feature Description
# =====================================
st.header("2️⃣ Feature Description")

st.markdown("""
Each card is encoded using One-Hot Encoding:

- Total features: 52 binary features
- Each index represents a unique card in the deck
- Value = 1 if card is present, 0 otherwise

This structured format allows compatibility with traditional ML models.
""")

# =====================================
# 3. Data Preparation
# =====================================
st.header("3️⃣ Data Preparation Process")

st.markdown("""
Steps performed:

1. Card indexing (0–51 mapping)
2. One-hot encoding transformation
3. Duplicate checking
4. Train-test splitting
5. Label encoding for classification
""")

# =====================================
# 4. Model Development
# =====================================
st.header("4️⃣ Ensemble Model Development")

st.markdown("""
This project uses an Ensemble Machine Learning approach:

- Logistic Regression
- Random Forest
- XGBoost

Final prediction is computed by averaging probabilities:

Final Probability = (P_log + P_rf + P_xgb) / 3

The player with the highest averaged probability is selected as the winner.
""")

# =====================================
# 5. Algorithm Theory
# =====================================
st.header("5️⃣ Algorithm Theory")

st.markdown("""
- **Logistic Regression:** Linear probabilistic classifier
- **Random Forest:** Ensemble of decision trees
- **XGBoost:** Gradient boosting framework

Ensemble learning improves model stability and reduces overfitting.
""")

# =====================================
# 6. References
# =====================================
st.header("6️⃣ References")

st.markdown("""
- Dataset generated using ChatGPT (OpenAI)  
- Scikit-learn Documentation  
- XGBoost Documentation  
""")

st.success("✅ This ML system demonstrates ensemble learning on structured data.")