import streamlit as st

st.set_page_config(
    page_title="Sports / Poker AI Project",
    page_icon="🏆",
    layout="wide"
)

st.title("🏆 Sports / Poker AI Project")
st.markdown("---")

st.header("🎯 Project Objective")

st.markdown("""
This project compares two types of Artificial Intelligence models:

- Machine Learning (Ensemble Model)
- Neural Network (Deep Learning)

The objective is to demonstrate how different AI approaches
handle structured and unstructured datasets.
""")

st.markdown("---")

st.header("📂 Datasets Used")

st.markdown("""
### 🎴 Poker Dataset
- Type: Structured Data
- Source: Generated using ChatGPT
- Task: Predict winning player

### 🏟 Sports Image Dataset
- Type: Unstructured Data (Images)
- Source: Kaggle
- Task: Classify sports category
""")

st.markdown("---")

st.header("🧠 Models Implemented")

st.markdown("""
### 1️⃣ Ensemble Machine Learning
- Logistic Regression
- Random Forest
- XGBoost

### 2️⃣ Neural Network
- EfficientNetB0 (Transfer Learning)
""")

st.markdown("---")

st.header("⚖ Model Comparison")

st.markdown("""
| Model Type | Data Type | Strength |
|------------|-----------|----------|
| Ensemble ML | Structured | Fast, stable, interpretable |
| Neural Network | Image | High accuracy, feature extraction |
""")

st.markdown("---")

st.success("🚀 Use the sidebar to explore model explanations and testing pages.")