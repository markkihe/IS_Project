import streamlit as st

st.set_page_config(
    page_title="NN Explain",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Neural Network Model Explanation")
st.markdown("---")

# =====================================
# 1. Dataset Information
# =====================================
st.header("1️⃣ Dataset Information")

st.markdown("""
### 🏟 Sports Image Classification Dataset

- **Type:** Unstructured Dataset (Image Data)
- **Source:** Kaggle
- **Link:** https://www.kaggle.com/datasets/gpiosenka/sports-classification

The dataset contains multiple sports categories
with labeled images for supervised learning.

Each class contains several hundred images.
""")

# =====================================
# 2. Data Preparation
# =====================================
st.header("2️⃣ Data Preparation Process")

st.markdown("""
Steps performed:

1. Image resizing to 224x224 pixels
2. RGB conversion
3. Normalization using EfficientNet preprocess_input
4. Train-validation split
5. Optional data augmentation
""")

# =====================================
# 3. Model Architecture
# =====================================
st.header("3️⃣ Model Architecture")

st.markdown("""
The model is based on EfficientNetB0 (Pretrained on ImageNet).

Architecture:

- EfficientNetB0 base model
- Global Average Pooling
- Dense layer with Softmax activation

Transfer learning was applied to leverage pretrained features.
""")

# =====================================
# 4. Neural Network Theory
# =====================================
st.header("4️⃣ Neural Network Theory")

st.markdown("""
Convolutional Neural Networks (CNN) extract spatial features
from image data using convolution and pooling layers.

Softmax activation is used for multi-class classification.
""")

# =====================================
# 5. References
# =====================================
st.header("5️⃣ References")

st.markdown("""
- Kaggle Sports Classification Dataset  
  https://www.kaggle.com/datasets/gpiosenka/sports-classification  
- TensorFlow Documentation  
- EfficientNet Paper (Tan & Le, 2019)
""")

st.success("🚀 This Neural Network model applies transfer learning for image classification.")