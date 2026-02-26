import streamlit as st
import numpy as np
import json
import os
from PIL import Image

st.set_page_config(
    page_title="AI Sports Vision",
    page_icon="⚽",
    layout="wide"
)

st.title("🧠⚽ AI Sports Image Classifier")

# ===============================
# Paths (ถ้าไฟล์อยู่โฟลเดอร์เดียวกัน)
# ===============================
model_path = "nn_model.keras"
json_path = "class_indices.json"

# ===============================
# Load Model (โหลดครั้งเดียวจริง ๆ)
# ===============================
@st.cache_resource(show_spinner="🤖 Loading AI model...")
def load_model_safe(path):
    import tensorflow as tf   # 👉 import ตรงนี้แทน (ลด initial lag)
    try:
        if not os.path.exists(path):
            st.error(f"❌ Model file not found: {path}")
            return None
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        return None

# ===============================
# Load Classes
# ===============================
@st.cache_resource
def load_classes(path):
    try:
        if not os.path.exists(path):
            st.error(f"❌ class_indices.json not found: {path}")
            return None
        with open(path, "r") as f:
            class_indices = json.load(f)

        class_names = [None] * len(class_indices)
        for class_name, index in class_indices.items():
            class_names[index] = class_name

        return class_names
    except Exception as e:
        st.error(f"❌ JSON Load Error: {e}")
        return None

class_names = load_classes(json_path)

# ===============================
# Upload
# ===============================
uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    # โหลด model ตอนมีรูปจริง ๆ เท่านั้น
    model = load_model_safe(model_path)

    if model is None or class_names is None:
        st.stop()

    from tensorflow.keras.applications.efficientnet import preprocess_input

    image = Image.open(uploaded_file).convert("RGB")
    image_resized = image.resize((224, 224))

    img_array = np.array(image_resized)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🤖 AI is analyzing the image..."):
        prediction = model.predict(img_array)

    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(prediction)) * 100

    # แสดงรูปกลางจอ
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption="Uploaded Image", width=350)

    st.markdown("---")

    st.success(f"🎯 Predicted Sport: {predicted_class}")

    if confidence < 60:
        st.warning("⚠ Model is not confident about this prediction.")

    st.info(f"Confidence: {confidence:.2f}%")

    st.markdown("### 🔎 Top 3 Predictions")
    top3 = np.argsort(prediction[0])[-3:][::-1]
    for i in top3:
        st.write(f"{class_names[i]} — {prediction[0][i]*100:.2f}%")