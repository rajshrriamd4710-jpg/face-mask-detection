import streamlit as st
import numpy as np
import cv2
from tensorflow import keras
from PIL import Image

# Page settings
st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")

st.markdown("""
<style>

.main-title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#2E86C1;
}

.sub-text{
    text-align:center;
    font-size:18px;
    color:gray;
}

.result-box{
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:22px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = keras.models.load_model("face_mask_model.h5")

# Title
st.markdown('<p class="main-title">😷 Face Mask Detection AI</p>', unsafe_allow_html=True)

st.markdown('<p class="sub-text">Upload an image and our AI will detect whether a person is wearing a mask.</p>', unsafe_allow_html=True)

st.write("")

# Upload Section
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    col1, col2, col3 = st.columns([1,2,1])

    image = Image.open(uploaded_file)

    with col2:
        st.image(image, caption="Uploaded Image", width=300)

    img = np.array(image)

    img_resized = cv2.resize(img, (128,128))
    img_scaled = img_resized / 255.0
    img_reshaped = np.reshape(img_scaled, (1,128,128,3))

    prediction = model.predict(img_reshaped)

    class_index = np.argmax(prediction)

    st.write("")

    if class_index == 0:
        st.success("😷 Mask Detected")
    else:
        st.error("❌ No Mask Detected")
