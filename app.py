import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog

st.set_page_config(
    page_title="Image Classification with SVM",
    page_icon="🖼️",
    layout="wide",
)

# Minimal styling to give the app a more polished look.
st.markdown(
    """
    <style>
        body {background: #0f172a; color: #e2e8f0;}
        .main {padding: 2rem 3rem;}
        .hero {
            background: linear-gradient(135deg, #0ea5e9, #22c55e);
            color: #0b1221;
            padding: 1.25rem 1.5rem;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 1rem;
        }
        .card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.25rem;
            box-shadow: 0 12px 24px rgba(0,0,0,0.24);
        }
        .metric {border-radius: 12px !important;}
        .prob-row {margin-bottom: 0.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the trained model and label encoder
model = joblib.load("svm_multiclass.pkl")
encoder = joblib.load("label_encoder.pkl")


def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128))
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
    ).reshape(1, -1)
    return image, features

st.markdown(
    """
    <div class="hero">
        <h2 style="margin:0;">Image Classification with SVM</h2>
        <p style="margin:4px 0 0 0;">Upload an image to see its predicted class and confidence.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("How it works")
    st.write("""
    1) Upload a JPG or PNG image.
    2) The image is resized to 128x128 and converted to grayscale.
    3) HOG features are extracted and fed into an SVM model.
    4) The model outputs the predicted class with confidence.
    """)
    st.divider()
    st.subheader("Classes")
    st.write(" • ".join(encoder.classes_))

uploader_col, info_col = st.columns([1.2, 1])
with uploader_col:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
with info_col:
    st.subheader("Model")
    st.write("Linear SVM trained on HOG features.")
    st.write("Optimized for quick predictions with lightweight preprocessing.")

if uploaded_file is not None:
    image, features = preprocess_image(uploaded_file)
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)[0]
    predicted_label = encoder.inverse_transform(prediction)[0]
    confidence = float(np.max(probabilities) * 100)

    img_col, result_col = st.columns([1.2, 1])
    with img_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Preview**")
        st.image(image, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with result_col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Prediction**")
        st.success(f"{predicted_label}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("**Confidence per Class**")
        for class_name, prob in zip(encoder.classes_, probabilities):
            st.write(f"{class_name}: {prob*100:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.markdown("<div class=\"card\">Upload an image to see predictions.</div>", unsafe_allow_html=True)