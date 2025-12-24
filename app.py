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

# Load the trained models and label encoders
model_object = joblib.load("svm_multiclass.pkl")
encoder_object = joblib.load("label_encoder.pkl")
model_face = joblib.load("svm_multiclass_wajah.pkl")
encoder_face = joblib.load("label_encoder_wajah.pkl")


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
        <h2 style="margin:0;">🚀 Kuantum Peps ML Detection</h2>
        <p style="margin:4px 0 0 0;">Powerful image classification powered by Support Vector Machine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Section Selector
tab1, tab2 = st.tabs(["🎯 Vehicle Detection", "👤 Face Detection - Kuantum Peps"])

with tab1:
    with st.sidebar:
        st.header("How it works")
        st.write("""
        1) Upload a JPG or PNG image.
        2) The image is resized to 128x128 and converted to grayscale.
        3) HOG features are extracted and fed into an SVM model.
        4) The model outputs the predicted class with confidence.
        """)
        st.divider()
        st.subheader("Object Classes")
        st.write(" • ".join(encoder_object.classes_))

    uploader_col, info_col = st.columns([1.2, 1])
    with uploader_col:
        st.subheader("Vehicle Detection")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="object")
    with info_col:
        st.subheader("Model")
        st.write("Linear SVM trained on HOG features.")
        st.write("Optimized for quick predictions with lightweight preprocessing.")

    if uploaded_file is not None:
        image, features = preprocess_image(uploaded_file)
        prediction = model_object.predict(features)
        probabilities = model_object.predict_proba(features)[0]
        predicted_label = encoder_object.inverse_transform(prediction)[0]
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
            for class_name, prob in zip(encoder_object.classes_, probabilities):
                st.write(f"{class_name}: {prob*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div class=\"card\">Upload an image to see predictions.</div>", unsafe_allow_html=True)

with tab2:
    with st.sidebar:
        st.header("Face Detection")
        st.write("""
        1) Upload a face image.
        2) The image is processed using HOG features.
        3) SVM model identifies the person.
        4) Confidence score is displayed.
        """)
        st.divider()
        st.subheader("Recognized Faces")
        st.write(" • ".join(encoder_face.classes_))

    st.markdown("### 👤 Kuantum Peps Face Recognition")
    
    uploader_col2, info_col2 = st.columns([1.2, 1])
    with uploader_col2:
        st.subheader("Upload Face Image")
        uploaded_face = st.file_uploader("Choose a face image", type=["jpg", "jpeg", "png"], key="face")
    with info_col2:
        st.subheader("Model Info")
        st.write("Trained on Kuantum Peps team members.")
        st.write("Using HOG features + Linear SVM.")

    if uploaded_face is not None:
        image, features = preprocess_image(uploaded_face)
        prediction = model_face.predict(features)
        probabilities = model_face.predict_proba(features)[0]
        predicted_label = encoder_face.inverse_transform(prediction)[0]
        confidence = float(np.max(probabilities) * 100)

        img_col, result_col = st.columns([1.2, 1])
        with img_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Face Preview**")
            st.image(image, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with result_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Recognized Person**")
            st.success(f"👤 {predicted_label.upper()}")
            st.info(f"Confidence: {confidence:.2f}%")
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("**All Probabilities**")
            for class_name, prob in zip(encoder_face.classes_, probabilities):
                st.write(f"{class_name}: {prob*100:.2f}%")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div class=\"card\">Upload a face image to identify the person.</div>", unsafe_allow_html=True)