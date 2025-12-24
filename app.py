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
import os

model_object = joblib.load("svm_multiclass.pkl")
encoder_object = joblib.load("label_encoder.pkl")

# Check if face detection models exist
face_models_available = os.path.exists("svm_multiclass_wajah.pkl") and os.path.exists("label_encoder_wajah.pkl")
if face_models_available:
    model_face = joblib.load("svm_multiclass_wajah.pkl")
    encoder_face = joblib.load("label_encoder_wajah.pkl")
else:
    model_face = None
    encoder_face = None


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
        <h2 style="margin:0;">🚀 ML Detection</h2>
        <p style="margin:4px 0 0 0;">Powerful image classification powered by Support Vector Machine</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Section Selector
if face_models_available:
    tab1, tab2, tab3 = st.tabs(["🎯 Vehicle Detection", "👤 Face Detection - Kuantum Peps", "📹 Webcam Face Recognition"])
else:
    tab1 = st.container()
    tab2 = None
    tab3 = None

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

if face_models_available and tab2 is not None:
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

if face_models_available and tab3 is not None:
    with tab3:
        with st.sidebar:
            st.header("Webcam Face Recognition")
            st.write("""
            1) Press START to activate webcam.
            2) The model detects faces in real-time.
            3) Recognized faces are labeled with confidence.
            4) Press STOP to end the session.
            """)
            st.divider()
            st.subheader("Available Persons")
            st.write(" • ".join(label_map.values()) if 'label_map' in globals() else "No face model available")

        st.markdown("### 📹 Real-time Webcam Face Recognition (LBPH)")
        
        # Load face model for webcam
        if os.path.exists("face_model.yml") and os.path.exists("label_map.npy"):
            model_webcam = cv2.face.LBPHFaceRecognizer_create()
            model_webcam.read("face_model.yml")
            label_map_webcam = np.load("label_map.npy", allow_pickle=True).item()
            
            # Haar Cascade
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )

            url = st.text_input("Enter URL for IP Camera (leave blank for local webcam):", "")
            col1, col2 = st.columns(2)
            
            with col1:
                start = st.button("▶️ START Webcam")
            with col2:
                stop = st.button("⏹️ STOP Webcam")

            frame_placeholder = st.empty()
            status_placeholder = st.empty()

            # Session state to control camera
            if "run_camera" not in st.session_state:
                st.session_state.run_camera = False

            if start:
                st.session_state.run_camera = True
            if stop:
                st.session_state.run_camera = False

            cap = None

            if st.session_state.run_camera:
                status_placeholder.info("🔴 Webcam is ACTIVE - Press STOP to end")
                try:
                    cap = cv2.VideoCapture(url if url else 0)
                    
                    if not cap.isOpened():
                        status_placeholder.error("❌ Cannot access webcam/IP camera. Check connection.")
                    else:
                        while st.session_state.run_camera:
                            ret, frame = cap.read()
                            if not ret:
                                status_placeholder.warning("⚠️ Cannot read frame from camera.")
                                break

                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                            for (x, y, w, h) in faces:
                                face = gray[y:y+h, x:x+w]
                                label, confidence = model_webcam.predict(face)
                                name = label_map_webcam.get(label, "Unknown")

                                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                                cv2.putText(
                                    frame,
                                    f"{name} ({confidence:.1f})",
                                    (x, y-10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2
                                )

                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                except Exception as e:
                    status_placeholder.error(f"❌ Error: {str(e)}")
                finally:
                    if cap is not None:
                        cap.release()
                    cv2.destroyAllWindows()
                    status_placeholder.success("✅ Webcam stopped")
            else:
                status_placeholder.info("⚪ Webcam is INACTIVE - Press START to begin")
        else:
            st.warning("⚠️ Face model files (face_model.yml or label_map.npy) not found. Please train the model first.")