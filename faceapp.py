import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load model LBPH
model = cv2.face.LBPHFaceRecognizer_create()
model.read("face_model.yml")

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Label mapping (samakan dengan training)
label_map = np.load("label_map.npy", allow_pickle=True).item()

st.title("Pengenalan Wajah Sederhana")
st.write("Silahkan Upload Gambar Untuk Mengenali Orang")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        label, confidence = model.predict(face)

        name = label_map.get(label, "Unknown")

        # Draw bounding box & label
        cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            img_np,
            f"{name} ({confidence:.2f})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    st.image(img_np, caption="Hasil Pengenalan", use_container_width=True)

    if len(faces) == 0:
        st.warning("Tidak Ada Wajah Yang Dideteksi")
