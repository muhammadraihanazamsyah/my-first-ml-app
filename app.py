import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from skimage.feature import hog

# Load the trained model and label encoder
model = joblib.load("svm_multiclass.pkl")
encoder = joblib.load("label_encoder.pkl")

# Streamlit app
st.title("Image Classification with SVM")
st.write("Upload an image to classify it.")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# process and predict the uploaded image
if uploaded_file is not None:
    # image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    # image = image.resize((128, 128))
    # img_array = np.array(image)
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (128, 128))

    # Extract HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2)
    ).reshape(1, -1)

    # Predict the class
    prediction = model.predict(features)
    predicted_label = encoder.inverse_transform(prediction)[0]

    # Display the result
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Class: **{predicted_label}**")
    
if uploaded_file is not None:
    # === TAMPILKAN GAMBAR ===
    st.image(
        image,
        caption="Uploaded Image",
        use_column_width=True
    )

    # === OUTPUT ===
    probabilities = model.predict_proba(features)[0]
    confidence = np.max(probabilities) * 100
    
    st.subheader("Prediction Result")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Predicted Class**")
        st.success(predicted_label)
    with col2:
        st.markdown("**Confidence**")
        st.info(f"{confidence:.2f}%")

    st.subheader("Confidence per Class")
    for class_name, prob in zip(encoder.classes_, probabilities):
        st.write(f"- **{class_name}** : {prob*100:.2f}%")