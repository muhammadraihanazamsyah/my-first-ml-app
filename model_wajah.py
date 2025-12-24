import cv2
import numpy as np
import os

# Load Haar Cascade untuk deteksi wajah
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

dataset_path = "dataset_wajah"
labels = []
faces = []
label_map = {}
current_label = 0

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_path):
        continue

    label_map[current_label] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        detected_faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            face = img[y:y+h, x:x+w]
            faces.append(face)
            labels.append(current_label)

    current_label += 1

labels = np.array(labels)

# Buat dan latih model LBPH
model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

# Simpan model
model.save("face_model.yml")
np.save("label_map.npy", label_map)
print("Training selesai. Model disimpan sebagai face_model.yml")
print("Label mapping:", label_map)
