import cv2
import numpy as np
import pickle
from deepface import DeepFace

# Load embeddings and names
with open('KF.pkl', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Convert embeddings to NumPy array for vectorized computation
known_face_encodings = np.array(known_face_encodings)

def cosine_similarity_vectorized(embedding, known_embeddings):
    dot_product = np.dot(known_embeddings, embedding)
    norms = np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding)
    return dot_product / norms

# Initialize video capture
video_capture = cv2.VideoCapture(0)

frame_count = 0
threshold = 0.5  # Similarity threshold

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 != 0:  # Skip frames for speed
        continue

    # Detect faces using Haar Cascade
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') \
            .detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_image = frame[y:y + h, x:x + w]
        face_image = cv2.resize(face_image, (224, 224))
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        real_time_embedding = DeepFace.represent(img_path=face_image, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
        similarities = cosine_similarity_vectorized(real_time_embedding, known_face_encodings)
        best_match_index = np.argmax(similarities)
        best_match_similarity = similarities[best_match_index]

        name = known_face_names[best_match_index] if best_match_similarity > threshold else "Unknown"

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
