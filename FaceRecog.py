import cv2
import numpy as np
import pickle
from deepface import DeepFace
import os

# Output file for embeddings
EMBEDDINGS_FILE = 'KF.pkl'

# Initialize video capture
video_capture = cv2.VideoCapture(0)


def cosine_similarity_vectorized(embedding, known_embeddings):
    """
    Calculate cosine similarity between a given embedding and known embeddings.
    """
    dot_product = np.dot(known_embeddings, embedding)
    norms = np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding)
    return dot_product / norms


def load_embeddings():
    """
    Load existing embeddings and names from the file.
    """
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return [], []


def save_embeddings(known_face_encodings, known_face_names):
    """
    Save updated embeddings and names to the file.
    """
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)


def encodings():
    """
    Capture and save face embeddings.
    """
    known_face_encodings, known_face_names = load_embeddings()

    # Ensure the directory for saving photos exists
    if not os.path.exists("StudentPhotos"):
        os.makedirs("StudentPhotos")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        # Display the video feed
        cv2.imshow('Real-Time Face Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save face embedding
            name = input("Enter the name for the captured face: ").strip()
            if not name:
                print("Name cannot be empty. Try again.")
                continue

            # Save the captured image
            image_path = os.path.join("StudentPhotos", f"{name}.jpg")
            cv2.imwrite(image_path, frame)

            # Extract face embedding
            try:
                embedding = DeepFace.represent(img_path=image_path, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
                known_face_encodings.append(embedding)
                known_face_names.append(name)
                save_embeddings(known_face_encodings, known_face_names)
                print(f"Face of '{name}' saved successfully!")
                print("Starting real-time recognition...")
                recognition()  # Call recognition after saving the face
            except Exception as e:
                print(f"Error extracting embedding: {e}")
        elif key == ord('q'):  # Quit the program
            print("Exiting...")
            break


def recognition():
    """
    Perform real-time face recognition.
    """
    known_face_encodings, known_face_names = load_embeddings()

    if not known_face_encodings:
        print("No embeddings found. Please add new faces first.")
        return

    known_face_encodings = np.array(known_face_encodings)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frame_count = 0
    threshold = 0.5  # Similarity threshold

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        frame_count += 1
        if frame_count % 5 != 0:  # Skip frames for efficiency
            continue

        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_image = frame[y:y + h, x:x + w]
            face_image = cv2.resize(face_image, (224, 224))
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

            try:
                # Get emotion recognition using DeepFace
                emotion_result = DeepFace.analyze(img_path=face_image, actions=['emotion'], enforce_detection=False)
                dominant_emotion = emotion_result[0]['dominant_emotion']
                real_time_embedding = DeepFace.represent(img_path=face_image, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
                similarities = cosine_similarity_vectorized(real_time_embedding, known_face_encodings)
                best_match_index = np.argmax(similarities)
                best_match_similarity = similarities[best_match_index]

                name = known_face_names[best_match_index] if best_match_similarity > threshold else "Unknown"

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({dominant_emotion})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            except Exception as e:
                print(f"Error during recognition: {e}")

        cv2.imshow('Real-Time Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break


if __name__ == "__main__":
    try:
        mode = input("Choose mode (New/Attendance): ").strip().lower()
        if mode == "new":
            encodings()
        elif mode == "attendance":
            recognition()
        else:
            print("Invalid input. Please choose 'New' or 'Attendance'.")
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
