import cv2
import os
import pickle
from deepface import DeepFace

# Lists to store face encodings and corresponding names
known_face_encodings = []
known_face_names = []

# Output file for embeddings
pkl_file = 'KF.pkl'

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load existing embeddings if the file already exists
if os.path.exists(pkl_file):
    with open(pkl_file, 'rb') as f:
        known_face_encodings, known_face_names = pickle.load(f)

# Initialize webcam
cap = cv2.VideoCapture(0)

print("Press 's' to save face and 'q' to quit.")

while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Display the video feed
    cv2.imshow('Real-Time Face Capture', frame)
                    
    # Wait for user input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Save face embedding
        name = input("Enter the name for the captured face: ")

        # Save the captured image to a temporary file
        temp_image_path = "temp_image.jpg"
        cv2.imwrite(temp_image_path, frame)

        # Extract face embedding
        try:
            embedding = DeepFace.represent(img_path=temp_image_path, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]

            # Append the face embedding and the corresponding name
            known_face_encodings.append(embedding)
            known_face_names.append(name)

            # Save the updated embeddings to the .pkl file
            with open(pkl_file, 'wb') as f:
                pickle.dump((known_face_encodings, known_face_names), f)

            print(f"Face of '{name}' saved successfully!")

        except Exception as e:
            print(f"Error extracting embedding: {e}")

        # Remove the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    elif key == ord('q'):  # Quit the program
        print("Exiting...")
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
