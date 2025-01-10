# Real-Time Face Recognition and Emotion Detection System

This project is a real-time face recognition and emotion detection system that uses a webcam to capture face embeddings, stores them in a database, and recognizes faces while identifying emotions during a live video feed. It is built using OpenCV, NumPy, and the DeepFace library.

---

## Features

- **Real-Time Face Embedding Capture**: Allows users to save face embeddings with associated names for future recognition.
- **Face Recognition**: Matches live faces with stored embeddings and displays the identified names in real-time.
- **Emotion Detection**: Detects and displays the dominant emotion of the recognized faces.
- **Interactive Modes**:
  - Add new face embeddings to the database.
  - Perform live face recognition with emotion detection.
- **Face Embedding Storage**: All embeddings and their associated names are stored in a `.pkl` file for future use.

---

## Requirements

- Python 3.8 or higher
- Libraries:
  ```bash
  pip install opencv-python numpy deepface
  ```

---

## How It Works

1. **Initialize Program**:
   - Upon running, choose between:
     - Extracting embeddings of a new face (`New`)
     - Performing attendance/recognition (`Attendance`)

2. **Face Embedding Extraction**:
   - Capture a face using the webcam by pressing `S`.
   - Provide a name for the face.
   - Save embeddings for future recognition.

3. **Face Recognition and Emotion Detection**:
   - Detects faces in real-time using the webcam.
   - Matches detected faces with stored embeddings using cosine similarity.
   - Displays the recognized name and the dominant emotion of each face.

4. **Quit Program**:
   - Press `Q` at any point to exit.

---

## File Structure

- **`FaceRecog.py`**: The main Python script for face embedding extraction, recognition, and emotion detection.
- **`KF.pkl`**: The generated file to store face embeddings and names.

---

## Usage

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```

2. Run the script:
   ```bash
   python FaceRecog.py
   ```

3. Follow on-screen instructions to:
   - Add new faces.
   - Recognize faces in real-time with emotion detection.

---

## Key Dependencies

- **OpenCV**: For video capture and face detection.
- **DeepFace**: For deep learning-based face embedding extraction and emotion detection.
- **NumPy**: For embedding similarity computation.

---

## Future Improvements

- Add support for additional face recognition models.
- Optimize performance for larger databases of embeddings.
- Enhance the user interface for easier interaction.

---

## License

This project is open source and available under the [MIT License](LICENSE).

