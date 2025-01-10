# Real-Time Face Recognition System

This project is a real-time face recognition system that uses a webcam to capture face embeddings, stores them in a database, and recognizes faces during live video feed. The system is built using OpenCV, NumPy, and the DeepFace library for deep learning-based face representation.

---

## Features

- **Real-Time Face Embedding Capture**: Allows users to save face embeddings with a specific name for recognition later.
- **Face Recognition**: Matches live faces with saved embeddings and displays the identified names in real-time.
- **Interactive Modes**:
  - Add new face embeddings to the database.
  - Perform live face recognition.
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
   Upon running, choose between:
   - Extracting embeddings of a new face (`New`)
   - Performing attendance/recognition (`Attendance`)

2. **Face Embedding Extraction**:
   - Capture a face using the webcam by pressing `S`.
   - Provide a name for the face.
   - Save embeddings for future recognition.

3. **Face Recognition**:
   - Detects faces in real-time using the webcam.
   - Matches detected faces with stored embeddings using cosine similarity.
   - Displays the matched name or "Unknown" for unmatched faces.

4. **Quit Program**:
   - Press `Q` at any point to exit.

---

## File Structure

- **`FaceRecog.py`**: The main Python script for face embedding extraction and recognition.
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
   - Recognize faces in real-time.

---

## Key Dependencies

- **OpenCV**: For video capture and face detection.
- **DeepFace**: For deep learning-based face embedding extraction.
- **NumPy**: For embedding similarity computation.

---

## Future Improvements

- Add support for more face recognition models.
- Optimize performance for large embedding databases.
- Enhance GUI for easier interaction.

---

## License

This project is open source and available under the [MIT License](LICENSE).

