# Face Recognition System

This repository contains two Python scripts designed for a simple face recognition system using the DeepFace library.

## Overview

1. **Face_Encodings.py**:
   - Captures live images from a webcam.
   - Extracts face embeddings (numerical representations of facial features) and saves them along with corresponding names to a `.pkl` file (`KF.pkl`).

2. **FaceRecog.py**:
   - Performs real-time face recognition using embeddings stored in `KF.pkl`.
   - Labels recognized faces or marks them as "Unknown" based on similarity.

---

## Installation

### Required Modules

Install the necessary Python modules using the following commands:

1. **DeepFace**:
   ```bash
   pip install deepface
   ```

2. **OpenCV**:
   ```bash
   pip install opencv-python
   pip install opencv-contrib-python
   ```

3. **NumPy**:
   ```bash
   pip install numpy
   ```

4. **Pickle**:
   No need to install this, as it is part of Python's standard library.

---

## Usage

### 1. Face Embedding Script (Face_Encodings.py)

This script allows you to capture images and save face embeddings.

#### Steps:
1. Run `Face_Encodings.py`:
   ```bash
   python Face_Encodings.py
   ```
2. Follow the instructions in the terminal:
   - **Press `s`**: Capture and save a face.
   - Enter the person's name when prompted.
   - The script will save the face embedding and name to `KF.pkl`.
   - **Press `q`**: Quit the script.

---

### 2. Face Recognition Script (FaceRecog.py)

This script performs real-time face recognition using the embeddings saved in `KF.pkl`.

#### Steps:
1. Ensure that `KF.pkl` exists and contains face embeddings.
2. Run `FaceRecog.py`:
   ```bash
   python FaceRecog.py
   ```
3. The script will:
   - Capture frames from the webcam.
   - Detect faces and compare them with stored embeddings.
   - Display recognized names or "Unknown" on the live video feed.
   - **Press `q`**: Quit the script.

---

## Troubleshooting

- **cv2.imshow not working**:
  - Ensure OpenCV is installed with GUI support (`opencv-python`).
  - Verify that your environment supports window display (e.g., avoid headless servers).

- **Performance Issues**:
  - The recognition script skips 4 out of 5 frames (`frame_count % 5`) to enhance speed. Adjust this value for better accuracy or performance.

---

## Future Improvements

- Add multi-threading for faster processing.
- Improve face detection by integrating advanced models like MTCNN.
- Support for multiple model backends (e.g., `Facenet`, `OpenFace`).

---

Feel free to contribute or open issues for further enhancements!

