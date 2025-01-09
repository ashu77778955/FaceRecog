# FaceRecog
Face_Encodings.py:

Captures live images from a webcam.
Saves face embeddings (numerical representations of facial features) and associated names to a .pkl file (KF.pkl).
FaceRecog.py:

Recognizes faces in real-time using the embeddings saved in KF.pkl.
Uses a similarity threshold to identify known faces or label them as "Unknown."
Modules Installation
To ensure the scripts work correctly, install the required Python modules:

DeepFace:

bash
Copy code
pip install deepface
OpenCV:

bash
Copy code
pip install opencv-python
pip install opencv-contrib-python
NumPy:

bash
Copy code
pip install numpy
Pickle: No need to install this, as it is part of Python's standard library.

Procedure
1. Setting Up Face Embedding Script (Face_Encodings.py)
Run Face_Encodings.py.
Follow these steps:
Press s to save a face.
Enter the person's name when prompted.
The script will capture the face, compute the embedding, and save it to KF.pkl.
Press q to quit the script.
2. Running Face Recognition (FaceRecog.py)
Ensure KF.pkl contains the embeddings from Face_Encodings.py.
Run FaceRecog.py.
The script will:
Capture frames from the webcam.
Detect faces and compare them to stored embeddings.
Display the recognized name or "Unknown" on the live video feed.
Press q to quit.
Troubleshooting
If cv2.imshow fails to work, ensure OpenCV is installed with GUI support (opencv-python) and your environment supports window display.
For real-time speed, the recognition script skips every 4 out of 5 frames (frame_count % 5).
