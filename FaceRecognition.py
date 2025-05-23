import sys, os, pickle, csv
import cv2
import numpy as np
from datetime import datetime
from deepface import DeepFace
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QLineEdit,
    QVBoxLayout, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import ( QTimer, Qt )

# =================== Directory Setup ===================
DATA_DIR = "data"
PHOTOS_DIR = os.path.join(DATA_DIR, "photos")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings", "KF.pkl")
ATTENDANCE_DIR = os.path.join(DATA_DIR, "attendance")

os.makedirs(PHOTOS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

marked_names = {}  # {name: (roll, time)}

# =================== Helper ===================
def cosine_similarity_vectorized(embedding, known_embeddings):
    dot_product = np.dot(known_embeddings, embedding)
    norms = np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding)
    return dot_product / norms

def exit_program():
    print("Exiting the program...")
    sys.exit(0)

# =================== Main App ===================
class FaceRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("✨ Face Recognition Attendance System ✨")
        self.setStyleSheet("background-color: #1e1e1e; color: white;")
        self.video_capture = cv2.VideoCapture(0)
        self.known_encodings, self.known_names, self.known_rolls = self.load_embeddings()

        self.init_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.frame_count = 0
        self.mode = None
        self.threshold = 0.5

    def init_ui(self):
        self.setMinimumSize(1000, 700)

        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 400)
        self.video_label.setStyleSheet("border: 2px solid white;")

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter name")
        self.name_input.setFixedHeight(40)
        self.name_input.setStyleSheet("padding: 10px; font-size: 16px;")
        #self.name_input.setAlignment(Qt.AlignCenter)

        self.roll_input = QLineEdit()
        self.roll_input.setPlaceholderText("Enter roll number")
        self.roll_input.setFixedHeight(40)
        self.roll_input.setStyleSheet("padding: 10px; font-size: 16px;")
        self.roll_input.setAlignment(Qt.AlignCenter)

        form_layout = QHBoxLayout()
        form_layout.addWidget(self.name_input)
        form_layout.addWidget(self.roll_input)
        form_layout.setSpacing(20)

        add_button = QPushButton("➕ Add New Face")
        add_button.setStyleSheet("background-color: #007acc; padding: 10px; font-size: 16px;")
        add_button.setFixedHeight(45)
        add_button.clicked.connect(self.add_new_face_mode)

        attendance_button = QPushButton("✅ Start Attendance")
        attendance_button.setStyleSheet("background-color: #2d7d46; padding: 10px; font-size: 16px;")
        attendance_button.setFixedHeight(45)
        attendance_button.clicked.connect(self.attendance_mode)

        save_button = QPushButton("💾 Save Attendance")
        save_button.setStyleSheet("background-color: #ffc107; padding: 10px; font-size: 16px;")
        save_button.setFixedHeight(45)
        save_button.clicked.connect(self.save_attendance_clicked)

        button_layout = QHBoxLayout()
        button_layout.addWidget(add_button)
        button_layout.addWidget(attendance_button)
        button_layout.addWidget(save_button)
        button_layout.setSpacing(20)

        self.status_label = QLabel("👋 Welcome! Choose an action.")
        self.status_label.setFont(QFont("Arial", 16))
        self.status_label.setStyleSheet("padding: 15px;")

        vbox = QVBoxLayout()
        vbox.addWidget(self.video_label, alignment=Qt.AlignCenter)
        vbox.addLayout(form_layout)
        vbox.addLayout(button_layout)
        vbox.addWidget(self.status_label)
        vbox.setSpacing(20)
        vbox.setContentsMargins(40, 30, 40, 30)

        self.setLayout(vbox)

    def update_frame(self):
        ret, frame = self.video_capture.read()
        if not ret:
            return

        self.frame_count += 1
        if self.mode == "attendance" and self.frame_count % 5 == 0:
            self.recognize_faces(frame)
        elif self.mode == "add" and self.frame_count % 30 == 0:
            self.capture_face(frame)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img.shape
        bytes_per_line = ch * w
        qt_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.width(), self.video_label.height()))

    def add_new_face_mode(self):
        name = self.name_input.text().strip()
        roll = self.roll_input.text().strip()
        if not name or not roll:
            self.status_label.setText("⚠️ Please enter both name and roll number!")
            return
        self.mode = "add"
        self.status_label.setText(f"📸 Adding new face: {name} (Roll: {roll})")
        self.timer.start(30)

    def attendance_mode(self):
        if not self.known_encodings:
            self.status_label.setText("⚠️ No known faces. Add some first!")
            return
        self.mode = "attendance"
        self.status_label.setText("🧠 Starting attendance...")
        self.timer.start(30)

    def capture_face(self, frame):
        name = self.name_input.text().strip()
        roll = self.roll_input.text().strip()
        if not roll:
            self.status_label.setText("⚠️ Please enter a roll number!")
            return

        path = os.path.join(PHOTOS_DIR, f"{name}_{roll}.jpg")
        cv2.imwrite(path, frame)

        try:
            embedding = DeepFace.represent(img_path=path, model_name="VGG-Face", enforce_detection=False)[0]['embedding']
            self.known_encodings.append(embedding)
            self.known_names.append(name)
            self.known_rolls.append(roll)
            self.save_embeddings(self.known_encodings, self.known_names, self.known_rolls)
            self.status_label.setText(f"✅ Face of {name} (Roll: {roll}) saved!")
            self.timer.stop()
        except Exception as e:
            self.status_label.setText(f"❌ Error capturing face: {e}")

    def recognize_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            roi = frame[y:y+h, x:x+w]
            face_img = cv2.resize(roi, (224, 224))
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            try:
                emb = DeepFace.represent(img_path=face_rgb, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]
                sims = cosine_similarity_vectorized(emb, np.array(self.known_encodings))
                idx = np.argmax(sims)
                sim_score = sims[idx]
                if sim_score > self.threshold:
                    name = self.known_names[idx]
                    roll = self.known_rolls[idx]
                else:
                    name = "Unknown"
                    roll = None

                if name != "Unknown" and roll is not None:
                    self.mark_attendance(name, roll)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
            except Exception as e:
                print(f"Recognition error: {e}")

    def mark_attendance(self, name, roll):
        if name not in marked_names:
            now = datetime.now().strftime("%H:%M:%S")
            marked_names[name] = (roll, now)
            print(f"[ATTENDANCE] {name} (Roll: {roll}) marked at {now}")
            self.status_label.setText(f"🎉 {name} (Roll: {roll}) marked at {now}")

    def save_attendance_clicked(self):
        if not marked_names:
            self.status_label.setText("⚠️ No attendance data to save yet.")
            return
        self.save_attendance()
        self.status_label.setText("📁 Attendance saved successfully!")

    def save_attendance(self):
        filename = os.path.join(ATTENDANCE_DIR, f"Attendance_{datetime.now().strftime('%Y-%m-%d')}.csv")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Roll Number", "Name", "Status", "Time"])

            all_students = list(zip(self.known_rolls, self.known_names))
            present_students = {name: (roll, time) for name, (roll, time) in marked_names.items()}

            for roll, name in all_students:
                if name in present_students:
                    _, time = present_students[name]
                    writer.writerow([roll, name, "Present", time])
                else:
                    writer.writerow([roll, name, "Absent", ""])
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        # Create a quick map of marked students for lookup
        # present_students = {name: (roll, time) for name, (roll, time) in marked_names.items()}
        for roll, name in zip(self.known_rolls, self.known_names):
            student_filename = os.path.join(ATTENDANCE_DIR, f"{name}_{roll}.csv")
            file_exists = os.path.isfile(student_filename)
            with open(student_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Date", "Time", "Status"])
                if name in present_students:
                    writer.writerow([date_str, time_str, "Present"])
                    print(f"[SAVED] {name} marked Present in {student_filename}")
                else:
                    writer.writerow([date_str, time_str, "Absent"])
                    print(f"[SAVED] {name} marked Absent in {student_filename}")
        self.status_label.setText("✅ Attendance saved for all students!")
        print(f"[SAVED] Attendance -> {filename}")
        exit_program()

    def load_embeddings(self):
        if os.path.exists(EMBEDDINGS_FILE):
            with open(EMBEDDINGS_FILE, 'rb') as f:
                return pickle.load(f)
        return [], [], []

    def save_embeddings(self, encodings, names, rolls):
        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump((encodings, names, rolls), f)

    def closeEvent(self, event):
        self.video_capture.release()
        self.timer.stop()
        event.accept()

# =================== Run ===================
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.resize(1000, 700)  # Medium-sized window
    window.show()
    sys.exit(app.exec_())
