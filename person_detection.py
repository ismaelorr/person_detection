import cv2
import mediapipe as mp
from PyQt5 import QtCore, QtGui, QtWidgets
from deepface import DeepFace

#Code made by Ismaelorr

class VideoWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Estimación de Edad, Género y Detección de Manos")
        self.video_label = QtWidgets.QLabel(self)
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        # Initialize mediaPipe to hands detection
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize VideoWindow
        self.cap = cv2.VideoCapture(0)

        # Iniciar timer to update frames
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        # Convert frame in RGB for MeidaPipe and DeepFace
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            #Initialize DeepFace
            analysis = DeepFace.analyze(frame_rgb, actions=['age', 'gender'], enforce_detection=False, silent=True)
            age = analysis[0]["age"]
            gender = analysis[0]["dominant_gender"]
            label = f"Age: {age}, Gender: {gender}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        except Exception as e:
            print("Error in DeepFace:", e)

        # Detect hands
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw references in hanfs
                self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Count fingers in hands
                fingers_up = self.count_fingers(hand_landmarks)

                # Show number of fingers in each hands
                cv2.putText(frame, f"Mano {hand_index + 1}: {fingers_up} dedos levantados",
                            (10, 70 + hand_index * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QtGui.QImage(frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(q_image))

    def count_fingers(self, hand_landmarks):
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        return sum(fingers)

    def closeEvent(self, event):
        self.cap.release()

app = QtWidgets.QApplication([])
window = VideoWindow()
window.show()
app.exec_()
