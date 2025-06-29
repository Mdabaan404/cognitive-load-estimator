import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math
from sklearn.linear_model import LogisticRegression
import joblib

# Initialize MediaPipe modules
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
pose = mp_pose.Pose()
hands = mp_hands.Hands()

# Webcam
cap = cv2.VideoCapture(0)

# Rolling buffers
blink_times = deque()
eye_positions = deque(maxlen=30)
frown_scores = deque(maxlen=30)

# Landmark indices
LEFT_IRIS = 468
RIGHT_IRIS = 473
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
LEFT_EYEBROW_UP = 105
LEFT_EYEBROW_LOW = 66
RIGHT_EYEBROW_UP = 334
RIGHT_EYEBROW_LOW = 296
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# Load ML model (pretrained for demonstration)
try:
    model = joblib.load("cognitive_load_model.pkl")
except:
    model = LogisticRegression()
    model.classes_ = np.array([0, 1])
    model.coef_ = np.array([[0.1]*5])
    model.intercept_ = np.array([0])

# Utility functions
def eye_aspect_ratio(landmarks, top, bottom, w, h):
    pt_top = np.array([landmarks[top].x * w, landmarks[top].y * h])
    pt_bottom = np.array([landmarks[bottom].x * w, landmarks[bottom].y * h])
    return np.linalg.norm(pt_top - pt_bottom)

def detect_blink(landmarks, w, h):
    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, w, h)
    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, w, h)
    ear = (left_ear + right_ear) / 2
    return ear < 4.0

def get_eye_center(landmarks):
    return (landmarks[LEFT_IRIS].x + landmarks[RIGHT_IRIS].x) / 2

def detect_frown(landmarks, w, h):
    left_diff = landmarks[LEFT_EYEBROW_UP].y - landmarks[LEFT_EYEBROW_LOW].y
    right_diff = landmarks[RIGHT_EYEBROW_UP].y - landmarks[RIGHT_EYEBROW_LOW].y
    brow_gap = (left_diff + right_diff) / 2
    return brow_gap * h

def mouth_openness(landmarks, h):
    return abs(landmarks[MOUTH_TOP].y - landmarks[MOUTH_BOTTOM].y) * h

last_blink_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_result = face_mesh.process(rgb)
    attention_score = 100
    now = time.time()
    feedback = []

    if face_result.multi_face_landmarks:
        lm = face_result.multi_face_landmarks[0].landmark

        # Blink detection
        if detect_blink(lm, w, h):
            if now - last_blink_time > 0.15:
                blink_times.append(now)
                last_blink_time = now
        while blink_times and now - blink_times[0] > 60:
            blink_times.popleft()
        blink_rate = len(blink_times)

        # Eye movement scatter
        iris_center = get_eye_center(lm)
        eye_positions.append(iris_center)
        scatter_value = np.std(eye_positions) if len(eye_positions) > 10 else 0
        if scatter_value > 0.01:
            attention_score -= 10
            feedback.append("Distracted Eye Movement")

        # Frown detection
        frown = detect_frown(lm, w, h)
        frown_scores.append(frown)
        avg_frown = np.mean(frown_scores)
        if avg_frown < 0.015 * h:
            attention_score -= 10
            feedback.append("Frowning Detected")

        # Microexpression: Mouth Open
        mouth_open = mouth_openness(lm, h)
        if mouth_open > 15:
            attention_score -= 5
            feedback.append("Mouth Movement")

        # Estimate cognitive load using ML model
        feature_vector = np.array([[blink_rate, scatter_value, avg_frown, mouth_open, attention_score]])
        pred = model.predict(feature_vector)[0]
        load_text = "High Load" if pred == 1 else "Normal"
        if pred == 1:
            feedback.append("⚠ Cognitive Load High")
            attention_score -= 10

        mp_drawing.draw_landmarks(frame, face_result.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION)

        # Draw feedback
        score_color = (0, 255, 0) if attention_score > 80 else (0, 165, 255) if attention_score > 50 else (0, 0, 255)
        cv2.putText(frame, f"Attention: {attention_score}%", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 3)
        cv2.putText(frame, f"Cognitive Load: {load_text}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
        for i, f in enumerate(feedback):
            cv2.putText(frame, f, (20, 120 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Cognitive Load Tracker", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
