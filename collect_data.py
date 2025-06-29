import cv2
import mediapipe as mp
import numpy as np
import csv
import os
from datetime import datetime

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Output setup
label = input("Enter label for this session (low_load/high_load): ")
date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"data/{label}/features_{date_str}.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# CSV header
header = ['ear', 'blink_rate', 'brow_dist', 'scatter', 'face_angle', 'mouth_open', 'label']

# Helpers
prev_eye_positions = []
blink_counter = 0
blink_frames = []

# Feature extraction function
def extract_advanced_features(landmarks, w, h, prev_eye_positions, blink_counter, blink_frames):
    # Eye aspect ratio (EAR)
    left_ear = np.linalg.norm(np.array(landmarks[159])*[w, h] - np.array(landmarks[145])*[w, h])
    right_ear = np.linalg.norm(np.array(landmarks[386])*[w, h] - np.array(landmarks[374])*[w, h])
    ear = (left_ear + right_ear) / 2

    # Blink detection
    blink_threshold = 4.5
    blink = 1 if ear < blink_threshold else 0
    blink_counter += blink
    blink_frames.append(blink)
    if len(blink_frames) > 60:
        blink_frames.pop(0)
    blink_rate = sum(blink_frames) / len(blink_frames)

    # Brow distance
    brow_dist = np.linalg.norm(np.array(landmarks[70])*[w, h] - np.array(landmarks[300])*[w, h])

    # Eye scatter
    eye_center = np.mean([landmarks[33][0], landmarks[263][0]])
    prev_eye_positions.append(eye_center)
    if len(prev_eye_positions) > 20:
        prev_eye_positions.pop(0)
    scatter = np.std(prev_eye_positions)

    # Head angle (simplified)
    nose_tip = np.array(landmarks[1])*[w, h]
    chin = np.array(landmarks[152])*[w, h]
    face_angle = np.arctan2(chin[1] - nose_tip[1], chin[0] - nose_tip[0])

    # Mouth openness
    mouth = abs(landmarks[13][1] - landmarks[14][1]) * h

    return [ear, blink_rate, brow_dist, scatter, face_angle, mouth], blink_counter

# Start webcam
cap = cv2.VideoCapture(0)
with open(output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = [(lm.x, lm.y) for lm in face.landmark]

            features, blink_counter = extract_advanced_features(
                landmarks, image.shape[1], image.shape[0],
                prev_eye_positions, blink_counter, blink_frames
            )

            row = features + [label]
            writer.writerow(row)

        cv2.imshow('Cognitive Load Data Collector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
