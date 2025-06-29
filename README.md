# Cognitive Load Estimator via Webcam

This Python project estimates a user's **cognitive load** in real time using webcam input. It uses **computer vision** (OpenCV, MediaPipe) to extract facial features like blink rate, eye movement scatter, facial microexpressions, and head movement. These features are then fed into a **machine learning model** to predict mental workload (low or high).

---

## 🔍 Features

-  Real-time webcam feed processing
-  Blink rate analysis (fatigue detection)
-  Eye movement scatter (attention/distraction detection)
-  Facial microexpression tracking (frown, squint, mouth open)
-  Machine Learning classifier (Random Forest / Logistic Regression)
-  Scalable attention scoring system
-  Alerts or logs cognitive load levels
-  Collect your own data for training and retraining

---

## 🛠️ Tech Stack

| Component     | Technology            |
|---------------|------------------------|
| Language       | Python 3.x            |
| Computer Vision| OpenCV, MediaPipe     |
| ML Model       | Scikit-learn          |
| Data Handling  | Pandas, NumPy         |
| Visualization  | Matplotlib (optional) |

---


---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/YOUR_USERNAME/cognitive-load-estimator.git
cd cognitive-load-estimator


