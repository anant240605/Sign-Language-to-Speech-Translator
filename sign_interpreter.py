# sign_interpreter.py
import cv2
import mediapipe as mp
import numpy as np
import argparse
import pickle
import os
from collections import deque, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import threading, queue
import time

mp_hands = mp.solutions.hands

def extract_landmarks(results):
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0].landmark
    arr = np.array([[l.x, l.y, l.z] for l in lm]).flatten()
    return arr

def normalize_landmarks(arr):
    lm = arr.reshape(21,3).astype(np.float32)
    origin = lm[0].copy()
    lm = lm - origin
    maxv = np.max(np.abs(lm))
    if maxv > 0:
        lm = lm / maxv
    return lm.flatten()

def collect_mode(args):
    labels = args.labels.split(",") if args.labels else ["hello","yes","no","thanks","stop"]
    out_file = args.datafile
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    data = {lab: [] for lab in labels}
    idx = 0
    recording = False
    print("Controls: 'r' toggle recording, 'n' next label, 'b' prev label, 'q' quit & save")
    print("Labels order:", labels)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        lm = None
        if results.multi_hand_landmarks:
            lm_raw = extract_landmarks(results)
            lm = normalize_landmarks(lm_raw)
            for handLms in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
        status = f"Label: {labels[idx]} | Recording: {'ON' if recording else 'OFF'}"
        cv2.putText(frame, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        counts_text = " | ".join([f"{lab}:{len(data[lab])}" for lab in labels])
        cv2.putText(frame, counts_text, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 1)
        cv2.imshow("Collect Data", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
            time.sleep(0.2)
        elif key == ord('n'):
            idx = (idx + 1) % len(labels)
            time.sleep(0.2)
        elif key == ord('b'):
            idx = (idx - 1) % len(labels)
            time.sleep(0.2)
        elif key == ord('q'):
            break
        if recording and lm is not None:
            data[labels[idx]].append(lm.tolist())
    cap.release()
    cv2.destroyAllWindows()
    with open(out_file, "wb") as f:
        pickle.dump(data, f)
    print("Saved data to", out_file)

def train_mode(args):
    datafile = args.datafile
    model_file = args.modelfile
    if not os.path.exists(datafile):
        print("Data file not found:", datafile)
        return
    with open(datafile, "rb") as f:
        data = pickle.load(f)
    X, y = [], []
    for label, samples in data.items():
        for s in samples:
            X.append(s)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    if len(X) == 0:
        print("No samples found. Collect data first.")
        return
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, ypred))
    print("Classification report:")
    print(classification_report(y_test, ypred))
    joblib.dump(clf, model_file)
    print("Saved model to", model_file)

def predict_mode(args):
    model_file = args.modelfile
    if not os.path.exists(model_file):
        print("Model not found:", model_file)
        return
    clf = joblib.load(model_file)
    classes = clf.classes_
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                           min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    
    pred_buffer = deque(maxlen=3)  # reduced size for faster update
    last_announced = None
    threshold = args.threshold
    
    import pyttsx3
    engine = pyttsx3.init()
    
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        label_text = "No Hand"
        prob_text = ""
        
        if results.multi_hand_landmarks:
            lm_raw = extract_landmarks(results)
            lm = normalize_landmarks(lm_raw)
            proba = clf.predict_proba([lm])[0]
            idx = np.argmax(proba)
            lab = classes[idx]
            conf = proba[idx]
            
            pred_buffer.append((lab, conf))
            
            # Calculate majority in buffer with threshold
            votes = [l for l,c in pred_buffer if c >= threshold]
            if votes:
                most = Counter(votes).most_common(1)[0]
                maj_label, count = most[0], most[1]
                label_text = maj_label
                prob_text = f"{conf:.2f}"
                
                # Announce if changed
                if last_announced != maj_label:
                    engine.say(maj_label)
                    engine.runAndWait()
                    last_announced = maj_label
        
        cv2.putText(frame, f"Pred: {label_text} {prob_text}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
        cv2.imshow("Sign Language - Predict", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["collect","train","predict"], help="Mode to run")
    parser.add_argument("--labels", type=str, default=None, help="Comma-separated labels for collect (e.g. hello,yes,no)")
    parser.add_argument("--datafile", type=str, default="gesture_data.pkl", help="Where to save/load data")
    parser.add_argument("--modelfile", type=str, default="gesture_model.joblib", help="Where to save/load model")
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for speak")
    args = parser.parse_args()
    if args.mode == "collect":
        collect_mode(args)
    elif args.mode == "train":
        train_mode(args)
    elif args.mode == "predict":
        predict_mode(args)

if __name__ == "__main__":
    main()
