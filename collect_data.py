import cv2, mediapipe as mp, numpy as np, pickle

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

labels = ["hello", "yes", "no"]  # change to your gestures
data = {label: [] for label in labels}
current_label = 0

print("Press 'n' to move to next gesture, 'q' to quit")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for lm in results.multi_hand_landmarks[0].landmark:
            cv2.circle(frame, (int(lm.x*frame.shape[1]), int(lm.y*frame.shape[0])), 3, (0,255,0), -1)
        # get landmark coordinates as flat array
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]).flatten()
        data[labels[current_label]].append(landmarks)
    cv2.putText(frame, f"Collecting: {labels[current_label]}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Collect Data", frame)
    key = cv2.waitKey(1)
    if key == ord('n'):  # move to next gesture
        current_label = (current_label + 1) % len(labels)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# save data
with open("gesture_data.pkl", "wb") as f:
    pickle.dump(data, f)

# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# import joblib

# # Load gesture data
# with open("gesture_data.pkl", "rb") as f:
#     data = pickle.load(f)

# X, y = [], []
# for label, samples in data.items():
#     for s in samples:
#         X.append(s)
#         y.append(label)

# print("Total samples:", len(X))

# import numpy as np
# X = np.array(X)
# y = np.array(y)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# clf = RandomForestClassifier(n_estimators=200, random_state=42)
# clf.fit(X_train, y_train)




# y_pred = clf.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))


# joblib.dump(clf, "gesture_model.joblib")
# print("Model saved as gesture_model.joblib")

