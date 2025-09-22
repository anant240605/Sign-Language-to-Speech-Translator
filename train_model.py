# train_model.py
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# 1️⃣ Load gesture data
with open("gesture_data.pkl", "rb") as f:
    data = pickle.load(f)

X, y = [], []
for label, samples in data.items():
    for s in samples:
        X.append(s)
        y.append(label)

print("Total samples:", len(X))

# 2️⃣ Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# 3️⃣ Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4️⃣ Train RandomForest
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# 5️⃣ Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 6️⃣ Save model
joblib.dump(clf, "gesture_model.joblib")
print("Model saved as gesture_model.joblib")
