import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib

# Load dataset
data = pd.read_csv("dataset/heart.csv", header=None)
data = data.replace("?", pd.NA)
data = data.apply(pd.to_numeric)
data = data.fillna(data.median())

# Input features
X = data.iloc[:, :-1]
# Target column
y = data.iloc[:, -1]

# Convert target into 2 classes
# 0 = No heart disease
# 1 = Heart disease
y = y.apply(lambda x: 0 if x == 0 else 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/heart_disease_model.pkl")

print("\nModel saved successfully!")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(model, "models/heart_disease_model.pkl")
# Confusion Matrix Graph
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5, 4))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0, 1], ["No Disease", "Disease"])
plt.yticks([0, 1], ["No Disease", "Disease"])

plt.colorbar()

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.savefig("outputs/confusion_matrix.png")
plt.close()
 # Feature Importance
feature_names = [
    "Age",
    "Sex",
    "Chest Pain",
    "Blood Pressure",
    "Cholesterol",
    "Fasting Blood Sugar",
    "Resting ECG",
    "Maximum Heart Rate",
    "Exercise Angina",
    "Oldpeak",
    "Slope",
    "Major Vessels",
    "Thal"
]

importance = model.feature_importances_

plt.figure(figsize=(8, 5))
plt.barh(feature_names, importance)

plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()




