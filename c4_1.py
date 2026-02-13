import os
import numpy as np
import pandas as pd
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Download Iris dataset using kagglehub
path = kagglehub.dataset_download("uciml/iris")
print("Path to dataset files:", path)

# Load CSV file
csv_file = os.path.join(path, "Iris.csv")
df = pd.read_csv(csv_file)

# Display first few rows
print("\nDataset Preview:")
print(df.head())

# Drop ID column
df = df.drop(columns=["Id"])

# Convert target to binary classification
# Iris-setosa -> 1, others -> 0
df["Species"] = (df["Species"] == "Iris-setosa").astype(int)

# Features and target
X = df.drop(columns=["Species"])
y = df["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Extract TN, FP, FN, TP
TN, FP, FN, TP = cm.ravel()

print("\nConfusion Matrix Values:")
print("True Positives (TP):", TP)
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)

# -------- Manual Metric Calculations --------

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

# Print metrics
print("\nEvaluation Metrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1_score:.4f}")
