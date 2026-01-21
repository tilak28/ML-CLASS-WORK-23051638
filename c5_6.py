# =========================================================
# STEP 1: IMPORT LIBRARIES
# =========================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# STEP 2: LOAD DATASET
# =========================================================

df = pd.read_csv("world_airports.csv")

print("Dataset Loaded Successfully")
print(df.head())
print("Shape:", df.shape)

# =========================================================
# STEP 3: DATA CLEANING
# =========================================================

# Keep only required columns
df = df[[
    "STATE",
    "LATITUDE",
    "LONGITUDE"
]]

# Drop missing values
df = df.dropna()

print("\nAfter Cleaning Shape:", df.shape)

# =========================================================
# STEP 4: ENCODE TARGET FOR CLASSIFICATION
# =========================================================

label_encoder = LabelEncoder()
df["STATE_encoded"] = label_encoder.fit_transform(df["STATE"])

# =========================================================
# STEP 5: CLASSIFICATION TASK
# Predict STATE
# =========================================================

X_clf = df[["LATITUDE", "LONGITUDE"]]
y_clf = df["STATE_encoded"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# =========================================================
# STEP 6: CLASSIFICATION ALGORITHM COMPARISON
# =========================================================

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

print("\n===== CLASSIFICATION MODEL COMPARISON =====\n")

clf_results = []

for name, model in classifiers.items():
    model.fit(Xc_train, yc_train)
    preds = model.predict(Xc_test)
    acc = accuracy_score(yc_test, preds)

    clf_results.append((name, acc))
    print(f"{name} Accuracy: {acc:.4f}")

clf_df = pd.DataFrame(clf_results, columns=["Model", "Accuracy"])
print("\nClassification Comparison Table:")
print(clf_df)

# =========================================================
# STEP 7: CLASSIFICATION REPORT (BEST MODEL)
# =========================================================

best_clf = RandomForestClassifier(n_estimators=100)
best_clf.fit(Xc_train, yc_train)
best_preds = best_clf.predict(Xc_test)

print("\nClassification Report (Random Forest):\n")
print(classification_report(
    yc_test,
    best_preds,
    zero_division=0,
    target_names=label_encoder.classes_
))

# =========================================================
# STEP 8: REGRESSION TASK
# Predict LATITUDE using LONGITUDE
# =========================================================

X_reg = df[["LONGITUDE"]]
y_reg = df["LATITUDE"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# =========================================================
# STEP 9: REGRESSION ALGORITHM COMPARISON
# =========================================================

regressors = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100)
}

print("\n===== REGRESSION MODEL COMPARISON =====\n")

reg_results = []

for name, model in regressors.items():
    model.fit(Xr_train, yr_train)
    preds = model.predict(Xr_test)

    mse = mean_squared_error(yr_test, preds)
    r2 = r2_score(yr_test, preds)

    reg_results.append((name, mse, r2))
    print(f"{name} -> MSE: {mse:.2f}, R2: {r2:.4f}")

reg_df = pd.DataFrame(
    reg_results,
    columns=["Model", "MSE", "R2 Score"]
)

print("\nRegression Comparison Table:")
print(reg_df)

# =========================================================
# STEP 10: SAVE OUTPUT
# =========================================================

output = pd.DataFrame({
    "LATITUDE": Xc_test["LATITUDE"].values,
    "LONGITUDE": Xc_test["LONGITUDE"].values,
    "Actual_STATE": label_encoder.inverse_transform(yc_test),
    "Predicted_STATE": label_encoder.inverse_transform(best_preds)
})

output.to_csv("airport_state_predictions.csv", index=False)
print("\nFile saved as airport_state_predictions.csv")

