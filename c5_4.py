
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Classification models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("cmd_commands.csv")

print("Dataset Loaded Successfully")
print(df.head())
print("Shape:", df.shape)


X = df["description"]
y = df["name"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF for classification
tfidf_clf = TfidfVectorizer(stop_words="english")
X_train_tfidf = tfidf_clf.fit_transform(X_train)
X_test_tfidf = tfidf_clf.transform(X_test)

# =========================================================
# STEP 4: CLASSIFICATION ALGORITHM COMPARISON
# =========================================================

classifiers = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=300),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

print("\n===== CLASSIFICATION MODEL COMPARISON =====\n")

clf_results = []

for name, model in classifiers.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    clf_results.append((name, acc))
    print(f"{name} Accuracy: {acc:.4f}")

clf_df = pd.DataFrame(clf_results, columns=["Model", "Accuracy"])
print("\nClassification Comparison Table:")
print(clf_df)

# =========================================================
# STEP 5: CLEAN CLASSIFICATION REPORT
# =========================================================

best_model = MultinomialNB()
best_model.fit(X_train_tfidf, y_train)
best_preds = best_model.predict(X_test_tfidf)

print("\nClassification Report (Naive Bayes):\n")
print(classification_report(
    y_test,
    best_preds,
    zero_division=0
))

# =========================================================
# STEP 6: CUSTOM USER INPUT TEST
# =========================================================

custom_input = ["delete a directory recursively"]
custom_vector = tfidf_clf.transform(custom_input)
custom_prediction = best_model.predict(custom_vector)

print("\nCustom Input Prediction:")
print("Input:", custom_input[0])
print("Predicted Command:", custom_prediction[0])

# =========================================================
# STEP 7: REGRESSION TASK
# Predict command name length
# =========================================================

df["command_length"] = df["name"].apply(len)

X_reg = df["description"]
y_reg = df["command_length"]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Separate TF-IDF for regression
tfidf_reg = TfidfVectorizer(stop_words="english")
Xr_train_tfidf = tfidf_reg.fit_transform(Xr_train)
Xr_test_tfidf = tfidf_reg.transform(Xr_test)

# =========================================================
# STEP 8: REGRESSION MODEL COMPARISON
# =========================================================

regressors = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100)
}

print("\n===== REGRESSION MODEL COMPARISON =====\n")

reg_results = []

for name, model in regressors.items():
    model.fit(Xr_train_tfidf, yr_train)
    preds = model.predict(Xr_test_tfidf)

    mse = mean_squared_error(yr_test, preds)
    r2 = r2_score(yr_test, preds)

    reg_results.append((name, mse, r2))
    print(f"{name} -> MSE: {mse:.4f}, R2: {r2:.4f}")

reg_df = pd.DataFrame(
    reg_results,
    columns=["Model", "MSE", "R2 Score"]
)

print("\nRegression Comparison Table:")
print(reg_df)

# =========================================================
# STEP 9: SAVE OUTPUT (LOCAL SAFE)
# =========================================================

output = pd.DataFrame({
    "Description": X_test.values,
    "Actual Command": y_test.values,
    "Predicted Command": best_preds
})

output.to_csv("classification_predictions.csv", index=False)
print("\nFile saved successfully as classification_predictions.csv")
