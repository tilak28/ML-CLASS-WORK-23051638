# =========================================================
# STEP 1: IMPORT LIBRARIES
# =========================================================

import numpy as np
import pandas as pd
import os

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
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

# =========================================================
# STEP 2: LOAD DATASET
# =========================================================

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('cmd_commands.csv')


print("\nDataset Loaded Successfully")
print(df.head())

# =========================================================
# STEP 3: CLASSIFICATION TASK
# Predict command name from description
# =========================================================

X = df['description']
y = df['name']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

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

classification_results = []

for name, model in classifiers.items():
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    
    acc = accuracy_score(y_test, preds)
    classification_results.append((name, acc))
    
    print(f"{name} Accuracy: {acc:.4f}")

# Display comparison table
clf_result_df = pd.DataFrame(
    classification_results,
    columns=["Model", "Accuracy"]
)

print("\nClassification Comparison Table:")
print(clf_result_df)

# =========================================================
# STEP 5: DETAILED REPORT FOR BEST CLASSIFIER
# =========================================================

best_model = MultinomialNB()
best_model.fit(X_train_tfidf, y_train)
best_preds = best_model.predict(X_test_tfidf)

print("\nClassification Report (Naive Bayes):\n")
print(classification_report(y_test, best_preds))

# =========================================================
# STEP 6: CUSTOM USER INPUT PREDICTION
# =========================================================

custom_text = ["delete a directory recursively"]
custom_vector = tfidf.transform(custom_text)
prediction = best_model.predict(custom_vector)

print("Custom Input Prediction:")
print("Input:", custom_text[0])
print("Predicted Command:", prediction[0])

# =========================================================
# STEP 7: REGRESSION TASK
# Predict command name length from description
# =========================================================

# Create regression target
df['command_length'] = df['name'].apply(len)

X_reg = df['description']
y_reg = df['command_length']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# TF-IDF for regression
Xr_train_tfidf = tfidf.fit_transform(Xr_train)
Xr_test_tfidf = tfidf.transform(Xr_test)

# =========================================================
# STEP 8: REGRESSION ALGORITHM COMPARISON
# =========================================================

regressors = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100)
}

print("\n===== REGRESSION MODEL COMPARISON =====\n")

regression_results = []

for name, model in regressors.items():
    model.fit(Xr_train_tfidf, yr_train)
    preds = model.predict(Xr_test_tfidf)
    
    mse = mean_squared_error(yr_test, preds)
    r2 = r2_score(yr_test, preds)
    
    regression_results.append((name, mse, r2))
    
    print(f"{name} -> MSE: {mse:.4f}, R2: {r2:.4f}")

# Display regression comparison table
reg_result_df = pd.DataFrame(
    regression_results,
    columns=["Model", "MSE", "R2 Score"]
)

print("\nRegression Comparison Table:")
print(reg_result_df)

# =========================================================
# STEP 9: SAVE OUTPUT FILE
# =========================================================

output = pd.DataFrame({
    "Description": X_test.values,
    "Actual Command": y_test.values,
    "Predicted Command": best_preds
})

output.to_csv('/kaggle/working/classification_predictions.csv', index=False)
print("\nOutput file saved successfully!")
