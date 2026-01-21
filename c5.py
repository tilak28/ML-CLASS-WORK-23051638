import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Models to compare
models = [
    ('Logistic Regression', LogisticRegression(max_iter=200)),
    ('Decision Tree', DecisionTreeClassifier()),
    ('KNN', KNeighborsClassifier()),
    ('SVM', SVC()),
    ('Naive Bayes', GaussianNB()),
    ('Random Forest', RandomForestClassifier())
]

# Evaluate using cross-validation
results = []
names = []

for name, model in models:
    cv = model_selection.cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results.append(cv)
    names.append(name)
    print(f"{name}: {cv.mean():.3f} ({cv.std():.3f})")

# Plot
plt.boxplot(results, labels=names)
plt.title("Classification Algorithm Comparison")
plt.xticks(rotation=45)
plt.show()
