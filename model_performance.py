import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, confusion_matrix
import joblib

# Create artifacts directory
os.makedirs("artifacts", exist_ok=True)

# Load dataset
dataset = pd.read_csv("data/cancer.csv")

# Split features and target
features_indices = len(dataset.columns) - 1
X_df = dataset.iloc[:, :features_indices]
y_df = dataset.iloc[:, -1]

# Normalize data
X_df_mean = X_df.mean()
X_df_std = X_df.std()
X_df_norm = (X_df - X_df_mean) / X_df_std

# Save mean and std
np.savez("artifacts/mean_std.npz", mean=X_df_mean.values, std=X_df_std.values, columns=X_df.columns.values)

# Define models to compare
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42)
}

results = []
best_model = None
best_f1 = 0
best_model_name = ""

# Train and evaluate models
X = np.float64(X_df_norm.values)
y = np.float64(y_df.values)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "F1 Score": round(f1, 4),
        "ROC-AUC": round(roc, 4)
    })

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name

# Save best model
joblib.dump(best_model, f"artifacts/best_model_{best_model_name}.pkl")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("artifacts/model_comparison.csv", index=False)
with open("artifacts/metrics.json", "w") as f:
    json.dump(results, f, indent=4)

# Display metrics table
print("📊 Model Comparison:")
print(results_df)

print(f"\n✅ Best model saved: {best_model_name} (F1 Score: {best_f1:.4f}) in 'artifacts/' folder.")
