import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
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

# Train and evaluate model
def train_and_eval_model(X_df, y_df, train_size=0.9):
    X = np.float64(X_df.values)
    y = np.float64(y_df.values)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auroc = roc_auc_score(y_test, y_pred)
    f1score = f1_score(y_test, y_pred)
    confusion_mtx = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "Accuracy": accuracy,
        "AUROC": auroc,
        "F1Score": f1score,
        "ConfusionMatrix": confusion_mtx
    }

    return model, metrics

model, model_metrics = train_and_eval_model(X_df_norm, y_df)

# Save model
joblib.dump(model, "artifacts/model.pkl")

# Save metrics
with open("artifacts/metrics.json", "w") as f:
    json.dump(model_metrics, f, indent=4)

print("✅ Artifacts saved in 'artifacts/' folder.")
