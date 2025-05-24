# 🧠 Cancer Prediction Model (Binary Classification using Random Forest)

This project is a **Streamlit-based web application** that predicts the **presence or absence of cancer** based on user input or batch CSV file uploads. It uses a **Random Forest Classifier** for binary classification, trained on a health-related dataset.

---

## 🚀 Features

- **Manual Prediction**: Enter personal health parameters manually.
- **Batch Prediction**: Upload a CSV file with multiple records for prediction.
- **Downloadable Results**: Get predictions as downloadable CSV files.
- **Real-time UI**: Simple, responsive interface built using Streamlit.
- **Timestamped Logs**: Batch results saved automatically in the `artifacts/` folder.

---

## 🔍 Problem Statement

The goal is to predict whether a person is likely to have cancer (`1`) or not (`0`) based on risk factors like age, BMI, smoking, genetic history, etc.

This is a **binary classification problem** tackled using **Random Forest**, a robust ensemble learning method ideal for tabular data and interpretability.

---

## 🧪 Dataset Description

- Format: CSV
- Location: `data/cancer.csv`
- Columns:
  - `Age`
  - `Gender` (0 = Male, 1 = Female)
  - `BMI`
  - `Smoking` (0 = No, 1 = Yes)
  - `Genetic Risk` (0 = Low, 1 = Medium, 2 = High)
  - `Physical Activity` (hours per week)
  - `Alcohol Intake` (units per week)
  - `Cancer History` (0 = No, 1 = Yes)
  - `Cancer` (Target column: 0 = No, 1 = Yes)

---

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier
- **Library**: `sklearn.ensemble.RandomForestClassifier`
- **Parameters**:
  - `n_estimators = 100`
  - `random_state = 42`
- **Preprocessing**: Feature standardization using `(x - mean) / std`

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/1DS22AI008AYUSHMEHTA/cancer_prediction.git
cd cancer-prediction-app
