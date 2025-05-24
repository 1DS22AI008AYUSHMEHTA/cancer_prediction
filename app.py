
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

# Load and process dataset
dataset = pd.read_csv("data/cancer.csv")
features_indices = len(dataset.columns) - 1
X_df = dataset.iloc[:, :features_indices]
y_df = dataset.iloc[:, -1]

# Normalize data
X_df_mean = X_df.mean()
X_df_std = X_df.std()
X_df_norm = (X_df - X_df_mean) / X_df_std

# Train model
def train_model(X_df, y_df, train_size=0.9):
    X = np.float64(X_df.values)
    y = np.float64(y_df.values)
    X_train, _, y_train, _ = train_test_split(X, y, train_size=train_size, shuffle=True, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_df_norm, y_df)

# Streamlit UI
st.title("🧠 Cancer Prediction Interface")

# Dropdown to select prediction mode
prediction_mode = st.sidebar.selectbox("Select Prediction Mode", options=["Manual Prediction", "Batch Prediction"])

if prediction_mode == "Manual Prediction":
    st.subheader("🧍 Manual Prediction")
    st.markdown("Enter your health information below to check for cancer risk.")

    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    gender = st.selectbox("Gender", options=["Male", "Female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
    smoking = st.radio("Do you smoke?", options=["No", "Yes"])
    genetic_risk = st.selectbox("Genetic risk", options=["Low", "Medium", "High"])
    physical_activity = st.number_input("Physical Activity (hrs/week)", min_value=0, max_value=168, value=5)
    alcohol_intake = st.number_input("Alcohol Intake (units/week)", min_value=0, max_value=100, value=2)
    cancer_history = st.radio("History of Cancer", options=["No", "Yes"])

    gender_val = 0 if gender == "Male" else 1
    smoking_val = 0 if smoking == "No" else 1
    genetic_map = {"Low": 0, "Medium": 1, "High": 2}
    genetic_val = genetic_map[genetic_risk]
    cancer_history_val = 0 if cancer_history == "No" else 1

    input_data = np.array([age, gender_val, bmi, smoking_val, genetic_val,
                           physical_activity, alcohol_intake, cancer_history_val], dtype=float)
    input_data = (input_data - X_df_mean.values) / X_df_std.values
    input_data = input_data.reshape(1, -1)

    if st.button("Predict"):
        prediction = model.predict(input_data)
        if prediction == 0:
            st.success("✅ You have been diagnosed to be cancer negative.")
        else:
            st.error("⚠️ Sorry, you have been diagnosed to be cancer positive.")

elif prediction_mode == "Batch Prediction":
    st.subheader("📄 Batch Prediction via CSV")
    st.markdown("Upload a CSV file with the following columns:")
    st.code("Age,Gender,BMI,Smoking,Genetic Risk,Physical Activity,Alcohol Intake,Cancer History")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            user_data = pd.read_csv(uploaded_file)
            expected_columns = ["Age", "Gender", "BMI", "Smoking", "Genetic Risk",
                                "Physical Activity", "Alcohol Intake", "Cancer History"]

            if not all(col in user_data.columns for col in expected_columns):
                st.error(f"CSV must contain columns: {expected_columns}")
            else:
                # Ensure proper order and types
                user_data = user_data[expected_columns]
                user_data = user_data.astype(float)

                # Normalize
                user_data_norm = (user_data - X_df_mean.values) / X_df_std.values

                predictions = model.predict(user_data_norm)
                user_data["Prediction"] = np.where(predictions == 0, "Cancer Negative", "Cancer Positive")

                st.success("✅ Predictions complete!")
                num_to_show = st.slider("How many results to show?", min_value=1, max_value=len(user_data), value=10)
                st.dataframe(user_data.head(num_to_show))

                # Optional: Download
                csv = user_data.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results as CSV", data=csv, file_name="cancer_predictions.csv", mime='text/csv')

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")


