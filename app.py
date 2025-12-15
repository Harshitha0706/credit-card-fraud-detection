import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

model = load_model()
FEATURES = list(model.feature_names_in_)

# ---------------- TITLE ----------------
st.title("💳 Credit Card Fraud Detection System")
st.markdown("""
This dashboard predicts whether a transaction is **Fraudulent** or **Genuine**
using a trained **XGBoost Machine Learning model**.
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Upload Transaction Data")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (same format as dataset)",
    type=["csv"]
)

threshold = st.sidebar.slider(
    "Fraud Detection Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

# ---------------- BATCH PREDICTION ----------------
if uploaded_file:
    data = pd.read_csv(uploaded_file)

    # Drop label if present
    if "Class" in data.columns:
        data = data.drop("Class", axis=1)

    # Enforce correct feature order
    data = data[FEATURES]

    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(data.head())

    # Predict
    probabilities = model.predict_proba(data)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = data.copy()
    results["Fraud_Probability"] = probabilities
    results["Prediction"] = np.where(
        results["Fraud_Probability"] >= threshold, "Fraud", "Genuine"
    )

    st.subheader("🔍 Prediction Results")
    st.dataframe(results.head())

    # Summary
    fraud_count = int(sum(predictions))
    total = len(predictions)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Transactions", total)
    col2.metric("Predicted Frauds", fraud_count)
    col3.metric("Fraud Percentage", f"{(fraud_count/total)*100:.2f}%")

else:
    st.info("👈 Upload a CSV file to start predictions")

# ---------------- SINGLE TRANSACTION ----------------
st.markdown("---")
st.subheader("🧪 Test a Single Transaction")

with st.form("single_tx"):
    cols = st.columns(5)
    inputs = []

    for i, col in enumerate(FEATURES):
        val = cols[i % 5].number_input(col, value=0.0)
        inputs.append(val)

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([inputs], columns=FEATURES)
    prob = model.predict_proba(input_df)[0][1]
    result = "Fraud" if prob >= threshold else "Genuine"

    st.success(f"Prediction: **{result}**")
    st.write(f"Fraud Probability: **{prob:.4f}**")
