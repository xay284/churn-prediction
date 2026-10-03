import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn risk", layout="centered")

bundle = joblib.load("models/churn_model.joblib")
model, threshold = bundle["model"], bundle["threshold"]
num_cols, cat_cols = bundle["numeric_cols"], bundle["categorical_cols"]
df = pd.read_csv("data/telco_churn.csv")

st.title("Customer churn risk")
st.caption("Enter a customer profile to estimate the probability of churn.")

inputs = {}
left, right = st.columns(2)
for i, c in enumerate(cat_cols):
    with (left if i % 2 == 0 else right):
        inputs[c] = st.selectbox(c, sorted(df[c].unique().tolist()))

tenure = st.slider("tenure (months)", 0, 72, 12)
monthly = st.number_input(
    "MonthlyCharges",
    min_value=float(df["MonthlyCharges"].min()),
    max_value=float(df["MonthlyCharges"].max()),
    value=float(df["MonthlyCharges"].median()),
)

row = pd.DataFrame([{**inputs, "tenure": tenure, "MonthlyCharges": monthly,
                     "TotalCharges": tenure * monthly}])[num_cols + cat_cols]
proba = model.predict_proba(row)[0, 1]

st.metric("Churn probability", f"{proba:.1%}")
if proba >= threshold:
    st.error(f"High risk (above the {threshold:.0%} decision threshold): consider a retention offer.")
else:
    st.success(f"Lower risk (below the {threshold:.0%} decision threshold).")