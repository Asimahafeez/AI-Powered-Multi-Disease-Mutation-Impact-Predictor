import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# ---------- FIX: Updated build_features ----------
def build_features(df):
    """Convert all non-numeric columns to numeric using label encoding."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    df = df.fillna(0)
    return df

st.set_page_config(page_title="AI Mutation Impact Predictor", layout="wide")
st.title("🧬 AI-Powered Multi-Disease Mutation Impact Predictor")

uploaded_file = st.file_uploader("📂 Upload Mutation Data CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(df.head())

    # Assume last column is target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Build features (handle strings → numbers)
    X = build_features(X)

    # Train simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # Metrics
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    st.write(f"✅ Accuracy: {acc:.2f}")
    st.write(f"📈 ROC AUC: {auc:.2f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    # Feature importance
    st.subheader("🔍 Feature Importance")
    importance = permutation_importance(model, X, y, random_state=42)
    feat_importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance.importances_mean
    }).sort_values(by="Importance", ascending=False)  # ✅ properly closed
    st.dataframe(feat_importance)

else:
    st.info("Please upload a CSV file to continue.")

st.markdown("---")
st.caption("Created for research demonstration purposes.")
