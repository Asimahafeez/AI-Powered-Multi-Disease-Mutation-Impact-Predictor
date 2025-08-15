
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

from utils.feature_engineering import build_features

st.set_page_config(page_title="AI Mutation Impact Predictor", layout="wide")
st.title("🧬 AI‑Powered Multi‑Disease Mutation Impact Predictor")
st.write("Upload a **variant CSV** with columns: `ref_aa`, `alt_aa`, `pos`, `seq_len`, "
         "`conservation` (optional), `disease` (optional), `gene` (optional), and **`label`** (0/1). "
         "Or leave empty to use the built‑in multi‑disease demo.")

# ==== Sidebar ====
st.sidebar.header("Settings")
use_demo = st.sidebar.checkbox("Use demo data", value=True)
model_choice = st.sidebar.selectbox("Model", ["Logistic Regression", "Gradient Boosting"])
test_size = st.sidebar.slider("Test size", 0.15, 0.4, 0.25, 0.05)
rs = st.sidebar.number_input("Random state", value=42, step=1)

uploaded = st.file_uploader("📂 Upload variants CSV", type=["csv"])

# ==== Load ====
if use_demo or uploaded is None:
    df = pd.read_csv("data/demo_variants.csv")
    st.info("Using demo variants (3 diseases).")
else:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()

if "label" not in df.columns:
    st.error("Your CSV must contain a binary **label** column (0=benign,1=pathogenic) for training.")
    st.stop()

st.subheader("📊 Data Preview")
st.dataframe(df.head())

# ==== Features ====
X, feat_names, df_enriched = build_features(df)
y = df["label"].astype(int)

# Guard: need at least 2 numeric features after imputation
if X.shape[1] < 2:
    st.error("Not enough features after preprocessing.")
    st.stop()

# ==== Split ====
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)), test_size=test_size, random_state=rs, stratify=y
)

# ==== Pipeline ====
if model_choice == "Logistic Regression":
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400))
    ])
else:
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(random_state=rs))
    ])

pipe.fit(X_train, y_train)
probs = pipe.predict_proba(X_test)[:,1]
preds = (probs >= 0.5).astype(int)

auc = roc_auc_score(y_test, probs)
acc = accuracy_score(y_test, preds)
cm = confusion_matrix(y_test, preds)

c1, c2, c3 = st.columns(3)
c1.metric("ROC AUC", f"{auc:.3f}")
c2.metric("Accuracy", f"{acc:.3f}")
c3.metric("# Test Variants", f"{len(y_test)}")

# ==== Disease-specific thresholding ====
st.subheader("🎯 Disease‑specific thresholds (Youden J)")
if "disease" in df.columns and df["disease"].nunique() >= 2:
    disease_thresholds = {}
    for dis in sorted(df["disease"].dropna().unique()):
        # compute threshold from train set for this disease
        mask_tr = (df.iloc[idx_train]["disease"] == dis).values
        if mask_tr.sum() >= 10:
            # probs for that disease on train via cross-fit proxy (use all train as proxy)
            y_tr_d = y_train[mask_tr]
            if len(np.unique(y_tr_d)) >= 2:
                pr_tr = pipe.predict_proba(X_train[mask_tr])[:,1]
                fpr, tpr, thr = roc_curve(y_tr_d, pr_tr)
                j = tpr - fpr
                disease_thresholds[dis] = thr[np.argmax(j)]
    if disease_thresholds:
        st.write({k: round(float(v),3) for k,v in disease_thresholds.items()})
    else:
        st.write("Not enough per‑disease data to compute robust thresholds.")
else:
    st.info("No `disease` column found; using global threshold 0.5.")

# ==== ROC ====
st.subheader("📈 ROC Curve")
from sklearn.metrics import RocCurveDisplay
fpr, tpr, _ = roc_curve(y_test, probs)
fig1, ax1 = plt.subplots()
ax1.plot(fpr, tpr, label=f"AUC={auc:.3f}")
ax1.plot([0,1],[0,1],'--')
ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.legend(loc="lower right")
st.pyplot(fig1)

# ==== Permutation importance ====
st.subheader("🧠 Global Feature Importance (Permutation)")
try:
    r = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=rs, n_jobs=-1)
    imp = pd.DataFrame({"feature": feat_names, "importance": r.importances_mean}).sort_values("importance", ascending=False)
    st.dataframe(imp.reset_index(drop=True))
    fig2, ax2 = plt.subplots(figsize=(8, max(4, len(imp)*0.35)))
    ax2.barh(imp["feature"][::-1], imp["importance"][::-1])
    ax2.set_xlabel("Mean decrease in score")
    st.pyplot(fig2)
except Exception as e:
    st.warning(f"Permutation importance skipped: {e}")

# ==== Per-variant contributions (linear) ====
if model_choice == "Logistic Regression":
    st.subheader("🔬 Per‑variant contribution (linear model)")
    # Show contributions for first 5 test variants
    import numpy as np
    # Get scaler if present
    scaler = pipe.named_steps.get("scaler", None)
    imp = pipe.named_steps["imp"]
    clf = pipe.named_steps["clf"]
    X_imp = imp.transform(X_test)
    if scaler is not None:
        X_std = scaler.transform(X_imp)
    else:
        X_std = X_imp
    coefs = clf.coef_.ravel()
    contrib = X_std * coefs  # contribution per feature
    dfc = pd.DataFrame(contrib[:5], columns=feat_names)
    st.write("Top 5 test variants contributions (feature value × coefficient):")
    st.dataframe(dfc)

# ==== Predict on new variants ====
st.subheader("🧪 Predict on New Variants")
st.write("Upload a CSV **without label** but with `ref_aa`, `alt_aa`, and optional columns (`pos`,`seq_len`,`conservation`,`disease`,`gene`).")
newf = st.file_uploader("Upload new variants CSV", type=["csv"], key="new")
if newf is not None:
    try:
        new_df = pd.read_csv(newf)
        X_new, _, new_enr = build_features(new_df)
        proba = pipe.predict_proba(X_new)[:,1]
        new_enr["patho_prob"] = proba
        # Apply disease-specific threshold if available
        if "disease" in new_enr.columns and 'disease_thresholds' in locals() and disease_thresholds:
            thr = new_enr["disease"].map(disease_thresholds).fillna(0.5)
            new_enr["pred_class"] = (new_enr["patho_prob"] >= thr).astype(int)
        else:
            new_enr["pred_class"] = (new_enr["patho_prob"] >= 0.5).astype(int)
        st.success("Predictions computed.")
        st.dataframe(new_enr[["gene","ref_aa","alt_aa","pos","disease","patho_prob","pred_class"]].head(20))
        # Download
        new_enr.to_csv("predictions.csv", index=False)
        with open("predictions.csv", "rb") as f:
            st.download_button("⬇️ Download predictions.csv", f, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.caption("Research prototype. Not for clinical use.")
