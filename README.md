# AI‑Powered Multi‑Disease Mutation Impact Predictor (Streamlit)

Advanced, light‑weight app that predicts variant pathogenicity across diseases using
biochemical features (Grantham, BLOSUM62, hydrophobicity, charge, polarity, volume, conservation, position).

## 🔧 Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Input format
CSV columns (train): 
- `ref_aa`, `alt_aa` (single letter AAs, required)  
- `pos` (int), `seq_len` (int), `conservation` (0..1), `disease`, `gene` (optional)  
- `label` (0/1, required for training)

CSV columns (predict-only): same as above **without** `label`.

## 🧠 Models
- Logistic Regression (with scaling) or Gradient Boosting.
- Disease‑specific thresholds (Youden's J on train splits).
- Global permutation importance; linear per‑variant contributions.

## 🧪 Demo data
`data/demo_variants.csv` is included (3 diseases) to try instantly.

## ⚠️ Disclaimer
Research tool only. Not for clinical use.
