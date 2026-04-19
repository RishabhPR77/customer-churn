# Customer Segmentation & Retention Analysis
### Dunnhumby — The Complete Journey Dataset

> End-to-end data science project: from raw retail transaction data to a deployable churn prediction model and actionable retention strategy.

---

## Project Overview

This project analyses **2,499 households** over a **2-year retail transaction window** using the Dunnhumby Complete Journey dataset (Kaggle). It combines unsupervised customer segmentation with supervised churn prediction to produce prioritised retention recommendations backed by a **$233,917 revenue-at-risk** estimate.

---

## Key Results

| Metric | Value |
|--------|-------|
| Households analysed | 2,499 |
| Features engineered | 22 behavioural features |
| Customer segments | 4 (KMeans, k=4) |
| Best model | Random Forest |
| Test ROC-AUC | **0.8793** |
| CV ROC-AUC | 0.8686 ± 0.0214 |
| Overall churn rate | 6.8% (171 households) |
| Revenue at risk | **$233,917** (392 high-risk HHs) |

---

## The 4 Customer Segments

| Segment | Size | Median Spend | Churn Rate | Priority |
|---------|------|-------------|------------|----------|
| Premium Loyalists | 546 | $5,464 | 1.1% | Protect & Delight |
| Occasional Shoppers | 1,008 | $2,050 | 3.1% | Grow & Re-engage |
| **Churn Risk** | **849** | **$524** | **15.7%** | **Immediate Action** |
| High-Value Deal Seekers | 96 | $5,047 | 1.0% | Curated Engagement |

---

## Pipeline

```
Phase 1  →  Data Quality Audit
Phase 2  →  Feature Engineering
Phase 3  →  Customer Segmentation (KMeans)
Phase 4  →  Churn Prediction (RF / XGBoost / Logistic Regression)
Phase 5  →  Retention Strategy & Business Narrative
```

### Phase 1 — Data Quality Audit
- Loaded all 8 CSV tables (2.5M+ transaction rows)
- Documented referential integrity: **68% of households had no demographics** — kept all households, used demographics as optional enrichment only
- Found 18,850 rows with SALES_VALUE ≤ 0 (returns/refunds) — isolated as a separate `return_rate` feature rather than dropping

### Phase 2 — Feature Engineering
Built a **one-row-per-household feature matrix** with 22 features across 4 groups:

- **RFM**: Recency (days since last purchase), Frequency (unique trips), Monetary (total spend), avg basket size, purchase rate
- **Basket behaviour**: avg/max items per basket, store loyalty score, time-of-day preference
- **Promotion sensitivity**: coupon redemption rate, % spend on discount, campaign targeting flag
- **Category affinity**: private label ratio, department diversity, unique products purchased

Key engineering decisions:
- `DAY` is an integer (1–711), not a calendar date — used as-is for recency/churn calculations
- Churn defined as **no purchase in the final 90 days** of the study window (day 622–711)
- Log-transformed 7 heavily right-skewed features before clustering

### Phase 3 — Customer Segmentation
- Tested k=2 to k=8 using elbow method and silhouette score
- Silhouette peaked at k=2 (mathematically optimal) but only produced "active vs inactive" — insufficient for business use
- **Overrode to k=4** to extract meaningful behavioural segments (justified by business interpretability)
- Applied PCA for 2D visualisation of cluster separation

### Phase 4 — Churn Prediction
- **Class imbalance**: 6.8% churn rate handled via SMOTE (inside pipeline, never touches test set) + `class_weight='balanced'`
- Compared 3 models: Logistic Regression (baseline), Random Forest, XGBoost
- **Random Forest selected** as best: highest test AUC (0.8793) and CV AUC (0.8686) with low variance (±0.021)
- Feature importance used to interpret top churn drivers — recency is the #1 signal

### Phase 5 — Retention Strategy
- Scored all 2,499 households with churn probability
- Assigned risk tiers: Low (<0.2), Medium (0.2–0.5), High (>0.5)
- Built priority action list ranked by `churn_proba × log(monetary)` — balances risk AND value
- Mapped segments to 2×2 retention matrix (value × churn risk)

---

## Retention Recommendations

| Segment | Action | Rationale |
|---------|--------|-----------|
| Churn Risk | Win-back coupon within 7 days of inactivity | 138 high-risk HHs, 15.7% churn rate — highest urgency |
| Occasional Shoppers | Loyalty multiplier + 10-day re-engagement trigger | Large segment (1,008), moderate risk, high upside |
| Premium Loyalists | VIP loyalty tier, surprise rewards, no heavy discounts | Low churn risk — protect margin, reward loyalty |
| High-Value Deal Seekers | Curated weekly deal bundles | High spend ($5,047), selective buyers — match their behaviour |

---

## Project Structure

```
├── data/
│   ├── transaction_data.csv        # raw (not in repo — download from Kaggle)
│   ├── product.csv
│   ├── hh_demographic.csv
│   ├── coupon.csv
│   ├── coupon_redempt.csv
│   ├── campaign_table.csv
│   ├── campaign_desc.csv
│   ├── causal_data.csv
│   ├── feature_matrix.csv          # Phase 2 output
│   ├── feature_matrix_segmented.csv # Phase 3 output
│   ├── final_scored.csv            # Phase 4 output
│   └── retention_action_list.csv   # Phase 5 output
│
├── phase1_data_audit.ipynb
├── phase2_feature_engineering.ipynb
├── phase3_segmentation.ipynb
├── phase4_churn_prediction.ipynb
├── phase5_retention_strategy.ipynb
│
├── app.py                    # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Dataset

**Dunnhumby — The Complete Journey**
Available on [Kaggle](https://www.kaggle.com/datasets/frtgnn/dunnhumby-the-complete-journey)

8 tables · 2,500 households · 2-year transaction window · Real retail data

---

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Data manipulation | pandas, numpy |
| Visualisation | matplotlib, seaborn, plotly |
| Machine learning | scikit-learn, xgboost |
| Class imbalance | imbalanced-learn (SMOTE) |
| Interpretability | SHAP |
| Dashboard | Streamlit |