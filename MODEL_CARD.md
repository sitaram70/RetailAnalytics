# Model Card – Retail Analytics

## Overview
This project ships two lightweight models for classroom use:

1. **RFM Segmentation (unsupervised)**  
   K-Means on customer-level Recency/​Frequency/​Monetary metrics computed from `transactions.csv`.

2. **Daily Revenue Forecast (supervised)**  
   Linear Regression using time and lagged revenue features; falls back to a naive mean-of-last-7 baseline if no model is trained.

Both are CPU-friendly and designed for small datasets.

---

## 1) RFM Segmentation

### Task
Group customers by recentness of purchase (R), purchase frequency (F), and monetary value (M) to support targeting.

### Data
- Built from `transactions.csv`.
- Aggregation per `customer_id`:
  - **Recency** = days since last purchase (`now - max(date)`).
  - **Frequency** = number of distinct `order_id`.
  - **Monetary** = sum of `revenue`.

### Model
- **Algorithm**: K-Means, `n_clusters=4`, `n_init=10`, fixed `random_state` for reproducibility.
- **Outputs**:
  - `models/rfm_segments.csv` with columns: `customer_id, recency, frequency, monetary, segment`.
  - UI shows a bar chart of counts per segment and a summary table with per-segment medians and an **auto label**:
    - Labels are derived by comparing medians to overall 33%/66% quantiles. Examples:
      - *Loyal / Champions*: recent + high F + high M  
      - *At-risk / Churn-prone*: stale + (low F or low M)  
      - *Potential Loyalists*: mid-recency + high F or high M  
      - *Regular*: otherwise

### Intended use / users
- Instructional segmentation and targeting demos; basic cohort comparisons.

### Limitations
- K-Means assumes spherical clusters and is sensitive to scaling/​outliers.
- Segment IDs (0..3) are arbitrary—interpret using medians.
- Segment composition changes on retrain; keep the random seed stable.

---

## 2) Daily Revenue Forecast

### Task
Predict daily revenue for the next **H** days.

### Features (LR model)
- `t` (time index), `rev_lag1` (yesterday), `rev_lag7` (7 days ago).  
  The API rolls forward autoregressively.

### Training artifact
- `models/daily_revenue_lr.joblib`.

### Inference behavior
- If the LR model exists, API returns `"model": "linear_regression_lags"`.
- Else, falls back to `"model": "naive-mean7"` (flat mean of last 7 days).

### Intended use / users
- Short-term planning demos (7–28 days).

### Limitations & failure modes
- With weak trend, autoregression can converge to a **flat line**.
- Doesn’t model weekly seasonality explicitly unless you extend it.
- Sensitive to holidays / promotions unless encoded.

### Possible improvements (student stretch goals)
- Add **day-of-week** one-hot features to LR.
- Provide a **seasonal-naive** option (repeat last 7 days).
- Evaluate with a rolling backtest: **MAE, RMSE, MAPE**.

---

## 3) Basket Analysis

### Task
Identify “frequently bought together” product pairs.

### Method
- Count co-occurrences of unique `product_id`s within the same `order_id`.
- API returns the top-N pairs by **count**.

### Limitations & improvements
- Raw counts are biased toward popular items; consider adding **support**, **confidence**, and **lift** and rank by lift × support.
- For large catalogs, computing all pairs can be heavy; filter by category/store first.

---

## 4) Anomaly Detection

### Method
- Rolling 14-day mean and std; flag days where `|z| ≥ threshold` (default 3.0).

### Notes
- With strong seasonality, consider a robust variant (median/MAD) or a longer window.

---

## 5) Compute & Environment

- CPU-only; runs comfortably on student laptops.
- Environment variables (API):
  - `DATA_DIR` – path to CSVs (default `data/raw`)
  - `MODEL_DIR` – path to model artifacts (default `models`)

---

## 6) Ethical & Operational Considerations
- Synthetic data by default; replace with real data only when you have permission.
- When using real data, ensure PII is removed and access is controlled.
- Segment labels are **heuristics** meant for instruction, not customer targeting in production.

---

## 7) Versioning & Maintenance
- Record simulator seed & parameters when you regenerate datasets.
- Tag releases when you change schema or model behavior.
