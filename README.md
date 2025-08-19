# AI-Driven Retail Analytics Dashboard (CPU-only)

Streamlit + FastAPI application for retail KPIs, customer segmentation (RFM), basket analysis (frequently bought together), anomaly detection, and simple forecasting. Designed to run on laptops with **CPU only**.

---

## Features
- **CSV Mode (local)**: Instant dashboard from `data/raw/daily_sales.csv`.
- **API Mode (http)**: UI talks to a FastAPI backend that serves metrics, segments, basket pairs, anomalies, and forecasts.
- **RFM segments**: K-Means on Recency, Frequency, Monetary with a labeled summary table.
- **Basket**: Top co-occurring product pairs.
- **Anomalies**: Rolling z-score (14-day window) with adjustable Z threshold.
- **Forecast**: Simple LR with lags (falls back to mean-7 if not trained).

---

## Project Structure (typical)
```
RetailAnalytics/
├─ src/
│  ├─ api.py                 # FastAPI service
│  ├─ simulate_retail.py     # synthetic data generator
│  ├─ preprocess_sales.py    # build daily KPI table
│  └─ train_models.py        # RFM + forecast training
├─ streamlit_app/
│  └─ app.py                 # Streamlit UI
├─ data/
│  └─ raw/                   # CSVs (products, stores, customers, visits, transactions, daily_sales)
├─ models/                   # saved models & segments
├─ README.md
├─ DATASETS.md
├─ MODEL_CARD.md
└─ requirements.txt
```

---

## Quickstart

### 1) Environment
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2) (Optional) Generate data and preprocess
```bash
python src/simulate_retail.py --out_dir data --start_date 2025-01-01 --days 120 --customers 500 --products 120 --stores 4
python src/preprocess_sales.py --transactions data/raw/transactions.csv --visits data/raw/visits.csv --out data/raw/daily_sales.csv
```

### 3) (Optional) Train models
```bash
python src/train_models.py --transactions data/raw/transactions.csv --daily data/raw/daily_sales.csv --model_out models
```

### 4) Run the backend API & UI
```bash
uvicorn src.api:app --reload --port 8000
streamlit run streamlit_app/app.py
```
- In the UI’s left sidebar, switch between **CSV Mode** and **API Mode** (`http://localhost:8000`).

---

## CSV vs API Mode

| Mode | Source | What works |
|---|---|---|
| **CSV Mode** | Reads `data/raw/daily_sales.csv` directly | KPIs, Revenue chart, Anomalies. Forecast uses a simple mean-7 baseline. |
| **API Mode** | Calls FastAPI endpoints | Everything: KPIs, Anomalies, **Segments**, **Basket**, and **trained Forecast**. |

---

## API (FastAPI) Endpoints

> Base URL: `http://localhost:8000`  
> Environment variables: `DATA_DIR` (default `data/raw`), `MODEL_DIR` (default `models`)

| Method | Path | What it returns | Key query params |
|---|---|---|---|
| GET | `/health` | `{ "status": "ok" }` | – |
| GET | `/metrics/overview` | Totals for revenue, orders, units, AOV, conversion, date range. | `date_from`, `date_to` (ISO date) |
| GET | `/metrics/daily` | Daily table `[ {date, revenue, orders, units, visits, aov, conversion}, … ]` | `date_from`, `date_to` |
| GET | `/rfm/segments` | `{counts: {seg: n, …}, sample: [...]}` | – |
| GET | `/rfm/summary` | Per-segment medians + auto label. | – |
| GET | `/basket/top_pairs` | `{pairs: [{p1, p2, count}, …]}` | `n` (top-N, default 10) |
| GET | `/alerts/anomalies` | Days with `|z| ≥ threshold` (14-day rolling). | `z` (default 3.0) |
| GET | `/forecast/daily` | `{model, pred:[{date, pred}, …]}` | `h` (horizon days, default 14) |
| POST | `/admin/train` | Trains RFM + forecast; saves to `models/`. | – |

---

## Common Tasks

**Retrain models from the UI**  
Segments tab → **“🔁 Train models now”**. (This calls `POST /admin/train`.)

**No anomalies listed?**  
Lower **Z threshold** to ~2.0.

**Forecast looks flat?**  
Train the LR model; otherwise the fallback is mean-7. (See MODEL_CARD for details.)

---

## Troubleshooting

- **API Mode errors**: verify `uvicorn` is running on `http://localhost:8000` and the URL in the sidebar matches.
- **Segments tab empty**: ensure `models/rfm_segments.csv` exists (use “Train models now”).
- **Import errors on Windows**: activate the venv (`.\.venv\Scripts\Activate.ps1`) before running python.

---

## License
Pick one (e.g., MIT) and add a `LICENSE` file if you plan to share publicly.
