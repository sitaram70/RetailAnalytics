#!/usr/bin/env python3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd, numpy as np, os
app=FastAPI(title='Retail Analytics API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])
DATA=os.environ.get('DATA_DIR','data/raw'); MODEL=os.environ.get('MODEL_DIR','models')

@app.get('/health')
def health(): return {'status':'ok'}

@app.get('/metrics/overview')
def overview(date_from: Optional[str]=None, date_to: Optional[str]=None):
    d=pd.read_csv(os.path.join(DATA,'daily_sales.csv'), parse_dates=['date']).sort_values('date')
    if date_from: d=d[d['date']>=pd.to_datetime(date_from)]
    if date_to: d=d[d['date']<=pd.to_datetime(date_to)]
    if d.empty: return {'error':'no data'}
    return {'revenue': float(d['revenue'].sum()), 'orders': int(d['orders'].sum()), 'units': int(d['units'].sum()), 'aov': float(d['revenue'].sum()/max(1,d['orders'].sum())), 'conversion': float(d['orders'].sum()/max(1,d['visits'].sum())), 'date_min': str(d['date'].min().date()), 'date_max': str(d['date'].max().date())}

@app.get('/metrics/daily')
def daily(date_from: Optional[str]=None, date_to: Optional[str]=None):
    d=pd.read_csv(os.path.join(DATA,'daily_sales.csv'), parse_dates=['date']).sort_values('date')
    if date_from: d=d[d['date']>=pd.to_datetime(date_from)]
    if date_to: d=d[d['date']<=pd.to_datetime(date_to)]
    return d.to_dict(orient='records')

@app.get('/rfm/segments')
def rfm():
    p=os.path.join(MODEL,'rfm_segments.csv')
    if not os.path.exists(p): return {'error':'no rfm_segments.csv found; run training.'}
    df=pd.read_csv(p); return {'counts': df['segment'].value_counts().sort_index().to_dict(), 'sample': df.head(20).to_dict(orient='records')}

@app.get('/basket/top_pairs')
def pairs(n:int=10):
    tx=pd.read_csv(os.path.join(DATA,'transactions.csv')); pairs={}
    for oid, grp in tx.groupby('order_id'):
        items=sorted(grp['product_id'].unique())
        for i in range(len(items)):
            for j in range(i+1,len(items)):
                key=(int(items[i]),int(items[j])); pairs[key]=pairs.get(key,0)+1
    out=sorted([{'p1':a,'p2':b,'count':c} for (a,b),c in pairs.items()], key=lambda x:x['count'], reverse=True)[:n]
    return {'pairs': out}

@app.get('/alerts/anomalies')
def anomalies(z: float = 3.0):
    d=pd.read_csv(os.path.join(DATA,'daily_sales.csv'), parse_dates=['date']).sort_values('date')
    x=d['revenue']; mu=x.rolling(14, min_periods=7).mean(); sd=x.rolling(14, min_periods=7).std().replace(0, np.nan); zscore=(x-mu)/sd; d['z']=zscore
    return d[abs(d['z'])>=z][['date','revenue','z']].to_dict(orient='records')

@app.get('/forecast/daily')
def forecast(h:int=14):
    import joblib
    d=pd.read_csv(os.path.join(DATA,'daily_sales.csv'), parse_dates=['date']).sort_values('date')
    mp=os.path.join(MODEL,'daily_revenue_lr.joblib')
    if (not os.path.exists(mp)) or (len(d)<10):
        mean=float(d.tail(7)['revenue'].mean()); last=d['date'].max()
        return {'model':'naive-mean7','pred':[{'date': str((last+pd.Timedelta(days=i)).date()), 'pred': mean} for i in range(1,h+1)]}
    lr=joblib.load(mp); df=d.copy(); df['t']=range(len(df)); 
    for lag in [1,7]: df[f'rev_lag{lag}']=df['revenue'].shift(lag)
    last=df['date'].max(); last_t=df['t'].iloc[-1]; lag1=df['revenue'].iloc[-1]; lag7=df['revenue'].iloc[-7] if len(df)>=7 else df['revenue'].iloc[0]
    preds=[]; 
    for i in range(1,h+1):
        t=last_t+i; import numpy as np; y=float(lr.predict([[t, lag1, lag7]])[0]); preds.append({'date': str((last+pd.Timedelta(days=i)).date()), 'pred': y}); lag7 = lag1 if i>=6 else lag7; lag1 = y
    return {'model':'linear_regression_lags','pred': preds}

from typing import Optional

@app.get("/metrics/by_category")
def by_category(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    store_id: Optional[int] = None,
    region: Optional[str] = None,
):
    import os, pandas as pd
    tx = pd.read_csv(os.path.join(DATA, "transactions.csv"), parse_dates=["date"])
    prod = pd.read_csv(os.path.join(DATA, "products.csv"))
    stores = pd.read_csv(os.path.join(DATA, "stores.csv"))

    df = tx.merge(prod[["product_id", "category"]], on="product_id", how="left")
    if store_id is not None:
        df = df[df["store_id"] == int(store_id)]
    if region:
        df = df.merge(stores[["store_id", "region"]], on="store_id", how="left")
        df = df[df["region"] == region]
    if date_from:
        df = df[df["date"] >= pd.to_datetime(date_from)]
    if date_to:
        df = df[df["date"] <= pd.to_datetime(date_to)]

    out = (
        df.groupby("category", as_index=False)["revenue"]
        .sum()
        .sort_values("revenue", ascending=False)
    )
    return out.to_dict(orient="records")

@app.post("/admin/train")
def admin_train():
    import subprocess, sys, os
    tx = os.path.join(DATA, "transactions.csv")
    daily = os.path.join(DATA, "daily_sales.csv")
    cmd = [sys.executable, "src/train_models.py", "--transactions", tx, "--daily", daily, "--model_out", MODEL]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {"ok": p.returncode == 0, "stdout": p.stdout[-500:], "stderr": p.stderr[-500:]}

from typing import Optional

@app.get("/rfm/summary")
def rfm_summary():
    """
    Returns per-segment medians of Recency / Frequency / Monetary, counts,
    and a simple auto label for each segment.
    """
    import os, pandas as pd
    p = os.path.join(MODEL, "rfm_segments.csv")
    if not os.path.exists(p):
        return {"error": "no rfm_segments.csv found; run /admin/train or train_models.py"}

    df = pd.read_csv(p)

    # Per-segment medians + counts
    med = (
        df.groupby("segment")
          .agg(
              recency=("recency", "median"),
              frequency=("frequency", "median"),
              monetary=("monetary", "median"),
              count=("customer_id", "count"),
          )
          .reset_index()
    )

    # Quantiles over the whole customer set to define bins (3-level)
    r33, r66 = df["recency"].quantile([0.33, 0.66]).tolist()
    f33, f66 = df["frequency"].quantile([0.33, 0.66]).tolist()
    m33, m66 = df["monetary"].quantile([0.33, 0.66]).tolist()

    def label_row(r, f, m):
        # recency: lower is better (more recent)
        rcat = "new" if r <= r33 else ("warm" if r <= r66 else "stale")
        fcat = "high" if f > f66 else ("mid" if f > f33 else "low")
        mcat = "high" if m > m66 else ("mid" if m > m33 else "low")

        # simple, readable names
        if rcat == "new" and fcat == "high" and mcat == "high":
            return "Loyal / Champions"
        if rcat == "stale" and (fcat != "high" or mcat != "high"):
            return "At-risk / Churn-prone"
        if rcat == "new" and (fcat == "low" or mcat == "low"):
            return "New / Onboarding"
        if fcat == "high" or mcat == "high":
            return "Potential Loyalists"
        return "Regular"

    med["label"] = med.apply(
        lambda r: label_row(r["recency"], r["frequency"], r["monetary"]), axis=1
    )

    # Return tidy rows
    return {
        "summary": med.sort_values("segment")[
            ["segment", "count", "recency", "frequency", "monetary", "label"]
        ].to_dict(orient="records")
    }
