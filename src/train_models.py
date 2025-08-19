#!/usr/bin/env python3
import argparse, os, pandas as pd, numpy as np, joblib
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
def build_rfm(tx_csv, out_csv, n_segments=4, random_state=42):
    tx=pd.read_csv(tx_csv, parse_dates=['date']); now=tx['date'].max() + pd.Timedelta(days=1)
    agg=tx.groupby('customer_id').agg(recency=('date', lambda s: (now - s.max()).days), frequency=('order_id','nunique'), monetary=('revenue','sum')).reset_index()
    feats=agg[['recency','frequency','monetary']].values.astype(float); km=KMeans(n_clusters=n_segments, n_init=10, random_state=random_state).fit(feats)
    agg['segment']=km.labels_; agg.to_csv(out_csv, index=False); return km
def build_forecast(daily_csv, model_out):
    df=pd.read_csv(daily_csv, parse_dates=['date']).sort_values('date'); df['t']=range(len(df))
    for lag in [1,7]: df[f'rev_lag{lag}']=df['revenue'].shift(lag); df=df.dropna()
    X=df[['t','rev_lag1','rev_lag7']].values; y=df['revenue'].values; lr=LinearRegression().fit(X,y)
    os.makedirs(model_out, exist_ok=True); joblib.dump(lr, os.path.join(model_out,'daily_revenue_lr.joblib')); print('Saved forecast model')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--transactions', default='data/raw/transactions.csv'); ap.add_argument('--daily', default='data/raw/daily_sales.csv'); ap.add_argument('--model_out', default='models')
    a=ap.parse_args(); os.makedirs(a.model_out, exist_ok=True); import joblib; km=build_rfm(a.transactions, os.path.join(a.model_out,'rfm_segments.csv')); joblib.dump(km, os.path.join(a.model_out,'rfm_kmeans.joblib')); build_forecast(a.daily, a.model_out)
