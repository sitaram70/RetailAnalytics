#!/usr/bin/env python3
#!/usr/bin/env python3
import argparse
import pandas as pd

def preprocess(transactions_csv, visits_csv, out_csv):
    tx = pd.read_csv(transactions_csv, parse_dates=['date'])
    vs = pd.read_csv(visits_csv, parse_dates=['date'])

    # ✅ Ensure both keys are datetime64[ns] at day precision
    tx['date'] = pd.to_datetime(tx['date']).dt.floor('D')
    vs['date'] = pd.to_datetime(vs['date']).dt.floor('D')

    daily = tx.groupby('date').agg(
        revenue=('revenue','sum'),
        orders=('order_id','nunique'),
        units=('quantity','sum')
    ).reset_index()

    v_daily = vs.groupby('date', as_index=False)['visits'].sum()

    out = daily.merge(v_daily, on='date', how='left')
    out['aov'] = out['revenue'] / out['orders'].clip(lower=1)
    out['conversion'] = out['orders'] / out['visits'].clip(lower=1)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {len(out)} rows")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--transactions", default="data/raw/transactions.csv")
    ap.add_argument("--visits", default="data/raw/visits.csv")
    ap.add_argument("--out", default="data/raw/daily_sales.csv")
    a = ap.parse_args()
    preprocess(a.transactions, a.visits, a.out)

