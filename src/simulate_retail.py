#!/usr/bin/env python3
import argparse, random, numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
CATEGORIES=['Grocery','Home','Electronics','Beauty','Apparel','Outdoor','Toys']
def _font(sz=28, bold=True):
    try: return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', sz)
    except: return ImageFont.load_default()
def make_img(path, text):
    img=Image.new('RGB',(320,240),(80,120,160)); d=ImageDraw.Draw(img); f=_font()
    w=d.textlength(text,font=f); d.text(((320-w)/2,100),text,font=f,fill=(255,255,255)); img.save(path)
def simulate(out_dir='data', start_date='2025-01-01', days=120, customers=500, products=120, stores=4):
    out=Path(out_dir); (out/'images'/'products').mkdir(parents=True, exist_ok=True)
    # products
    prows=[]; 
    for i in range(1, products+1):
        cat=random.choice(CATEGORIES); name=f"{cat[:3].upper()}-{i:03d}"; price=round(random.uniform(3,200),2)
        p=out/'images'/'products'/f"{name}.png"; make_img(p, name); prows.append([i,name,cat,price,str(p)])
    pd.DataFrame(prows, columns=['product_id','name','category','base_price','image_path']).to_csv(out/'raw'/'products.csv', index=False)
    # stores
    cities=['Springfield','Fairview','Franklin','Greenville','Madison','Georgetown']; regions=['North','South','East','West']
    srows=[[s, random.choice(cities), random.choice(regions), random.randint(60,180)] for s in range(1, stores+1)]
    pd.DataFrame(srows, columns=['store_id','city','region','size_index']).to_csv(out/'raw'/'stores.csv', index=False)
    # customers
    start=datetime.fromisoformat(start_date); crows=[]
    for c in range(1, customers+1):
        signup=(start - timedelta(days=random.randint(0,365))).date().isoformat()
        crows.append([c, signup, random.choice(CATEGORIES), random.choice(['L','M','H'])])
    pd.DataFrame(crows, columns=['customer_id','signup_date','pref_category','income_bracket']).to_csv(out/'raw'/'customers.csv', index=False)
    # visits & transactions
    dates=[start+timedelta(days=i) for i in range(days)]
    dfp=pd.read_csv(out/'raw'/'products.csv')
    promo_products=set(random.sample(list(dfp['product_id']), max(1, products//10)))
    promo_days=set(random.sample(range(days), max(1, days//6)))
    vis_rows=[]; tx_rows=[]; order_id=1
    for di, day in enumerate(dates):
        wknd=1.25 if day.weekday()>=5 else 1.0; season = 1.0 + (0.15 if day.month in [11,12] else (0.10 if day.month in [6,7] else 0.0))
        for st in srows:
            stid=st[0]; base_vis=st[3]
            visits=int(base_vis * (0.25*season*wknd) * random.uniform(0.8,1.2)); vis_rows.append([day.date().isoformat(), stid, visits])
            is_promo = di in promo_days; conv = 0.05 + (0.02 if is_promo else 0.0) + random.uniform(-0.003,0.01)
            orders=max(0, int(visits*conv))
            for _ in range(orders):
                cust=random.randint(1, customers); k=max(1,min(6,int(np.random.poisson(1.2)+1))); items=random.sample(list(dfp['product_id']), k)
                for pid in items:
                    base=float(dfp.loc[dfp['product_id']==pid,'base_price'].iloc[0])
                    disc=round(random.uniform(0.10,0.30),2) if (is_promo and pid in promo_products) else 0.0
                    qty=max(1,min(5,int(np.random.poisson(1)+1)))
                    price=round(base*(1.0-disc),2); rev=round(price*qty,2)
                    tx_rows.append([order_id, day.date().isoformat(), stid, cust, int(pid), qty, price, disc, rev])
                order_id+=1
    pd.DataFrame(vis_rows, columns=['date','store_id','visits']).to_csv(out/'raw'/'visits.csv', index=False)
    pd.DataFrame(tx_rows, columns=['order_id','date','store_id','customer_id','product_id','quantity','unit_price','discount','revenue']).to_csv(out/'raw'/'transactions.csv', index=False)
if __name__=='__main__':
    import argparse; ap=argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='data'); ap.add_argument('--start_date', default='2025-01-01')
    ap.add_argument('--days', type=int, default=120); ap.add_argument('--customers', type=int, default=500)
    ap.add_argument('--products', type=int, default=120); ap.add_argument('--stores', type=int, default=4)
    a=ap.parse_args(); simulate(a.out_dir, a.start_date, a.days, a.customers, a.products, a.stores)
