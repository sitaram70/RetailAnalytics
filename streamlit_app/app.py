import os, requests, pandas as pd, numpy as np, streamlit as st
st.set_page_config(page_title='Retail Analytics Dashboard', layout='wide')
st.title('🛒 AI-Driven Retail Analytics Dashboard')
with st.sidebar:
    st.header('Data Source')
    mode = st.radio('Mode', ['CSV Mode (local files)', 'API Mode (http://localhost:8000)'])
    api_url = st.text_input('API base URL', 'http://localhost:8000')
def load_daily_csv(path='data/raw/daily_sales.csv'):
    try: return pd.read_csv(path, parse_dates=['date']).sort_values('date')
    except Exception as e: st.error(f'Could not read {path}: {e}'); return pd.DataFrame()

tab1, tab2, tab3, tab4 = st.tabs(['Overview','Segments','Basket','Forecast'])
with tab1:
    st.subheader('Overview KPIs')
    if mode.startswith('API'):
        ov = requests.get(f'{api_url}/metrics/overview', timeout=30).json()
        daily = pd.DataFrame(requests.get(f'{api_url}/metrics/daily', timeout=30).json())
    else:
        daily = load_daily_csv()
        ov = {'revenue': float(daily['revenue'].sum()), 'orders': int(daily['orders'].sum()), 'units': int(daily['units'].sum()), 'aov': float(daily['revenue'].sum()/max(1,daily['orders'].sum())), 'conversion': float(daily['orders'].sum()/max(1,daily['visits'].sum()))}
    c1,c2,c3,c4 = st.columns(4)
    c1.metric('Revenue', f"${ov['revenue']:.0f}"); c2.metric('Orders', f"{ov['orders']}"); c3.metric('AOV', f"${ov['aov']:.2f}"); c4.metric('Conversion', f"{ov['conversion']*100:.2f}%")
    st.markdown('#### Revenue over time'); st.line_chart(daily.set_index('date')['revenue'])
    st.markdown('#### Anomalies (z-score)'); z = st.slider('Z threshold', 1.5, 5.0, 3.0, 0.1)
    if mode.startswith('API'):
        arr = requests.get(f'{api_url}/alerts/anomalies', params={'z': z}, timeout=30).json(); st.dataframe(pd.DataFrame(arr))
    else:
        x=daily['revenue']; mu=x.rolling(14, min_periods=7).mean(); sd=x.rolling(14, min_periods=7).std().replace(0, np.nan); zscore=(x-mu)/sd
        st.dataframe(daily.assign(z=zscore).loc[abs(zscore)>=z, ['date','revenue','z']])
with tab2:
    st.subheader("RFM Segments")

    if mode.startswith("API"):
        # --- Train button ---
        if st.button("🔁 Train models now"):
            try:
                r = requests.post(f"{api_url}/admin/train", timeout=300).json()
                if r.get("ok"):
                    st.success("Models trained. Refresh tabs to see updates.")
                else:
                    st.error(r)
            except Exception as e:
                st.error(f"Training failed: {e}")

        # --- Segment counts + sample table ---
        try:
            seg = requests.get(f"{api_url}/rfm/segments", timeout=30).json()
            if "error" in seg:
                st.warning(seg["error"])
            else:
                a, b = st.columns([1, 2])
                a.bar_chart(pd.DataFrame.from_dict(seg["counts"], orient="index",
                                                   columns=["count"]))
                b.dataframe(pd.DataFrame(seg["sample"]))
        except Exception as e:
            st.error(f"Could not fetch segments: {e}")

        # --- NEW: medians & auto labels table ---
        st.markdown("#### Segment medians & labels")
        try:
            summ = requests.get(f"{api_url}/rfm/summary", timeout=30).json()
            if "summary" in summ:
                df_sum = (pd.DataFrame(summ["summary"])
                          .rename(columns={
                              "recency": "recency_med",
                              "frequency": "frequency_med",
                              "monetary": "monetary_med"
                          }))
                st.dataframe(
                    df_sum.style.format({
                        "recency_med": "{:.0f}",
                        "frequency_med": "{:.0f}",
                        "monetary_med": "${:,.0f}",
                        "count": "{:,.0f}",
                    })
                )
            else:
                st.info("No summary available yet. Train models and try again.")
        except Exception as e:
            st.error(f"Could not fetch summary: {e}")

    else:
        st.info("Switch to API Mode to view RFM segments.")

with tab3:
    st.subheader('Top Co-occurring Product Pairs'); n = st.slider('How many pairs?', 5, 30, 10)
    if mode.startswith('API'):
        js = requests.get(f'{api_url}/basket/top_pairs', params={'n': n}, timeout=30).json(); st.dataframe(pd.DataFrame(js.get('pairs', [])))
    else: st.info('Use API Mode to compute across orders.')
with tab4:
    st.subheader('Forecast daily revenue'); h = st.slider('Horizon (days)', 7, 42, 14)
    if mode.startswith('API'):
        js = requests.get(f'{api_url}/forecast/daily', params={'h': h}, timeout=30).json(); pred = pd.DataFrame(js['pred']); st.line_chart(pred.set_index('date')['pred']); st.caption(f"Model: {js['model']}")
    else:
        daily = load_daily_csv(); mean = daily.tail(7)['revenue'].mean(); future = pd.DataFrame({'date': pd.date_range(daily['date'].max()+pd.Timedelta(days=1), periods=h).date, 'pred': mean}); st.line_chart(future.set_index('date')['pred']); st.caption('Model: naive-mean7')

# ---- Drill-downs ----
st.markdown("#### Revenue by category")
if mode.startswith("API"):
    # date range selector based on available daily data
    dmin = pd.to_datetime(daily["date"].min()).date()
    dmax = pd.to_datetime(daily["date"].max()).date()
    dfrom, dto = st.date_input("Date range", value=(dmin, dmax))
    store_id = st.text_input("Store ID (optional)", value="")
    region = st.text_input("Region (optional)", value="")
    params = {"date_from": str(dfrom), "date_to": str(dto)}
    if store_id.strip(): params["store_id"] = store_id.strip()
    if region.strip(): params["region"] = region.strip()
    cat = requests.get(f"{api_url}/metrics/by_category", params=params, timeout=30).json()
    df_cat = pd.DataFrame(cat)
    if not df_cat.empty:
        st.bar_chart(df_cat.set_index("category")["revenue"])
    else:
        st.info("No matching data for the selected filters.")
else:
    st.info("Switch to API Mode to use drill-downs.")

