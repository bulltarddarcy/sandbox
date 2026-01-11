# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- MODULE IMPORTS ---
# All logic now resides in utils_darcy
import utils_darcy as ud 

# ==========================================
# APP 1: DATABASE
# ==========================================
def run_database_app(df):
    st.title("📂 Database")
    max_data_date = ud.get_max_trade_date(df)
    
    if 'saved_db_ticker' not in st.session_state: st.session_state.saved_db_ticker = ""
    if 'saved_db_start' not in st.session_state: st.session_state.saved_db_start = max_data_date
    if 'saved_db_end' not in st.session_state: st.session_state.saved_db_end = max_data_date
    if 'saved_db_exp' not in st.session_state: st.session_state.saved_db_exp = (date.today() + timedelta(days=365))
    if 'saved_db_inc_cb' not in st.session_state: st.session_state.saved_db_inc_cb = True
    if 'saved_db_inc_ps' not in st.session_state: st.session_state.saved_db_inc_ps = True
    if 'saved_db_inc_pb' not in st.session_state: st.session_state.saved_db_inc_pb = True

    def save_db_state(key, saved_key):
        st.session_state[saved_key] = st.session_state[key]
    
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        db_ticker = st.text_input("Ticker (blank=all)", value=st.session_state.saved_db_ticker, key="db_ticker_input", on_change=save_db_state, args=("db_ticker_input", "saved_db_ticker")).strip().upper()
    with c2: start_date = st.date_input("Trade Start Date", value=st.session_state.saved_db_start, key="db_start", on_change=save_db_state, args=("db_start", "saved_db_start"))
    with c3: end_date = st.date_input("Trade End Date", value=st.session_state.saved_db_end, key="db_end", on_change=save_db_state, args=("db_end", "saved_db_end"))
    with c4:
        db_exp_end = st.date_input("Expiration Range (end)", value=st.session_state.saved_db_exp, key="db_exp", on_change=save_db_state, args=("db_exp", "saved_db_exp"))
    
    ot1, ot2, ot3, ot_pad = st.columns([1.5, 1.5, 1.5, 5.5])
    with ot1: inc_cb = st.checkbox("Calls Bought", value=st.session_state.saved_db_inc_cb, key="db_inc_cb", on_change=save_db_state, args=("db_inc_cb", "saved_db_inc_cb"))
    with ot2: inc_ps = st.checkbox("Puts Sold", value=st.session_state.saved_db_inc_ps, key="db_inc_ps", on_change=save_db_state, args=("db_inc_ps", "saved_db_inc_ps"))
    with ot3: inc_pb = st.checkbox("Puts Bought", value=st.session_state.saved_db_inc_pb, key="db_inc_pb", on_change=save_db_state, args=("db_inc_pb", "saved_db_inc_pb"))
    
    # Filter
    f = df.copy()
    if db_ticker: f = f[f[ud.COL_SYMBOL].astype(str).str.upper().eq(db_ticker)]
    if start_date: f = f[f[ud.COL_TRADE_DATE].dt.date >= start_date]
    if end_date: f = f[f[ud.COL_TRADE_DATE].dt.date <= end_date]
    if db_exp_end: f = f[f["EXPIRY_DT"].dt.date <= db_exp_end]
    
    allowed_types = []
    if inc_cb: allowed_types.append("Calls Bought")
    if inc_pb: allowed_types.append("Puts Bought")
    if inc_ps: allowed_types.append("Puts Sold")
    f = f[f[ud.COL_ORDER_TYPE].isin(allowed_types)]
    
    if f.empty:
        st.warning("No data found matching these filters.")
        return
        
    f = f.sort_values(by=[ud.COL_TRADE_DATE, ud.COL_SYMBOL], ascending=[False, True])
    
    display_cols = [ud.COL_TRADE_DATE, ud.COL_ORDER_TYPE, ud.COL_SYMBOL, ud.COL_STRIKE, ud.COL_EXPIRY, ud.COL_CONTRACTS, ud.COL_DOLLARS]
    f_display = f[display_cols].copy()
    
    f_display[ud.COL_TRADE_DATE] = f_display[ud.COL_TRADE_DATE].dt.strftime("%d %b %y")
    # Expiry is string in raw, but we want fmt
    
    def highlight_db_order_type(val):
        if val in ["Calls Bought", "Puts Sold"]: return 'background-color: rgba(113, 210, 138, 0.15); color: #71d28a; font-weight: 600;'
        elif val == "Puts Bought": return 'background-color: rgba(242, 156, 160, 0.15); color: #f29ca0; font-weight: 600;'
        return ''
        
    st.subheader("Non-Expired Trades")
    st.dataframe(
        f_display.style.format({ud.COL_DOLLARS: "${:,.0f}", ud.COL_CONTRACTS: "{:,.0f}"})
        .applymap(highlight_db_order_type, subset=[ud.COL_ORDER_TYPE]), 
        use_container_width=True, hide_index=True, height=ud.get_table_height(f_display, max_rows=30)
    )
    st.markdown("<br><br><br>", unsafe_allow_html=True)

# ==========================================
# APP 2: RANKINGS
# ==========================================
def run_rankings_app(df):
    st.title("🏆 Rankings")
    max_data_date = ud.get_max_trade_date(df)
    start_default = max_data_date - timedelta(days=14)
    
    if 'saved_rank_start' not in st.session_state: st.session_state.saved_rank_start = start_default
    if 'saved_rank_end' not in st.session_state: st.session_state.saved_rank_end = max_data_date
    
    c1, c2 = st.columns(2)
    with c1: rank_start = st.date_input("Start", value=st.session_state.saved_rank_start)
    with c2: rank_end = st.date_input("End", value=st.session_state.saved_rank_end)
    
    # Calculate Smart Money Scores using Utils
    top_bulls, top_bears, valid_data = ud.calculate_smart_money_score(
        df, rank_start, rank_end, ud.MC_THRESHOLDS["10B"], False, 20
    )
    
    st.subheader("Bullish Smart Money")
    st.dataframe(top_bulls, use_container_width=True)

    st.subheader("Bearish Smart Money")
    st.dataframe(top_bears, use_container_width=True)


# ==========================================
# APP 3: PIVOT TABLES
# ==========================================
def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    max_data_date = ud.get_max_trade_date(df)
    
    col_filters, col_calculator = st.columns([1, 1], gap="medium")
    
    with col_filters:
        td_start = st.date_input("Start", value=max_data_date)
        td_end = st.date_input("End", value=max_data_date)
        ticker = st.text_input("Ticker").strip().upper()
        
    # Use the helper from Utils
    f = ud.filter_pivot_data(df, ticker, td_start, td_end, 0, 0, "All")
    
    if f.empty:
        st.warning("No trades.")
        return
        
    st.dataframe(f[[ud.COL_SYMBOL, ud.COL_ORDER_TYPE, ud.COL_STRIKE, ud.COL_DOLLARS]], use_container_width=True)


# ==========================================
# APP 4: EMA DISTANCE (Refactored)
# ==========================================
def run_ema_distance_app(df_global):
    st.title("📏 EMA Distance Analysis")
    
    col_in1, _ = st.columns([1, 2])
    with col_in1:
        ticker = st.text_input("Ticker", value="QQQ").upper().strip()
        
    if not ticker: return

    with st.spinner(f"Crunching data for {ticker}..."):
        # 1. Fetch History using optimized loader
        t_map = ud.load_ticker_map()
        df = ud.fetch_history_optimized(ticker, t_map)
        
        if df is None or df.empty:
            st.error("No data found.")
            return

        # 2. Calculate Distances (Using Uppercase Columns)
        close = df[ud.COL_CLOSE]
        
        # Calculate needed MAs if missing (Parquet might have some, but we need specific ones)
        if ud.COL_EMA8 not in df.columns: df[ud.COL_EMA8] = close.ewm(span=8).mean()
        if ud.COL_EMA21 not in df.columns: df[ud.COL_EMA21] = close.ewm(span=21).mean()
        df[ud.COL_SMA50] = close.rolling(50).mean()
        df[ud.COL_SMA100] = close.rolling(100).mean()
        if ud.COL_SMA200 not in df.columns: df[ud.COL_SMA200] = close.rolling(200).mean()
        
        # Gaps
        df['Dist_8'] = ((close - df[ud.COL_EMA8]) / df[ud.COL_EMA8]) * 100
        df['Dist_50'] = ((close - df[ud.COL_SMA50]) / df[ud.COL_SMA50]) * 100
        
        current_gap = df['Dist_50'].iloc[-1]
        
        st.metric(f"Current Gap vs 50SMA", f"{current_gap:.2f}%")
        
        # Chart
        chart_data = df[[ud.COL_DATE, 'Dist_50']].tail(500)
        st.bar_chart(chart_data, x=ud.COL_DATE, y='Dist_50')

# ==========================================
# APP 5: SEASONALITY
# ==========================================
def run_seasonality_app(df_global):
    st.title("📅 Seasonality & Recurring Trends")
    
    tab1, tab2 = st.tabs(["📊 Single Ticker Analysis", "🔭 Forward Returns Scanner"])
    
    # --- TAB 1: SINGLE TICKER ---
    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            ticker = st.text_input("Ticker Symbol", value="SPY", key="seas_ticker").upper().strip()
        with c2:
            lookback = st.slider("Lookback Years", 5, 20, 10, key="seas_lookback")
            
        if not ticker: return
            
        # Load Data
        t_map = ud.load_ticker_map()
        df = ud.fetch_history_optimized(ticker, t_map)
        
        if df is None or df.empty:
            st.error("No data found.")
            return

        # Prepare Monthly Stats (Logic moved to simple pandas ops here)
        df[ud.COL_DATE] = pd.to_datetime(df[ud.COL_DATE])
        df.set_index(ud.COL_DATE, inplace=True)
        
        # Resample to Monthly Returns
        monthly_closes = df[ud.COL_CLOSE].resample('M').last()
        monthly_ret = monthly_closes.pct_change()
        
        # Filter by years
        start_date = df.index.min() + pd.DateOffset(years=lookback) if len(df) > 0 else None
        if start_date:
            monthly_ret = monthly_ret[monthly_ret.index >= start_date]

        if monthly_ret.empty:
            st.warning("Not enough data for this lookback.")
            return

        # Pivot Table: Year vs Month
        m_df = monthly_ret.to_frame(name='Return')
        m_df['Year'] = m_df.index.year
        m_df['Month'] = m_df.index.strftime('%b')
        m_df['Month_Num'] = m_df.index.month
        
        pivot_ret = m_df.pivot_table(values='Return', index='Year', columns='Month_Num', aggfunc='sum')
        pivot_ret.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        
        # Heatmap Visualization
        st.subheader(f"Monthly Returns ({lookback}Y Lookback)")
        st.dataframe(pivot_ret.style.format("{:.1%}")
                     .background_gradient(cmap='RdYlGn', vmin=-0.1, vmax=0.1), 
                     use_container_width=True)
        
        # Aggregate Win Rate
        win_rates = (pivot_ret > 0).mean() * 100
        avg_rets = pivot_ret.mean() * 100
        
        stats_df = pd.DataFrame({
            "Win Rate %": win_rates,
            "Avg Return %": avg_rets
        }).T
        
        st.subheader("Aggregate Stats by Month")
        st.dataframe(stats_df.style.format("{:.1f}%").background_gradient(cmap='Blues'), use_container_width=True)

    # --- TAB 2: SCANNER ---
    with tab2:
        st.subheader("Find Tickers with High Win-Rates for UPCOMING dates")
        
        col_scan1, col_scan2 = st.columns(2)
        with col_scan1:
            scan_date = st.date_input("Target Date", value=date.today())
        with col_scan2:
            min_wr = st.slider("Min Win Rate (%)", 50, 90, 70)

        if st.button("Run Seasonality Scan"):
            t_map = ud.load_ticker_map()
            tickers_to_scan = list(t_map.keys()) # Or filter specific list
            
            results = []
            progress = st.progress(0)
            
            # Using ThreadPool from Utils would be cleaner, but keeping explicit here for progress bar updates
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(ud.calculate_forward_seasonality, t, t_map, scan_date, lookback): t for t in tickers_to_scan[:50]} # Limit to 50 for demo speed
                
                completed = 0
                for future in as_completed(futures):
                    res, _ = future.result()
                    if res and res['21d_WR'] >= min_wr:
                        results.append(res)
                    completed += 1
                    progress.progress(completed / len(futures))
            
            progress.empty()
            
            if results:
                res_df = pd.DataFrame(results)
                st.dataframe(res_df.style.format("{:.1f}"), use_container_width=True)
            else:
                st.info("No tickers found matching criteria.")

# ==========================================
# APP 6: RSI SCANNER
# ==========================================
def run_rsi_scanner_app():
    st.title("📡 RSI Scanner")
    
    # Configuration
    c1, c2, c3 = st.columns(3)
    with c1:
        mode = st.radio("Mode", ["Oversold (<30)", "Overbought (>70)"])
    with c2:
        rsi_thresh = st.number_input("RSI Threshold", value=30 if "Oversold" in mode else 70)
    with c3:
        min_mc = st.selectbox("Min Market Cap", options=ud.MC_THRESHOLDS.keys(), index=2)
    
    # Load Data (Using Parquet Config)
    pq_config = ud.get_parquet_config()
    if not pq_config:
        st.error("Parquet Config missing.")
        return
        
    # We will iterate through configured files to find opportunities
    results = []
    
    with st.spinner("Scanning Market Data..."):
        # Helper to process one file
        def process_file(key):
            df = ud.load_parquet_and_clean(key)
            if df is None or df.empty: return []
            
            # Filter by Date (latest)
            max_date = df[ud.COL_DATE].max()
            current = df[df[ud.COL_DATE] == max_date].copy()
            
            # Filter by RSI (Using Fixed Column)
            if ud.COL_RSI not in current.columns: return []
            
            if "Oversold" in mode:
                hits = current[current[ud.COL_RSI] < rsi_thresh].copy()
            else:
                hits = current[current[ud.COL_RSI] > rsi_thresh].copy()
                
            if hits.empty: return []
            
            hits['Source'] = key
            return hits.to_dict('records')

        # Run parallel scan
        with ThreadPoolExecutor() as exc:
            futures = [exc.submit(process_file, k) for k in pq_config.keys()]
            for f in as_completed(futures):
                res = f.result()
                if res: results.extend(res)
    
    if not results:
        st.info("No stocks found.")
        return
        
    df_res = pd.DataFrame(results)
    
    # Filter by Market Cap
    mc_limit = ud.MC_THRESHOLDS[min_mc]
    if mc_limit > 0:
        tickers = df_res[ud.COL_SYMBOL].unique()
        caps = ud.fetch_market_caps_batch(tickers)
        df_res['MC'] = df_res[ud.COL_SYMBOL].map(caps)
        df_res = df_res[df_res['MC'] >= mc_limit]
        
    st.dataframe(
        df_res[[ud.COL_SYMBOL, ud.COL_DATE, ud.COL_CLOSE, ud.COL_RSI, 'Source']],
        use_container_width=True
    )

# ==========================================
# APP 7: PRICE DIVERGENCES
# ==========================================
def run_price_divergences_app():
    st.title("📉 Price Divergences (RSI)")
    
    # Controls
    c1, c2, c3 = st.columns(3)
    with c1:
        div_ticker = st.text_input("Ticker", value="QQQ", key="div_ticker").upper().strip()
    with c2:
        lookback = st.number_input("Lookback Days", value=90)
    with c3:
        timeframe = st.selectbox("Timeframe", ["Daily", "Weekly"])
        
    if not div_ticker: return
    
    # Load History
    t_map = ud.load_ticker_map()
    df = ud.fetch_history_optimized(div_ticker, t_map)
    
    if df is None or df.empty:
        st.error("No data.")
        return
        
    # Prepare Data (Split Daily/Weekly)
    df_d, df_w = ud.prepare_data(df)
    
    target_df = df_w if timeframe == "Weekly" and df_w is not None else df_d
    
    if target_df is None:
        st.error("Weekly data not available.")
        return
        
    # Find Divergences
    divs = ud.find_divergences(target_df, div_ticker, timeframe, lookback_period=lookback)
    
    if not divs:
        st.info(f"No {timeframe} divergences found in last {lookback} periods.")
        
        # Plotting Price anyway
        st.line_chart(target_df.set_index(ud.COL_DATE if ud.COL_DATE in target_df.columns else target_df.index)[ud.COL_CLOSE])
        return
        
    # Display Results
    st.success(f"Found {len(divs)} divergences.")
    res_df = pd.DataFrame(divs)
    
    st.dataframe(
        res_df[['Type', 'Signal_Date_ISO', 'Price2', 'RSI2', 'Price1', 'RSI1']],
        use_container_width=True
    )
    
    # Visuals: Add Markers to chart
    chart_base = alt.Chart(target_df.tail(lookback*2)).encode(x=f'{ud.COL_DATE}:T')
    
    line = chart_base.mark_line().encode(y=ud.COL_CLOSE)
    
    points = chart_base.mark_point(color='red', size=100, shape='triangle-down').encode(
        y=ud.COL_CLOSE,
        tooltip=[ud.COL_DATE, ud.COL_CLOSE]
    ).transform_filter(
        alt.FieldOneOfPredicate(field=ud.COL_DATE, oneOf=res_df['Signal_Date_ISO'].tolist())
    )
    
    st.altair_chart(line + points, use_container_width=True)

# ==========================================
# MAIN ROUTER
# ==========================================
def run():
    # Sidebar Navigation
    st.sidebar.title("💎 Mr. Darcy's Manor")
    
    app_mode = st.sidebar.radio("Navigation", [
        "Database",
        "Rankings",
        "Pivot Tables",
        "EMA Distance",
        "Seasonality",
        "RSI Scanner",
        "Price Divergences"
    ])
    
    # Load Global Data (Only once)
    # Note: We rely on cache in utils to prevent re-downloading on every click
    DATA_URL = st.secrets.get("URL_DATA_MAIN")
    if not DATA_URL:
        st.error("Global Data URL missing in secrets.")
        return

    df_global = ud.load_and_clean_data(DATA_URL)
    
    # Router
    if app_mode == "Database":
        run_database_app(df_global)
    elif app_mode == "Rankings":
        run_rankings_app(df_global)
    elif app_mode == "Pivot Tables":
        run_pivot_tables_app(df_global)
    elif app_mode == "EMA Distance":
        run_ema_distance_app(df_global)
    elif app_mode == "Seasonality":
        run_seasonality_app(df_global)
    elif app_mode == "RSI Scanner":
        run_rsi_scanner_app()
    elif app_mode == "Price Divergences":
        run_price_divergences_app()

if __name__ == "__main__":
    run()
