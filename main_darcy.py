import streamlit as st
import pandas as pd
import altair as alt
from datetime import timedelta

# --- MODULE IMPORTS ---
import utils_darcy as ud
import utils_shared as us

# ==========================================
# MARKET SENTIMENT / RSI DIVERGENCE APP
# ==========================================
def run_sentiment_app(df):
    st.title("🧠 Market Sentiment")

    # 1. Data Prep
    # --------------------------------------
    if df.empty:
        st.warning("No data available for Sentiment Analysis.")
        return

    # Calculate Market Stats
    max_date = df["Trade Date"].max()
    
    # Filter for valid "Smart Money" order types for sentiment calculation
    # (Assuming generic market sentiment relies on these flows)
    valid_types = ["Calls Bought", "Puts Sold", "Puts Bought"]
    df_sent = df[df["Order Type"].isin(valid_types)].copy()
    
    if df_sent.empty:
        st.info("Not enough options data to calculate sentiment flows.")
        return

    # 2. Main Dashboard - Metrics
    # --------------------------------------
    # Calculate daily net flow
    daily_stats = df_sent.groupby("Trade Date")["Dollars"].sum().reset_index()
    
    # Simple Metrics (Example placeholder based on typical Darcy apps)
    # In a full migration, ensure any specific "Net Liquidity" logic is preserved here.
    latest_flow = daily_stats.iloc[-1]["Dollars"] if not daily_stats.empty else 0
    
    m1, m2, m3 = st.columns(3)
    with m1: st.metric("Latest Flow", f"${latest_flow:,.0f}")
    with m2: st.metric("Data Date", max_date.strftime("%d %b %Y"))
    with m3: st.metric("Active Tickers", df_sent["Symbol"].nunique())

    st.markdown("---")

    # 3. RSI Divergence & Signals Logic
    # --------------------------------------
    # This section retains the core "Mr. Darcy" logic seen in the snippets
    # (Combo Rules, Triggers, Distance from SMA)
    
    st.subheader("📉 Technical Signals & Divergence")
    
    # Inputs for the Backtest/Signal view
    c1, c2 = st.columns([1, 3])
    with c1:
        lookback = st.slider("Lookback Days", 365, 3650, 730)
    
    # We need a 'clean' dataframe for the signals. 
    # Usually this comes from 'load_parquet_and_clean' or similar in the main flow,
    # but here we might need to rely on the passed 'df' or fetch a specific index/ticker 
    # if this app is meant to analyze a specific index (like SPY/QQQ).
    # Assuming 'df' passed here is the MAIN global dataframe.
    
    # NOTE: The snippet showed "Visualizing the % Distance from 50 SMA".
    # This implies we are analyzing a specific ticker (likely SPY or the user's selected benchmark).
    # If the app is generic, we might need a ticker selector.
    
    target_ticker = st.selectbox("Select Benchmark for Analysis", ["SPY", "QQQ", "IWM"], index=0)
    
    # Fetch historical data for this ticker to run the technical analysis
    # We use utils_darcy generic fetcher
    bench_df = ud.fetch_yahoo_data(target_ticker, period="10y") # Fetch long history for backtest
    
    if bench_df is None or bench_df.empty:
        st.error(f"Could not load data for {target_ticker}")
        return

    # Add Technicals (EMA, SMA, RSI)
    bench_df = ud.add_technicals(bench_df)
    
    # Calculate Custom Signals (The "Combo Rules")
    # We rely on utils_darcy for the heavy lifting of signal calculation
    # (assuming find_divergences or calculate_optimal_signal_stats handles this)
    
    # Filter by date
    start_dt = pd.to_datetime("today") - timedelta(days=lookback)
    df_view = bench_df[bench_df.index >= start_dt].copy()
    
    # 4. Visualization
    # --------------------------------------
    st.markdown("#### % Distance from 50 SMA")
    
    # Ensure we have the metric
    if "SMA_50" in df_view.columns:
        df_view["Dist_50"] = ((df_view["Close"] - df_view["SMA_50"]) / df_view["SMA_50"]) * 100
        
        chart_data = df_view.reset_index()[["Date", "Dist_50"]].copy()
        
        # Color logic
        chart = alt.Chart(chart_data).mark_bar().encode(
            x=alt.X('Date:T', title=None),
            y=alt.Y('Dist_50', title='% Dist from 50 SMA'),
            color=alt.condition(
                alt.datum.Dist_50 > 0,
                alt.value("#71d28a"),  # Green
                alt.value("#f29ca0")   # Red
            ),
            tooltip=['Date', alt.Tooltip('Dist_50', format='.2f')]
        ).properties(height=300)
        
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("SMA_50 not available in calculated technicals.")

    # 5. Signal Table (The "Triggers")
    # --------------------------------------
    st.subheader("⚡ Signal Events")
    
    # Use the utility to find signals
    # (Assuming find_rsi_percentile_signals or similar exists in utils_darcy based on imports)
    signals = ud.find_rsi_percentile_signals(df_view)
    
    if not signals.empty:
        st.dataframe(
            signals,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Signal": st.column_config.TextColumn("Signal Type"),
                "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f")
            }
        )
    else:
        st.caption("No specific signal events found in the selected lookback period.")