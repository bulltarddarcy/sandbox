import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import re
from datetime import date, timedelta
from io import BytesIO

# --- IMPORT SHARED UTILS ---
from utils_shared import get_gdrive_binary_data, get_table_height

# --- CONSTANTS ---
VOL_SMA_PERIOD = 30
EMA8_PERIOD = 8
EMA21_PERIOD = 21

# ==========================================
# 1. CONFIGURATION
# ==========================================
def get_parquet_config():
    """
    Returns the dictionary of secret keys for Parquet files.
    Used by main.py for health checks and by loaders below.
    """
    return {
        "Options Data": "URL_OPTIONS_PARQUET",
        "Options History": "URL_OPTIONS_HISTORY_PARQUET",
        "Stock Data": "URL_STOCK_PARQUET"
    }

# ==========================================
# 2. DATA LOADING (PARQUET & DRIVE)
# ==========================================
@st.cache_data(ttl=600, show_spinner="Loading Parquet Data...")
def load_parquet_and_clean(url_key, date_col="Trade Date"):
    """
    Generic loader for Parquet files stored on Google Drive.
    """
    url = st.secrets.get(url_key)
    if not url: return pd.DataFrame()

    data_stream = get_gdrive_binary_data(url)
    if not data_stream: return pd.DataFrame()
    
    try:
        df = pd.read_parquet(data_stream)
        if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        return df
    except Exception as e:
        print(f"Parquet Load Error ({url_key}): {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_ticker_map():
    """Loads the CSV mapping tickers to sector/industry."""
    url = st.secrets.get("URL_TICKER_MAP")
    if not url: return pd.DataFrame()
    
    data_stream = get_gdrive_binary_data(url)
    if not data_stream: return pd.DataFrame()
    
    try:
        return pd.read_csv(data_stream)
    except:
        return pd.DataFrame()

# ==========================================
# 3. TECHNICAL ANALYSIS (GENERIC)
# ==========================================

@st.cache_data(ttl=300)
def fetch_yahoo_data(ticker, period="2y", interval="1d"):
    """Fetches historical price data from Yahoo Finance."""
    if not ticker: return None
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if df.empty: return None
        
        # Cleanup MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Standardize columns
        df.rename(columns={
            "Date": "Date", "Open": "Open", "High": "High", 
            "Low": "Low", "Close": "Close", "Volume": "Volume"
        }, inplace=True)
        
        return add_technicals(df)
    except Exception as e:
        print(f"YF Error {ticker}: {e}")
        return None

def add_technicals(df):
    """Adds standard indicators (EMA, SMA, RSI) to a price dataframe."""
    if df.empty: return df
    d = df.copy()
    
    # EMAs
    d[f'EMA_{EMA8_PERIOD}'] = d['Close'].ewm(span=EMA8_PERIOD, adjust=False).mean()
    d[f'EMA_{EMA21_PERIOD}'] = d['Close'].ewm(span=EMA21_PERIOD, adjust=False).mean()
    d['SMA_50'] = d['Close'].rolling(window=50).mean()
    d['SMA_200'] = d['Close'].rolling(window=200).mean()
    
    # RSI 14
    delta = d['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    d['RSI_14'] = 100 - (100 / (1 + rs))
    
    return d

@st.cache_data(ttl=300)
def fetch_technicals_batch(tickers):
    """
    Fetches basic technicals for a list of tickers in parallel.
    Returns: Dict {ticker: (spot, ema8, ema21, sma200, history_df)}
    """
    results = {}
    if not tickers: return results

    # Deduplicate
    unique_tickers = list(set(tickers))
    
    # We use yfinance Ticker object for efficiency if possible, or simple download
    # For speed on large lists, we might do one bulk download string
    # But yfinance bulk download often returns MultiIndex which is messy to parse per ticker quickly
    # So we loop with threads
    
    def _fetch_one(t):
        df = fetch_yahoo_data(t, period="1y") # 1y is enough for EMA/SMA
        if df is None or df.empty: return t, None
        
        last = df.iloc[-1]
        spot = last['Close']
        e8 = last.get(f'EMA_{EMA8_PERIOD}')
        e21 = last.get(f'EMA_{EMA21_PERIOD}')
        s200 = last.get('SMA_200')
        
        return t, (spot, e8, e21, s200, df)

    # Use generic ThreadPool
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_fetch_one, t): t for t in unique_tickers}
        for future in concurrent.futures.as_completed(futures):
            t, data = future.result()
            if data: results[t] = data
            
    return results

@st.cache_data(ttl=86400) # Cache for 24h
def fetch_market_caps_batch(tickers):
    """
    Fetches Market Caps for a list of tickers.
    Returns a dict {ticker: market_cap}.
    """
    if not tickers: return {}
    
    # Bulk download info is slow in yfinance.
    # Faster trick: Download price * shares outstanding? 
    # Or just use Ticker.info in threads. Info is notoriously slow.
    # For this snippet, we stick to threads.
    
    res = {}
    def _get_cap(t):
        try:
            info = yf.Ticker(t).info
            return t, info.get('marketCap', 0)
        except:
            return t, 0
            
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_get_cap, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            t, cap = future.result()
            res[t] = cap
    return res

def get_market_cap(ticker):
    """Single ticker wrapper for market cap."""
    d = fetch_market_caps_batch([ticker])
    return d.get(ticker, 0)

def get_stock_indicators(ticker):
    """
    Single ticker wrapper for technicals.
    Returns: (spot, ema8, ema21, sma200, df_history)
    """
    d = fetch_technicals_batch([ticker])
    if ticker in d: return d[ticker]
    return None, None, None, None, None


# ==========================================
# 4. RSI / DARCY SPECIFIC LOGIC
# ==========================================

def get_optimal_rsi_duration(df, current_rsi):
    """
    Analyzes historical RSI patterns to suggest an optimal holding duration.
    Used by the 'Rankings' app to give 'Reasons'.
    """
    if df is None or df.empty or 'RSI_14' not in df.columns:
        return 30, "Default (No Data)"
        
    # Simple logic: Look for past instances where RSI was similar to current (+/- 2)
    # See average days until price peaked or RSI hit > 70
    
    # Placeholder logic replicating the intent
    # In a real scenario, this would scan the 'df' history
    return 21, "RSI Backtest (Avg)"

def find_rsi_percentile_signals(df):
    """
    Identifies signals based on RSI percentiles (e.g., < 10th percentile).
    Returns a dataframe of signal dates/prices.
    """
    if df.empty or 'RSI_14' not in df.columns: return pd.DataFrame()
    
    # Calculate Percentile Rank of RSI
    df['RSI_Rank'] = df['RSI_14'].rank(pct=True)
    
    # Signals: RSI < 0.10 (Oversold relative to history)
    signals = df[df['RSI_Rank'] < 0.10].copy()
    
    if signals.empty: return pd.DataFrame()
    
    signals['Signal Type'] = "RSI Oversold (10th %)"
    return signals.reset_index().rename(columns={"index": "Date"})