import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import date, timedelta

# --- MODULE IMPORTS ---
import utils_shared as us
import utils_darcy as ud  # For technical data fetching

# ==========================================
# 1. CONSTANTS
# ==========================================

# --- DATABASE APP ---
DB_DEFAULT_EXPIRY_OFFSET = 365
DB_TABLE_MAX_ROWS = 30
DB_DATE_FMT = "%d %b %y"
STYLE_BULL_CSS = 'background-color: rgba(113, 210, 138, 0.15); color: #71d28a; font-weight: 600;'
STYLE_BEAR_CSS = 'background-color: rgba(242, 156, 160, 0.15); color: #f29ca0; font-weight: 600;'

# --- RANKING APP ---
RANK_LOOKBACK_DAYS = 14
RANK_LIMIT_DEFAULT = 20
RANK_MC_THRESHOLDS = {"0B": 0, "2B": 2e9, "10B": 1e10, "50B": 5e10, "100B": 1e11}
RANK_SM_WEIGHTS = {'Sentiment': 0.35, 'Impact': 0.30, 'Momentum': 0.35}
RANK_CONVICTION_DIVISOR = 25.0
RANK_TOP_IDEAS_COUNT = 3

# --- PIVOT APP ---
PIVOT_NOTIONAL_MAP = {"0M": 0, "5M": 5e6, "10M": 1e7, "50M": 5e7, "100M": 1e8}
PIVOT_MC_MAP = {"0B": 0, "10B": 1e10, "50B": 5e10, "100B": 1e11, "200B": 2e11, "500B": 5e11, "1T": 1e12}
PIVOT_TABLE_FMT = {"Dollars": "${:,.0f}", "Contracts": "{:,.0f}"}

COLUMN_CONFIG_PIVOT = {
    "Symbol": st.column_config.TextColumn("Sym", width=None),
    "Strike": st.column_config.TextColumn("Strike", width=None),
    "Expiry_Table": st.column_config.TextColumn("Exp", width=None),
    "Contracts": st.column_config.NumberColumn("Qty", width=None),
    "Dollars": st.column_config.NumberColumn("Dollars", width=None),
}

# --- STRIKE ZONES APP ---
SZ_DEFAULT_EXP_OFFSET = 365
SZ_DEFAULT_FIXED_SIZE = 10
SZ_AUTO_WIDTH_DENOM = 12.0
SZ_AUTO_STEPS = [1, 2, 5, 10, 25, 50, 100]
SZ_BUCKET_BINS = [0, 7, 30, 60, 90, 120, 180, 365, 10000]
SZ_BUCKET_LABELS = ["0-7d", "8-30d", "31-60d", "61-90d", "91-120d", "121-180d", "181-365d", ">365d"]


# ==========================================
# 2. SHARED HELPERS
# ==========================================

def get_max_trade_date(df):
    if not df.empty and "Trade Date" in df.columns:
        valid = df["Trade Date"].dropna()
        if not valid.empty: return valid.max().date()
    return date.today() - timedelta(days=1)

@st.cache_data(ttl=600)
def get_expiry_color_map():
    try:
        today = date.today()
        days_ahead = (4 - today.weekday()) % 7
        this_fri = today + timedelta(days=days_ahead)
        return {
            this_fri.strftime("%d %b %y"): "background-color: #b7e1cd; color: black;",
            (this_fri + timedelta(days=7)).strftime("%d %b %y"): "background-color: #fce8b2; color: black;",
            (this_fri + timedelta(days=14)).strftime("%d %b %y"): "background-color: #f4c7c3; color: black;"
        }
    except:
        return {}

def highlight_expiry(val):
    if not isinstance(val, str): return ""
    return get_expiry_color_map().get(val, "")

def clean_strike_fmt(val):
    try:
        f = float(val)
        if f.is_integer(): return str(int(f))
        return str(f)
    except: return str(val)

def find_whale_confluence(ticker, global_df, current_price, order_type_filter=None):
    if global_df.empty: return None

    f = global_df[global_df["Symbol"].astype(str).str.upper() == ticker].copy()
    if f.empty: return None

    today_dt = pd.Timestamp.now()
    f = f[f["Expiry_DT"] > today_dt]
    
    if order_type_filter:
        f = f[f["Order Type"] == order_type_filter]
    else:
        f = f[f["Order Type"].isin(["Puts Sold", "Calls Bought"])]
        
    if f.empty: return None
    
    f.sort_values(by="Dollars", ascending=False, inplace=True)
    whale = f.iloc[0]
    
    if whale["Order Type"] == "Puts Sold" and whale["Strike (Actual)"] > current_price:
        otm_puts = f[(f["Order Type"]=="Puts Sold") & (f["Strike (Actual)"] < current_price)]
        if not otm_puts.empty: whale = otm_puts.iloc[0]
    
    return {
        "Strike": whale["Strike (Actual)"],
        "Expiry": whale["Expiry_DT"].strftime("%d %b"),
        "Dollars": whale["Dollars"], "Type": whale["Order Type"]
    }

def analyze_trade_setup(ticker, t_df, global_df):
    score = 0; reasons = []
    suggestions = {'Buy Calls': None, 'Sell Puts': None, 'Buy Commons': None}
    
    if t_df is None or t_df.empty: return 0, ["No data"], suggestions

    last = t_df.iloc[-1]
    close = last.get('CLOSE', last.get('Price', 0))
    ema8 = last.get('EMA8', last.get('EMA_8', 0))
    ema21 = last.get('EMA21', last.get('EMA_21', 0))
    sma200 = last.get('SMA200', last.get('SMA_200', 0))
    rsi = last.get('RSI', last.get('RSI_14', 50))
    
    if close > ema8 and close > ema21:
        score += 2; reasons.append("Strong Trend (Price > EMA8 & EMA21)")
    elif close > ema21:
        score += 1; reasons.append("Moderate Trend (Price > EMA21)")
        
    if close > sma200:
        score += 2; reasons.append("Long-term Bullish (> SMA200)")
        
    if 45 < rsi < 65:
        score += 2; reasons.append(f"Healthy Momentum (RSI {rsi:.0f})")
    elif rsi >= 70:
        score -= 1; reasons.append("Overbought (RSI > 70)")
    
    opt_days, opt_reason = 30, "Standard 30d"
    # Note: get_optimal_rsi_duration is in utils_darcy, importing via ud
    if len(t_df) > 100: opt_days, opt_reason = ud.get_optimal_rsi_duration(t_df, rsi)
    
    target_date = date.today() + timedelta(days=opt_days)
    target_date_str = target_date.strftime("%d %b")
    
    put_whale = find_whale_confluence(ticker, global_df, close, "Puts Sold")
    call_whale = find_whale_confluence(ticker, global_df, close, "Calls Bought")
    
    sp_strike = math.floor(ema21) 
    sp_reason = "EMA21 Support"
    sp_exp = target_date_str
    
    if put_whale and put_whale["Strike"] < close:
        sp_strike = put_whale["Strike"]
        sp_reason = f"Whale Tailing (${put_whale['Dollars']/1e6:.1f}M sold)"
        sp_exp = put_whale["Expiry"] 
    elif call_whale:
         sp_exp = call_whale["Expiry"]
         sp_reason = f"EMA21 (Align with Call Whale Exp)"
    
    suggestions['Sell Puts'] = f"Strike ${sp_strike} ({sp_reason}), Exp ~{sp_exp}"

    bc_strike = math.ceil(close)
    bc_reason = "ATM Momentum"
    bc_exp = target_date_str
    
    if call_whale:
        bc_strike = call_whale["Strike"]
        bc_exp = call_whale["Expiry"]
        bc_reason = f"Tailing Call Whale (${call_whale['Dollars']/1e6:.1f}M)"
        
    if close > ema8 or call_whale:
        suggestions['Buy Calls'] = f"Strike ${bc_strike} ({bc_reason}), Exp ~{bc_exp}"
        
    suggestions['Buy Commons'] = f"Entry: ${close:.2f}. Stop Loss: ${ema21:.2f}"
    
    if "RSI Backtest" in opt_reason: reasons.append(f"Hist. Optimal Hold: {opt_days} Days")
    if put_whale: reasons.append(f"Whale: Sold Puts @ ${put_whale['Strike']}")
    if call_whale: reasons.append(f"Whale: Bought Calls @ ${call_whale['Strike']}")
    
    return score, reasons, suggestions


# ==========================================
# 3. DATABASE APP UTILS
# ==========================================

def initialize_database_state(max_date):
    """Initializes session state variables for the Database app."""
    defaults = {
        'saved_db_ticker': "",
        'saved_db_start': max_date,
        'saved_db_end': max_date,
        'saved_db_exp': (date.today() + timedelta(days=DB_DEFAULT_EXPIRY_OFFSET)),
        'saved_db_inc_cb': True,
        'saved_db_inc_ps': True,
        'saved_db_inc_pb': True
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def _highlight_db_order_type(val):
    """Internal styling function for order types."""
    if val in ["Calls Bought", "Puts Sold"]: 
        return STYLE_BULL_CSS
    elif val == "Puts Bought": 
        return STYLE_BEAR_CSS
    return ''

def get_database_styled_view(df):
    """Prepares the Styler object for the database table."""
    if df.empty: return df
    
    order_type_col = "Order Type" if "Order Type" in df.columns else "Order type"
    display_cols = ["Trade Date", order_type_col, "Symbol", "Strike", "Expiry", "Contracts", "Dollars"]
    
    f_display = df[display_cols].copy()
    
    # Format Dates
    if not pd.api.types.is_string_dtype(f_display["Trade Date"]):
        f_display["Trade Date"] = f_display["Trade Date"].dt.strftime(DB_DATE_FMT)
    
    try:
        f_display["Expiry"] = pd.to_datetime(f_display["Expiry"]).dt.strftime(DB_DATE_FMT)
    except:
        pass 
        
    return f_display.style.format({
        "Dollars": "${:,.0f}", 
        "Contracts": "{:,.0f}"
    }).map(_highlight_db_order_type, subset=[order_type_col])

def filter_database_trades(df, ticker, start_date, end_date, exp_end, inc_cb, inc_ps, inc_pb):
    """Filters the global dataframe based on UI inputs."""
    if df.empty: return pd.DataFrame()

    f = df.copy()
    
    # Date & Ticker Filters
    if ticker: 
        f = f[f["Symbol"].astype(str).str.upper().eq(ticker)]
    if start_date: 
        f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: 
        f = f[f["Trade Date"].dt.date <= end_date]
    if exp_end: 
        f = f[f["Expiry_DT"].dt.date <= exp_end]
    
    # Order Type Logic
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    allowed_types = []
    if inc_cb: allowed_types.append("Calls Bought")
    if inc_ps: allowed_types.append("Puts Sold")
    if inc_pb: allowed_types.append("Puts Bought")
    
    f = f[f[order_type_col].isin(allowed_types)]
    
    # Sort for display
    return f.sort_values(by=["Trade Date", "Symbol"], ascending=[False, True])


# ==========================================
# 4. RANKINGS APP UTILS
# ==========================================

def initialize_rankings_state(start_default, max_date):
    """Initializes session state for Rankings app."""
    defaults = {
        'saved_rank_start': start_default,
        'saved_rank_end': max_date,
        'saved_rank_limit': RANK_LIMIT_DEFAULT,
        'saved_rank_mc': "10B",
        'saved_rank_ema': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

@st.cache_data(ttl=600, show_spinner="Crunching Smart Money Data...")
def calculate_smart_money_score(df, start_d, end_d, mc_thresh, filter_ema, limit):
    f = df.copy()
    if start_d: f = f[f["Trade Date"].dt.date >= start_d]
    if end_d: f = f[f["Trade Date"].dt.date <= end_d]
    if f.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]
    f = f[f[order_type_col].isin(target_types)].copy()
    if f.empty: return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    f["Signed_Dollars"] = np.where(f[order_type_col].isin(["Calls Bought", "Puts Sold"]), f["Dollars"], -f["Dollars"])
    
    smart_stats = f.groupby("Symbol").agg(
        Signed_Dollars=("Signed_Dollars", "sum"),
        Trade_Count=("Symbol", "count"),
        Last_Trade=("Trade Date", "max")
    ).reset_index()
    
    smart_stats.rename(columns={"Signed_Dollars": "Net Sentiment ($)"}, inplace=True)
    unique_tickers = smart_stats["Symbol"].unique().tolist()
    # Using generic fetcher from Utils Darcy
    batch_caps = ud.fetch_market_caps_batch(unique_tickers)
    smart_stats["Market Cap"] = smart_stats["Symbol"].map(batch_caps)
    
    valid_data = smart_stats[smart_stats["Market Cap"] >= mc_thresh].copy()
    
    unique_dates = sorted(f["Trade Date"].unique())
    recent_dates = unique_dates[-3:] if len(unique_dates) >= 3 else unique_dates
    f_momentum = f[f["Trade Date"].isin(recent_dates)]
    mom_stats = f_momentum.groupby("Symbol")["Signed_Dollars"].sum().reset_index().rename(columns={"Signed_Dollars": "Momentum ($)"})
    
    valid_data = valid_data.merge(mom_stats, on="Symbol", how="left").fillna(0)
    top_bulls = pd.DataFrame(); top_bears = pd.DataFrame()

    if not valid_data.empty:
        valid_data["Impact"] = valid_data["Net Sentiment ($)"] / valid_data["Market Cap"]
        def normalize(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx != mn else 0

        valid_data["Base_Score_Bull"] = (0.35 * normalize(valid_data["Net Sentiment ($)"].clip(lower=0))) + \
                                        (0.30 * normalize(valid_data["Impact"].clip(lower=0))) + \
                                        (0.35 * normalize(valid_data["Momentum ($)"].clip(lower=0)))
        
        valid_data["Base_Score_Bear"] = (0.35 * normalize(-valid_data["Net Sentiment ($)"].clip(upper=0))) + \
                                        (0.30 * normalize(-valid_data["Impact"].clip(upper=0))) + \
                                        (0.35 * normalize(-valid_data["Momentum ($)"].clip(upper=0)))
        
        valid_data["Last Trade"] = valid_data["Last_Trade"].dt.strftime("%d %b")
        
        candidates_bull = valid_data.sort_values(by="Base_Score_Bull", ascending=False).head(limit * 3)
        candidates_bear = valid_data.sort_values(by="Base_Score_Bear", ascending=False).head(limit * 3)
        
        all_tickers = set(candidates_bull["Symbol"]).union(set(candidates_bear["Symbol"]))
        # Using generic fetcher from Utils Darcy
        batch_techs = ud.fetch_technicals_batch(list(all_tickers)) if filter_ema else {}

        def apply_ema_filter(df, mode="Bull"):
            if not filter_ema:
                df["Score"] = df[f"Base_Score_{mode}"] * 100
                df["Trend"] = "—"
                return df.head(limit)
            
            def check_row(t):
                s, e8, _, _, _ = batch_techs.get(t, (None, None, None, None, None))
                if not s or not e8: return False, "—"
                if mode == "Bull": return (s > e8), ("✅ >EMA8" if s > e8 else "⚠️ <EMA8")
                return (s < e8), ("✅ <EMA8" if s < e8 else "⚠️ >EMA8")
            
            results = [check_row(t) for t in df["Symbol"]]
            mask = [r[0] for r in results]; trends = [r[1] for r in results]
            
            filtered = df[mask].copy()
            filtered["Trend"] = [t for i, t in enumerate(trends) if mask[i]]
            filtered["Score"] = filtered[f"Base_Score_{mode}"] * 100
            return filtered.head(limit)

        top_bulls = apply_ema_filter(candidates_bull, "Bull")
        top_bears = apply_ema_filter(candidates_bear, "Bear")
        
    return top_bulls, top_bears, valid_data

def generate_top_ideas(top_bulls_df, global_df):
    """Analyzes top bullish candidates using updated constants."""
    if top_bulls_df.empty: return []

    candidates = []
    bull_list = top_bulls_df["Symbol"].tolist()
    # Using generic fetcher from Utils Darcy
    batch_results = ud.fetch_technicals_batch(bull_list)
    
    prog_bar = st.progress(0, text="Analyzing technicals...")
    
    for i, t in enumerate(bull_list):
        prog_bar.progress((i + 1) / len(bull_list), text=f"Checking {t}...")
        
        data_tuple = batch_results.get(t)
        t_df = data_tuple[4] if data_tuple else None

        if t_df is None or t_df.empty:
            t_df = ud.fetch_yahoo_data(t)

        if t_df is not None and not t_df.empty:
            sm_row = top_bulls_df[top_bulls_df["Symbol"] == t]
            sm_score = sm_row["Score"].iloc[0] if not sm_row.empty else 0
            
            # Using local analyze_trade_setup (which calls local find_whale_confluence)
            tech_score, reasons, suggs = analyze_trade_setup(t, t_df, global_df)
            
            # Use Constant for Conviction Divisor
            final_conviction = (sm_score / RANK_CONVICTION_DIVISOR) + tech_score 
            
            price = t_df.iloc[-1].get('CLOSE') or t_df.iloc[-1].get('Close')
            
            candidates.append({
                "Ticker": t,
                "Score": final_conviction,
                "Price": price,
                "Reasons": reasons,
                "Suggestions": suggs
            })
    
    prog_bar.empty()
    
    # Use Constant for Top N
    return sorted(candidates, key=lambda x: x['Score'], reverse=True)[:RANK_TOP_IDEAS_COUNT]

def filter_rankings_data(df, start_date, end_date):
    """Filters data for the rankings period and valid order types."""
    if df.empty: return pd.DataFrame()
    
    f = df.copy()
    if start_date: f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: f = f[f["Trade Date"].dt.date <= end_date]
    
    # Filter for valid directional trades only
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]
    f = f[f[order_type_col].isin(target_types)]
    
    return f

def calculate_volume_rankings(f_filtered, mc_thresh, filter_ema, limit):
    """
    Calculates the 'Bulltard' style volume rankings.
    Score = (Calls + Puts Sold) - (Puts Bought)
    """
    if f_filtered.empty: return pd.DataFrame(), pd.DataFrame()

    order_type_col = "Order Type" if "Order Type" in f_filtered.columns else "Order type"
    target_types = ["Calls Bought", "Puts Sold", "Puts Bought"]

    # 1. Pivot Count
    counts = f_filtered.groupby(["Symbol", order_type_col]).size().unstack(fill_value=0)
    for col in target_types:
        if col not in counts.columns: counts[col] = 0
        
    # 2. Calculate Score
    scores_df = pd.DataFrame(index=counts.index)
    scores_df["Score"] = counts["Calls Bought"] + counts["Puts Sold"] - counts["Puts Bought"]
    scores_df["Trade Count"] = counts.sum(axis=1)
    
    # 3. Add Last Trade Date
    last_trade_series = f_filtered.groupby("Symbol")["Trade Date"].max()
    scores_df["Last Trade"] = last_trade_series.dt.strftime(DB_DATE_FMT)
    
    res = scores_df.reset_index()
    
    # 4. Map Market Caps
    unique_ts = res["Symbol"].unique().tolist()
    caps = ud.fetch_market_caps_batch(unique_ts)
    res["Market Cap"] = res["Symbol"].map(caps)
    res = res[res["Market Cap"] >= mc_thresh]

    # 5. Sort Lists
    pre_bull_df = res.sort_values(by=["Score", "Trade Count"], ascending=[False, False])
    pre_bear_df = res.sort_values(by=["Score", "Trade Count"], ascending=[True, False])

    # 6. Apply EMA Filter (Helper)
    def _apply_ema_limit(source_df, mode="Bull"):
        if not filter_ema:
            return source_df.head(limit)
        
        # Fetch extra candidates to account for filtering attrition
        candidates = source_df.head(limit * 3)
        final_list = []
        needed_tickers = candidates["Symbol"].tolist()
        
        # Small batch fetch for just these candidates
        mini_batch = ud.fetch_technicals_batch(needed_tickers)
        
        for _, r in candidates.iterrows():
            try:
                # Tuple: (spot, ema8, ema21, sma200, history)
                t_data = mini_batch.get(r["Symbol"], (None, None, None, None, None))
                spot, ema8 = t_data[0], t_data[1]
                
                if spot and ema8:
                    if mode == "Bull" and spot > ema8: final_list.append(r)
                    elif mode == "Bear" and spot < ema8: final_list.append(r)
            except: 
                pass
            
            if len(final_list) >= limit: break
        
        return pd.DataFrame(final_list)

    bull_df = _apply_ema_limit(pre_bull_df, "Bull")
    bear_df = _apply_ema_limit(pre_bear_df, "Bear")
    
    return bull_df, bear_df


# ==========================================
# 5. PIVOT TABLE APP UTILS
# ==========================================

def initialize_pivot_state(start_default, max_date):
    """Initializes session state for Pivot app."""
    defaults = {
        'saved_pv_start': max_date,
        'saved_pv_end': max_date,
        'saved_pv_ticker': "",
        'saved_pv_notional': "0M",
        'saved_pv_mkt_cap': "0B",
        'saved_pv_ema': "All",
        'saved_calc_strike': 100.0,
        'saved_calc_premium': 2.50,
        'saved_calc_expiry': date.today() + timedelta(days=30)
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def generate_pivot_pools(d_range):
    """
    Splits data into pools and identifies Risk Reversals (Matched Calls/Puts).
    Replicates original logic: Matches 'Calls Bought' with 'Puts Sold' by Symbol/Date/Expiry.
    """
    order_type_col = "Order Type" if "Order Type" in d_range.columns else "Order type"
    
    cb_pool = d_range[d_range[order_type_col] == "Calls Bought"].copy()
    ps_pool = d_range[d_range[order_type_col] == "Puts Sold"].copy()
    pb_pool = d_range[d_range[order_type_col] == "Puts Bought"].copy()
    
    # Risk Reversal Matching Logic
    keys = ['Trade Date', 'Symbol', 'Expiry_DT', 'Contracts']
    
    # Use groupby cumcount to handle multiple identical trades on same day
    cb_pool['occ'] = cb_pool.groupby(keys).cumcount()
    ps_pool['occ'] = ps_pool.groupby(keys).cumcount()
    
    rr_matches = pd.merge(cb_pool, ps_pool, on=keys + ['occ'], suffixes=('_c', '_p'))
    
    if not rr_matches.empty:
        # Construct RR DataFrame
        rr_c = rr_matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_c', 'Strike_c']].copy()
        rr_c.rename(columns={'Dollars_c': 'Dollars', 'Strike_c': 'Strike'}, inplace=True)
        rr_c['Pair_ID'] = rr_matches.index
        rr_c['Pair_Side'] = 0 # Call Side
        
        rr_p = rr_matches[['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars_p', 'Strike_p']].copy()
        rr_p.rename(columns={'Dollars_p': 'Dollars', 'Strike_p': 'Strike'}, inplace=True)
        rr_p['Pair_ID'] = rr_matches.index
        rr_p['Pair_Side'] = 1 # Put Side
        
        df_rr = pd.concat([rr_c, rr_p])
        df_rr['Strike'] = df_rr['Strike'].apply(clean_strike_fmt)
        
        # Filter matched rows out of the original pools
        match_keys = keys + ['occ']
        def _filter_out(pool, matches):
            temp = matches[match_keys].copy()
            temp['_remove'] = True
            merged = pool.merge(temp, on=match_keys, how='left')
            return merged[merged['_remove'].isna()].drop(columns=['_remove'])
            
        cb_pool = _filter_out(cb_pool, rr_matches)
        ps_pool = _filter_out(ps_pool, rr_matches)
    else:
        df_rr = pd.DataFrame(columns=['Symbol', 'Trade Date', 'Expiry_DT', 'Contracts', 'Dollars', 'Strike', 'Pair_ID', 'Pair_Side'])
        
    return cb_pool, ps_pool, pb_pool, df_rr

def filter_pivot_dataframe(data, ticker_filter, min_notional, min_mkt_cap, ema_filter):
    """Applies the Filters (Ticker, Notional, Market Cap, EMA) to a dataframe."""
    if data.empty: return data
    f = data.copy()
    
    if ticker_filter: 
        f = f[f["Symbol"].astype(str).str.upper() == ticker_filter]
    
    f = f[f["Dollars"] >= min_notional]
    
    # Advanced Filters (Market Cap & EMA)
    if not f.empty and (min_mkt_cap > 0 or ema_filter == "Yes"):
        unique_symbols = f["Symbol"].unique()
        valid_symbols = set(unique_symbols)
        
        if min_mkt_cap > 0:
            valid_symbols = {s for s in valid_symbols if ud.get_market_cap(s) >= float(min_mkt_cap)}
        
        if ema_filter == "Yes":
            # Check EMA > SMA using batch fetch
            batch_results = ud.fetch_technicals_batch(list(valid_symbols))
            # Keep symbol if data missing OR if Price > EMA21 (Tuple index 0 is Price, index 2 is EMA21)
            valid_symbols = {
                s for s in valid_symbols 
                if batch_results.get(s, (None, None))[2] is None or 
                (batch_results[s][0] is not None and batch_results[s][2] is not None and batch_results[s][0] > batch_results[s][2])
            }
        
        f = f[f["Symbol"].isin(valid_symbols)]
        
    return f

def get_pivot_styled_view(data, is_rr=False):
    """Aggregates data by Symbol/Strike/Expiry for display."""
    if data.empty: 
        return pd.DataFrame(columns=["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"])
    
    sr = data.groupby("Symbol")["Dollars"].sum().rename("Total_Sym_Dollars")
    
    if is_rr: 
        piv = data.merge(sr, on="Symbol").sort_values(by=["Total_Sym_Dollars", "Pair_ID", "Pair_Side"], ascending=[False, True, True])
    else:
        piv = data.groupby(["Symbol", "Strike", "Expiry_DT"]).agg({"Contracts": "sum", "Dollars": "sum"}).reset_index().merge(sr, on="Symbol")
        piv = piv.sort_values(by=["Total_Sym_Dollars", "Dollars"], ascending=[False, False])
    
    piv["Expiry_Fmt"] = piv["Expiry_DT"].dt.strftime("%d %b %y")
    
    # Hide repeated symbols for cleaner look
    piv["Symbol_Display"] = np.where(piv["Symbol"] == piv["Symbol"].shift(1), "", piv["Symbol"])
    
    return piv.drop(columns=["Symbol"]).rename(columns={"Symbol_Display": "Symbol", "Expiry_Fmt": "Expiry_Table"})[["Symbol", "Strike", "Expiry_Table", "Contracts", "Dollars"]]


# ==========================================
# 6. STRIKE ZONES APP UTILS
# ==========================================

def initialize_strike_zone_state(exp_default):
    """Initializes session state for Strike Zones app."""
    defaults = {
        'saved_sz_ticker': "AMZN",
        'saved_sz_start': None,
        'saved_sz_end': None,
        'saved_sz_exp': exp_default,
        'saved_sz_view': "Price Zones",
        'saved_sz_width_mode': "Auto",
        'saved_sz_fixed': SZ_DEFAULT_FIXED_SIZE,
        'saved_sz_inc_cb': True,
        'saved_sz_inc_ps': True,
        'saved_sz_inc_pb': True
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

def filter_strike_zone_data(df, ticker, start_date, end_date, exp_end, inc_cb, inc_ps, inc_pb):
    """Filters the dataframe for the Strike Zones editor."""
    f = df[df["Symbol"].astype(str).str.upper().eq(ticker)].copy()
    if f.empty: return f

    if start_date: f = f[f["Trade Date"].dt.date >= start_date]
    if end_date: f = f[f["Trade Date"].dt.date <= end_date]
    
    # Expiry Filter (Today to End Range)
    today_val = date.today()
    f = f[(f["Expiry_DT"].dt.date >= today_val) & (f["Expiry_DT"].dt.date <= exp_end)]
    
    # Order Types
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    allowed = []
    if inc_cb: allowed.append("Calls Bought")
    if inc_ps: allowed.append("Puts Sold")
    if inc_pb: allowed.append("Puts Bought")
    
    f = f[f[order_type_col].isin(allowed)].copy()
    
    # Initialize Include col for editor
    if not f.empty and "Include" not in f.columns:
        f.insert(0, "Include", True)
        
    return f

def get_strike_zone_technicals(ticker):
    """Retrieves technicals, falling back to Yahoo calculation if needed."""
    # Using generic fetcher from Utils Darcy
    spot, ema8, ema21, sma200, _ = ud.get_stock_indicators(ticker)

    if spot is None:
        df_y = ud.fetch_yahoo_data(ticker)
        if df_y is not None and not df_y.empty:
            try:
                spot = float(df_y["CLOSE"].iloc[-1])
                # We need periods from Utils Darcy or hardcode them
                # Since we moved constants here but Utils Darcy still has logic...
                # Ideally, we rely on DataFrame having columns. 
                # If yahoo data is fresh, it might not have EMA8 calculated yet if add_technicals didn't run.
                # fetch_yahoo_data in Utils Darcy calls add_technicals, so we should be good.
                ema8 = float(df_y["EMA_8"].iloc[-1]) if "EMA_8" in df_y.columns else float(df_y["CLOSE"].ewm(span=8, adjust=False).mean().iloc[-1])
                ema21 = float(df_y["EMA_21"].iloc[-1]) if "EMA_21" in df_y.columns else float(df_y["CLOSE"].ewm(span=21, adjust=False).mean().iloc[-1])
                sma200 = float(df_y["SMA_200"].iloc[-1]) if "SMA_200" in df_y.columns and len(df_y) >= 200 else None
            except: 
                pass

    if spot is None: spot = 100.0
    return spot, ema8, ema21, sma200

def generate_price_zones_html(df, spot, width_mode, fixed_size, hide_empty=True):
    """Calculates price zones and returns the HTML string for the chart."""
    f = df.copy()
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    
    # Calculate Signed Dollars
    f["Signed Dollars"] = np.where(f[order_type_col].isin(["Calls Bought", "Puts Sold"]), 1, -1) * f["Dollars"].fillna(0.0)
    
    # Determine Zone Width
    strike_vals = f["Strike (Actual)"].values
    strike_min, strike_max = float(np.nanmin(strike_vals)), float(np.nanmax(strike_vals))
    
    if width_mode == "Auto": 
        denom = SZ_AUTO_WIDTH_DENOM
        raw_w = max(1e-9, strike_max - strike_min) / denom
        # Find nearest step >= raw_w from constants
        zone_w = float(next((s for s in SZ_AUTO_STEPS if s >= raw_w), 100))
    else: 
        zone_w = float(fixed_size)
    
    # Calculate Buckets relative to Spot
    n_dn = int(math.ceil(max(0.0, (spot - strike_min)) / zone_w))
    n_up = int(math.ceil(max(0.0, (strike_max - spot)) / zone_w))
    
    lower_edge = spot - n_dn * zone_w
    total = max(1, n_dn + n_up)
    
    f["ZoneIdx"] = np.clip(
        np.floor((f["Strike (Actual)"].fillna(0) - lower_edge) / zone_w).astype(int), 
        0, 
        total - 1
    )

    # Aggregation
    agg = f.groupby("ZoneIdx").agg(
        Net_Dollars=("Signed Dollars","sum"), 
        Trades=("Signed Dollars","count")
    ).reset_index()
    
    zone_df = pd.DataFrame([(z, lower_edge + z*zone_w, lower_edge + (z+1)*zone_w) for z in range(total)], columns=["ZoneIdx","Zone_Low","Zone_High"])
    zs = zone_df.merge(agg, on="ZoneIdx", how="left").fillna(0)
    
    if hide_empty: 
        zs = zs[~((zs["Trades"]==0) & (zs["Net_Dollars"].abs()<1e-6))]
    
    # Build HTML
    html_out = ['<div class="zones-panel">']
    max_val = max(1.0, zs["Net_Dollars"].abs().max())
    sorted_zs = zs.sort_values("ZoneIdx", ascending=False)
    
    upper_zones = sorted_zs[sorted_zs["Zone_Low"] + (zone_w/2) > spot]
    lower_zones = sorted_zs[sorted_zs["Zone_Low"] + (zone_w/2) <= spot]
    
    def _fmt_neg(x): return f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"
    
    def _add_rows(rows):
        for _, r in rows.iterrows():
            color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
            pct = (abs(r['Net_Dollars']) / max_val) * 100
            val_str = _fmt_neg(r["Net_Dollars"])
            html_out.append(f'<div class="zone-row"><div class="zone-label">${r.Zone_Low:.0f}-${r.Zone_High:.0f}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{pct:.1f}%"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div></div>')

    _add_rows(upper_zones)
    html_out.append(f'<div class="price-divider"><div class="price-badge">SPOT: ${spot:,.2f}</div></div>')
    _add_rows(lower_zones)
    
    html_out.append('</div>')
    return "".join(html_out)

def generate_expiry_buckets_html(df):
    """Calculates time buckets and returns the HTML string."""
    f = df.copy()
    order_type_col = "Order Type" if "Order Type" in f.columns else "Order type"
    
    # Calculate Signed Dollars
    f["Signed Dollars"] = np.where(f[order_type_col].isin(["Calls Bought", "Puts Sold"]), 1, -1) * f["Dollars"].fillna(0.0)
    
    days_diff = (pd.to_datetime(f["Expiry_DT"]).dt.date - date.today()).apply(lambda x: x.days)
    
    f["Bucket"] = pd.cut(days_diff, bins=SZ_BUCKET_BINS, labels=SZ_BUCKET_LABELS, include_lowest=True)
    
    agg = f.groupby("Bucket").agg(
        Net_Dollars=("Signed Dollars","sum"), 
        Trades=("Signed Dollars","count")
    ).reset_index()
    
    max_val = max(1.0, agg["Net_Dollars"].abs().max())
    html_out = []
    
    def _fmt_neg(x): return f"(${abs(x):,.0f})" if x < 0 else f"${x:,.0f}"

    for _, r in agg.iterrows():
        color = "zone-bull" if r["Net_Dollars"] >= 0 else "zone-bear"
        pct = (abs(r['Net_Dollars']) / max_val) * 100
        val_str = _fmt_neg(r["Net_Dollars"])
        html_out.append(f'<div class="zone-row"><div class="zone-label">{r.Bucket}</div><div class="zone-wrapper"><div class="zone-bar {color}" style="width:{pct:.1f}%"></div><div class="zone-value">{val_str} | n={int(r.Trades)}</div></div></div>')
    
    return "".join(html_out)