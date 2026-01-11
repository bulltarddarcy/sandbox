import streamlit as st
import pandas as pd
from datetime import timedelta, date

# --- MODULE IMPORTS ---
import utils_options as uo
import utils_shared as us

# ==========================================
# 1. DATABASE APP
# ==========================================
def run_database_app(df):
    st.title("📂 Database")
    
    # Get max date from the dataframe specifically for Options data
    max_data_date = uo.get_max_trade_date(df)
    
    # Initialize State
    uo.initialize_database_state(max_data_date)

    def save_db_state(key, saved_key):
        if key in st.session_state:
            st.session_state[saved_key] = st.session_state[key]
    
    # UI Inputs
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        db_ticker = st.text_input("Ticker (blank=all)", value=st.session_state.saved_db_ticker, key="db_ticker_input", on_change=save_db_state, args=("db_ticker_input", "saved_db_ticker")).strip().upper()
    with c2: 
        start_date = st.date_input("Trade Start Date", value=st.session_state.saved_db_start, key="db_start", on_change=save_db_state, args=("db_start", "saved_db_start"))
    with c3: 
        end_date = st.date_input("Trade End Date", value=st.session_state.saved_db_end, key="db_end", on_change=save_db_state, args=("db_end", "saved_db_end"))
    with c4:
        db_exp_end = st.date_input("Expiration Range (end)", value=st.session_state.saved_db_exp, key="db_exp", on_change=save_db_state, args=("db_exp", "saved_db_exp"))
    
    ot1, ot2, ot3, _ = st.columns([1.5, 1.5, 1.5, 5.5])
    with ot1: 
        inc_cb = st.checkbox("Calls Bought", value=st.session_state.saved_db_inc_cb, key="db_inc_cb", on_change=save_db_state, args=("db_inc_cb", "saved_db_inc_cb"))
    with ot2: 
        inc_ps = st.checkbox("Puts Sold", value=st.session_state.saved_db_inc_ps, key="db_inc_ps", on_change=save_db_state, args=("db_inc_ps", "saved_db_inc_ps"))
    with ot3: 
        inc_pb = st.checkbox("Puts Bought", value=st.session_state.saved_db_inc_pb, key="db_inc_pb", on_change=save_db_state, args=("db_inc_pb", "saved_db_inc_pb"))
    
    # Filter Data
    filtered_df = uo.filter_database_trades(
        df, db_ticker, start_date, end_date, db_exp_end, inc_cb, inc_ps, inc_pb
    )
    
    if filtered_df.empty:
        st.warning("No data found matching these filters.")
        return
        
    # Display Styled Dataframe
    st.subheader("Non-Expired Trades")
    st.caption("⚠️ User should check OI to confirm trades are still open")
    
    styled_df = uo.get_database_styled_view(filtered_df)
    
    st.dataframe(
        styled_df, 
        use_container_width=True, 
        hide_index=True, 
        height=us.get_table_height(filtered_df, max_rows=uo.DB_TABLE_MAX_ROWS)
    )
    st.markdown("<br><br><br>", unsafe_allow_html=True)


# ==========================================
# 2. RANKINGS APP
# ==========================================
def run_rankings_app(df):
    st.title("🏆 Rankings")
    max_data_date = uo.get_max_trade_date(df)
    
    # Use Constant for Lookback (No hardcoded '14')
    start_default = max_data_date - timedelta(days=uo.RANK_LOOKBACK_DAYS)
    
    uo.initialize_rankings_state(start_default, max_data_date)

    def save_rank_state(key, saved_key):
        if key in st.session_state:
            st.session_state[saved_key] = st.session_state[key]
    
    # UI Inputs
    c1, c2, c3, c4 = st.columns([1, 1, 0.7, 1.3], gap="small")
    with c1: 
        rank_start = st.date_input("Trade Start Date", value=st.session_state.saved_rank_start, key="rank_start", on_change=save_rank_state, args=("rank_start", "saved_rank_start"))
    with c2: 
        rank_end = st.date_input("Trade End Date", value=st.session_state.saved_rank_end, key="rank_end", on_change=save_rank_state, args=("rank_end", "saved_rank_end"))
    with c3: 
        limit = st.number_input("Limit", value=st.session_state.saved_rank_limit, min_value=1, max_value=200, key="rank_limit", on_change=save_rank_state, args=("rank_limit", "saved_rank_limit"))
    with c4: 
        # Dynamic Options from Constants
        mc_options = list(uo.RANK_MC_THRESHOLDS.keys())
        default_mc_index = mc_options.index("10B") if "10B" in mc_options else 0
        
        min_mkt_cap_rank = st.selectbox("Min Market Cap", mc_options, index=default_mc_index, key="rank_mc", on_change=save_rank_state, args=("rank_mc", "saved_rank_mc"))
        filter_ema = st.checkbox("Hide < 8 EMA", value=False, key="rank_ema", on_change=save_rank_state, args=("rank_ema", "saved_rank_ema"))
        
    # Filter Data
    f_filtered = uo.filter_rankings_data(df, rank_start, rank_end)
    
    if f_filtered.empty:
        st.warning("No trades found matching these dates.")
        return

    tab_rank, tab_ideas, tab_vol = st.tabs(["🧠 Smart Money", "💡 Top 3", "🤡 Bulltard"])

    # Get Threshold Value from Constants
    mc_thresh = uo.RANK_MC_THRESHOLDS.get(min_mkt_cap_rank, 1e10)

    # Calculate Smart Money Scores
    top_bulls, top_bears, valid_data = uo.calculate_smart_money_score(df, rank_start, rank_end, mc_thresh, filter_ema, limit)

    # --- TAB 1: SMART MONEY ---
    with tab_rank:
        if valid_data.empty:
            st.warning("Not enough data for Smart Money scores.")
        else:
            sm_config = {
                "Symbol": st.column_config.TextColumn("Ticker", width=60),
                "Score": st.column_config.ProgressColumn("Score", format="%d", min_value=0, max_value=100),
                "Trade_Count": st.column_config.NumberColumn("Qty", width=50),
                "Last Trade": st.column_config.TextColumn("Last", width=70)
            }
            cols_to_show = ["Symbol", "Score", "Trade_Count", "Last Trade"]
            
            sm1, sm2 = st.columns(2, gap="large")
            with sm1:
                st.markdown(f"<div style='color: #71d28a; font-weight:bold;'>Top Bullish Scores</div>", unsafe_allow_html=True)
                if not top_bulls.empty:
                    st.dataframe(top_bulls[cols_to_show], use_container_width=True, hide_index=True, column_config=sm_config, height=us.get_table_height(top_bulls, max_rows=100))
            
            with sm2:
                st.markdown(f"<div style='color: #f29ca0; font-weight:bold;'>Top Bearish Scores</div>", unsafe_allow_html=True)
                if not top_bears.empty:
                    st.dataframe(top_bears[cols_to_show], use_container_width=True, hide_index=True, column_config=sm_config, height=us.get_table_height(top_bears, max_rows=100))
        st.markdown("<br><br>", unsafe_allow_html=True)

    # --- TAB 2: TOP IDEAS ---
    with tab_ideas:
        if top_bulls.empty:
            st.info("No Bullish candidates found to analyze.")
        else:
            st.caption(f"ℹ️ Analyzing the Top {len(top_bulls)} 'Smart Money' tickers for confluence...")
            
            best_ideas = uo.generate_top_ideas(top_bulls, df)
            
            cols = st.columns(3)
            for i, cand in enumerate(best_ideas):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"### #{i+1} {cand['Ticker']}")
                        st.metric("Conviction", f"{cand['Score']:.1f}/10", f"${cand['Price']:.2f}")
                        st.markdown("**Strategy:**")
                        
                        if cand['Suggestions']['Sell Puts']:
                            st.success(f"🛡️ **Sell Put:** {cand['Suggestions']['Sell Puts']}")
                        if cand['Suggestions']['Buy Calls']:
                            st.info(f"🟢 **Buy Call:** {cand['Suggestions']['Buy Calls']}")
                            
                        st.markdown("---")
                        for r in cand['Reasons']:
                            st.caption(f"• {r}")
        st.markdown("<br><br>", unsafe_allow_html=True)

    # --- TAB 3: VOLUME RANKINGS (Legacy) ---
    with tab_vol:
        st.caption("ℹ️ Legacy Methodology: Score = (Calls + Puts Sold) - (Puts Bought).")
        
        bull_df, bear_df = uo.calculate_volume_rankings(f_filtered, mc_thresh, filter_ema, limit)
        
        rank_col_config = {
            "Symbol": st.column_config.TextColumn("Symbol", width=60),
            "Trade Count": st.column_config.NumberColumn("#", width=50),
            "Last Trade": st.column_config.TextColumn("Last Trade", width=90),
            "Score": st.column_config.NumberColumn("Score", width=50),
        }
        cols_final = ["Symbol", "Trade Count", "Last Trade", "Score"]
        
        v1, v2 = st.columns(2)
        with v1:
            st.markdown(f"<div style='color: #71d28a; font-weight:bold;'>Bullish Volume</div>", unsafe_allow_html=True)
            if not bull_df.empty:
                st.dataframe(bull_df[cols_final], use_container_width=True, hide_index=True, column_config=rank_col_config, height=us.get_table_height(bull_df, max_rows=100))
        with v2:
            st.markdown(f"<div style='color: #f29ca0; font-weight:bold;'>Bearish Volume</div>", unsafe_allow_html=True)
            if not bear_df.empty:
                st.dataframe(bear_df[cols_final], use_container_width=True, hide_index=True, column_config=rank_col_config, height=us.get_table_height(bear_df, max_rows=100))
        st.markdown("<br><br>", unsafe_allow_html=True)


# ==========================================
# 3. PIVOT TABLES APP
# ==========================================
def run_pivot_tables_app(df):
    st.title("🎯 Pivot Tables")
    max_data_date = uo.get_max_trade_date(df)
    
    # 1. Initialize State via Utils
    uo.initialize_pivot_state(max_data_date, max_data_date)

    def save_pv_state(key, saved_key):
        if key in st.session_state:
            st.session_state[saved_key] = st.session_state[key]

    # 2. UI: Filters & Calculator
    col_filters, col_calculator = st.columns([1, 1], gap="medium")
    
    with col_filters:
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>🔍 Filters</h4>", unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1: 
            td_start = st.date_input("Trade Start Date", value=st.session_state.saved_pv_start, key="pv_start", on_change=save_pv_state, args=("pv_start", "saved_pv_start"))
        with fc2: 
            td_end = st.date_input("Trade End Date", value=st.session_state.saved_pv_end, key="pv_end", on_change=save_pv_state, args=("pv_end", "saved_pv_end"))
        with fc3: 
            ticker_filter = st.text_input("Ticker (blank=all)", value=st.session_state.saved_pv_ticker, key="pv_ticker", on_change=save_pv_state, args=("pv_ticker", "saved_pv_ticker")).strip().upper()
        
        fc4, fc5, fc6 = st.columns(3)
        with fc4: 
            # Use Keys from Constants
            opts_not = list(uo.PIVOT_NOTIONAL_MAP.keys())
            curr_not = st.session_state.saved_pv_notional
            idx_not = opts_not.index(curr_not) if curr_not in opts_not else 0
            sel_not = st.selectbox("Min Dollars", options=opts_not, index=idx_not, key="pv_notional", on_change=save_pv_state, args=("pv_notional", "saved_pv_notional"))
            min_notional = uo.PIVOT_NOTIONAL_MAP[sel_not]
            
        with fc5: 
            # Use Keys from Constants
            opts_mc = list(uo.PIVOT_MC_MAP.keys())
            curr_mc = st.session_state.saved_pv_mkt_cap
            idx_mc = opts_mc.index(curr_mc) if curr_mc in opts_mc else 0
            sel_mc = st.selectbox("Mkt Cap Min", options=opts_mc, index=idx_mc, key="pv_mkt_cap", on_change=save_pv_state, args=("pv_mkt_cap", "saved_pv_mkt_cap"))
            min_mkt_cap = uo.PIVOT_MC_MAP[sel_mc]
            
        with fc6: 
            opts_ema = ["All", "Yes"]
            curr_ema = st.session_state.saved_pv_ema
            idx_ema = opts_ema.index(curr_ema) if curr_ema in opts_ema else 0
            ema_filter = st.selectbox("Over 21 Day EMA", options=opts_ema, index=idx_ema, key="pv_ema_filter", on_change=save_pv_state, args=("pv_ema_filter", "saved_pv_ema"))

    with col_calculator:
        st.markdown("<h4 style='font-size: 1rem; margin-top: 0; margin-bottom: 10px;'>💰 Puts Sold Calculator</h4>", unsafe_allow_html=True)
        
        cc1, cc2, cc3 = st.columns(3)
        with cc1: c_strike = st.number_input("Strike Price", min_value=0.01, value=st.session_state.saved_calc_strike, step=1.0, format="%.2f", key="calc_strike", on_change=save_pv_state, args=("calc_strike", "saved_calc_strike"))
        with cc2: c_premium = st.number_input("Premium", min_value=0.00, value=st.session_state.saved_calc_premium, step=0.05, format="%.2f", key="calc_premium", on_change=save_pv_state, args=("calc_premium", "saved_calc_premium"))
        with cc3: c_expiry = st.date_input("Expiration", value=st.session_state.saved_calc_expiry, key="calc_expiry", on_change=save_pv_state, args=("calc_expiry", "saved_calc_expiry"))
        
        dte = (c_expiry - date.today()).days
        coc_ret = (c_premium / c_strike) * 100 if c_strike > 0 else 0.0
        annual_ret = (coc_ret / dte) * 365 if dte > 0 else 0.0

        st.session_state["calc_out_ann"] = f"{annual_ret:.1f}%"
        st.session_state["calc_out_coc"] = f"{coc_ret:.1f}%"
        st.session_state["calc_out_dte"] = str(max(0, dte))

        cc4, cc5, cc6 = st.columns(3)
        with cc4: st.text_input("Annualised Return", key="calc_out_ann")
        with cc5: st.text_input("Cash on Cash Return", key="calc_out_coc")
        with cc6: st.text_input("Days to Expiration", key="calc_out_dte")

    st.markdown("""
    <div style="display: flex; gap: 20px; font-size: 14px; margin-top: 10px; margin-bottom: 20px; align-items: center;">
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#b7e1cd"></div> This Friday</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#fce8b2"></div> Next Friday</div>
        <div style="display: flex; align-items: center; gap: 6px;"><div style="width: 14px; height: 14px; border-radius: 3px; background:#f4c7c3"></div> Two Fridays</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="light-note" style="margin-top: 5px;">ℹ️ Market Cap filtering can be buggy. If empty, reset \'Mkt Cap Min\' to 0B.</div>', unsafe_allow_html=True)
    st.markdown('<div class="light-note" style="margin-top: 5px;">ℹ️ Scroll down to see the Risk Reversals table.</div>', unsafe_allow_html=True)

    # 3. Data Processing via Utils
    d_range = df[(df["Trade Date"].dt.date >= td_start) & (df["Trade Date"].dt.date <= td_end)].copy()
    if d_range.empty: return

    # Helper: Split & Match RR
    cb_pool, ps_pool, pb_pool, df_rr = uo.generate_pivot_pools(d_range)

    # Helper: Apply Filters
    df_cb_f = uo.filter_pivot_dataframe(cb_pool, ticker_filter, min_notional, min_mkt_cap, ema_filter)
    df_ps_f = uo.filter_pivot_dataframe(ps_pool, ticker_filter, min_notional, min_mkt_cap, ema_filter)
    df_pb_f = uo.filter_pivot_dataframe(pb_pool, ticker_filter, min_notional, min_mkt_cap, ema_filter)
    df_rr_f = uo.filter_pivot_dataframe(df_rr, ticker_filter, min_notional, min_mkt_cap, ema_filter)

    # 4. Display Tables
    row1_c1, row1_c2, row1_c3 = st.columns(3)
    
    with row1_c1:
        st.subheader("Calls Bought")
        tbl = uo.get_pivot_styled_view(df_cb_f)
        if not tbl.empty: 
            st.dataframe(tbl.style.format(uo.PIVOT_TABLE_FMT).map(uo.highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=us.get_table_height(tbl, max_rows=50), column_config=uo.COLUMN_CONFIG_PIVOT)
            
    with row1_c2:
        st.subheader("Puts Sold")
        tbl = uo.get_pivot_styled_view(df_ps_f)
        if not tbl.empty: 
            st.dataframe(tbl.style.format(uo.PIVOT_TABLE_FMT).map(uo.highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=us.get_table_height(tbl, max_rows=50), column_config=uo.COLUMN_CONFIG_PIVOT)
            
    with row1_c3:
        st.subheader("Puts Bought")
        tbl = uo.get_pivot_styled_view(df_pb_f)
        if not tbl.empty: 
            st.dataframe(tbl.style.format(uo.PIVOT_TABLE_FMT).map(uo.highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=us.get_table_height(tbl, max_rows=50), column_config=uo.COLUMN_CONFIG_PIVOT)
    
    st.subheader("Risk Reversals")
    tbl_rr = uo.get_pivot_styled_view(df_rr_f, is_rr=True)
    if not tbl_rr.empty: 
        st.dataframe(tbl_rr.style.format(uo.PIVOT_TABLE_FMT).map(uo.highlight_expiry, subset=["Expiry_Table"]), use_container_width=True, hide_index=True, height=us.get_table_height(tbl_rr, max_rows=50), column_config=uo.COLUMN_CONFIG_PIVOT)
        st.markdown("<br><br>", unsafe_allow_html=True)
    else: 
        st.caption("No matched RR pairs found.")


# ==========================================
# 4. STRIKE ZONES APP
# ==========================================
def run_strike_zones_app(df):
    st.title("📊 Strike Zones")
    
    # 1. Initialize State via Utils
    exp_range_default = (date.today() + timedelta(days=uo.SZ_DEFAULT_EXP_OFFSET))
    uo.initialize_strike_zone_state(exp_range_default)

    def save_sz_state(key, saved_key):
        if key in st.session_state:
            st.session_state[saved_key] = st.session_state[key]
    
    # 2. UI Inputs
    col_settings, col_visuals = st.columns([1, 2.5], gap="large")
    
    with col_settings:
        ticker = st.text_input("Ticker", value=st.session_state.saved_sz_ticker, key="sz_ticker", on_change=save_sz_state, args=("sz_ticker", "saved_sz_ticker")).strip().upper()
        td_start = st.date_input("Trade Date (start)", value=st.session_state.saved_sz_start, key="sz_start", on_change=save_sz_state, args=("sz_start", "saved_sz_start"))
        td_end = st.date_input("Trade Date (end)", value=st.session_state.saved_sz_end, key="sz_end", on_change=save_sz_state, args=("sz_end", "saved_sz_end"))
        exp_end = st.date_input("Exp. Range (end)", value=st.session_state.saved_sz_exp, key="sz_exp", on_change=save_sz_state, args=("sz_exp", "saved_sz_exp"))
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            st.markdown("**View Mode**")
            view_mode = st.radio("Select View", ["Price Zones", "Expiry Buckets"], index=0 if st.session_state.saved_sz_view == "Price Zones" else 1, label_visibility="collapsed", key="sz_view", on_change=save_sz_state, args=("sz_view", "saved_sz_view"))
            
            st.markdown("**Zone Width**")
            width_mode = st.radio("Select Sizing", ["Auto", "Fixed"], index=0 if st.session_state.saved_sz_width_mode == "Auto" else 1, label_visibility="collapsed", key="sz_width_mode", on_change=save_sz_state, args=("sz_width_mode", "saved_sz_width_mode"))
            if width_mode == "Fixed": 
                fixed_size_choice = st.select_slider("Fixed bucket size ($)", options=[1, 5, 10, 25, 50, 100], value=st.session_state.saved_sz_fixed, key="sz_fixed", on_change=save_sz_state, args=("sz_fixed", "saved_sz_fixed"))
            else: fixed_size_choice = uo.SZ_DEFAULT_FIXED_SIZE
        
        with c_sub2:
            st.markdown("**Include**")
            inc_cb = st.checkbox("Calls Bought", value=st.session_state.saved_sz_inc_cb, key="sz_inc_cb", on_change=save_sz_state, args=("sz_inc_cb", "saved_sz_inc_cb"))
            inc_ps = st.checkbox("Puts Sold", value=st.session_state.saved_sz_inc_ps, key="sz_inc_ps", on_change=save_sz_state, args=("sz_inc_ps", "saved_sz_inc_ps"))
            inc_pb = st.checkbox("Puts Bought", value=st.session_state.saved_sz_inc_pb, key="sz_inc_pb", on_change=save_sz_state, args=("sz_inc_pb", "saved_sz_inc_pb"))
            
    with col_visuals:
        chart_container = st.container()

    # 3. Filter Data via Utils
    edit_pool_raw = uo.filter_strike_zone_data(df, ticker, td_start, td_end, exp_end, inc_cb, inc_ps, inc_pb)
    
    if edit_pool_raw.empty:
        with col_visuals:
            st.warning("No trades match current filters.")
        return

    # 4. Data Editor (UI Component)
    # We keep this in main because it interacts directly with the user
    order_type_col = "Order Type" if "Order Type" in edit_pool_raw.columns else "Order type"
    editor_input = edit_pool_raw[["Include", "Trade Date", order_type_col, "Symbol", "Strike", "Expiry_DT", "Contracts", "Dollars"]].copy()
    
    editor_input["Dollars"] = pd.to_numeric(editor_input["Dollars"], errors='coerce').fillna(0)
    editor_input["Contracts"] = pd.to_numeric(editor_input["Contracts"], errors='coerce').fillna(0)

    column_configuration = {
        "Include": st.column_config.CheckboxColumn("Include", default=True),
        "Trade Date": st.column_config.DateColumn("Trade Date", format="DD MMM YY"),
        "Expiry_DT": st.column_config.DateColumn("Expiry", format="DD MMM YY"),
        "Dollars": st.column_config.NumberColumn("Dollars", format="$%d"),
        "Contracts": st.column_config.NumberColumn("Qty", format="%d"),
        order_type_col: st.column_config.TextColumn("Order Type"),
        "Symbol": st.column_config.TextColumn("Symbol"),
        "Strike": st.column_config.TextColumn("Strike"),
    }
    
    st.subheader("Data Table & Selection")
    
    edited_df = st.data_editor(
        editor_input,
        column_config=column_configuration,
        disabled=["Trade Date", order_type_col, "Symbol", "Strike", "Expiry_DT", "Contracts", "Dollars"],
        hide_index=True,
        use_container_width=True,
        key="sz_editor"
    )
    
    # Apply Editor Mask
    f_final = edit_pool_raw[edited_df["Include"]].copy()
    st.markdown("<br><br>", unsafe_allow_html=True)

    # 5. Render Chart via Utils
    with chart_container:
        if f_final.empty:
            st.info("No rows selected. Check the 'Include' boxes below.")
        else:
            # Get Technicals
            spot, ema8, ema21, sma200 = uo.get_strike_zone_technicals(ticker)
            
            # Helper for badge text
            def pct_from_spot(x):
                if x is None or np.isnan(x): return "—"
                return f"{(x/spot-1)*100:+.1f}%"
            
            badges = [f'<span class="price-badge-header">Price: ${spot:,.2f}</span>']
            if ema8: badges.append(f'<span class="badge">EMA(8): ${ema8:,.2f} ({pct_from_spot(ema8)})</span>')
            if ema21: badges.append(f'<span class="badge">EMA(21): ${ema21:,.2f} ({pct_from_spot(ema21)})</span>')
            if sma200: badges.append(f'<span class="badge">SMA(200): ${sma200:,.2f} ({pct_from_spot(sma200)})</span>')
            st.markdown('<div class="metric-row">' + "".join(badges) + "</div>", unsafe_allow_html=True)

            # Generate HTML based on view mode
            if view_mode == "Price Zones":
                html_code = uo.generate_price_zones_html(f_final, spot, width_mode, fixed_size_choice)
                st.markdown(html_code, unsafe_allow_html=True)
            else:
                html_code = uo.generate_expiry_buckets_html(f_final)
                st.markdown(html_code, unsafe_allow_html=True)
            
            st.caption("ℹ️ You can exclude individual trades from the graphic by unchecking them in the Data Tables box below.")