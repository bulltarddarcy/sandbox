import streamlit as st
import pandas as pd
from datetime import date

# --- MODULE IMPORTS ---
import main_options  # New module for Options apps
import main_darcy    # Reduced module for Darcy/Sentiment
import main_sector   # Existing Sector module
import utils_darcy as ud  # For global data loading & health checks

# ==========================================
# 0. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Trading Toolbox", layout="wide", page_icon="💎")

# ==========================================
# 1. CSS STYLING
# ==========================================
st.markdown("""<style>
.block-container{padding-top:3.5rem;padding-bottom:1rem;}
.zones-panel{padding:14px 0; border-radius:10px;}
.zone-row{display:flex; align-items:center; gap:10px; margin:8px 0;}
.zone-label{width:90px; font-weight:700; text-align:right; flex-shrink: 0; font-size: 13px;}
.zone-wrapper{
    flex-grow: 1; 
    position: relative; 
    height: 24px; 
    background-color: rgba(0,0,0,0.03);
    border-radius: 4px;
    overflow: hidden;
}
.zone-bar{
    position: absolute;
    left: 0; 
    top: 0; 
    bottom: 0; 
    z-index: 1;
    border-radius: 3px;
    opacity: 0.65;
}
.zone-bull{background-color: #71d28a;}
.zone-bear{background-color: #f29ca0;}
.zone-value{
    position: absolute;
    right: 8px;
    top: 0; bottom: 0;
    display: flex; align-items: center;
    font-size: 11px; font-weight: 600;
    color: #444; z-index: 2;
}
.price-divider{
    display: flex; justify-content: center; margin: 12px 0; position: relative;
}
.price-divider::before{
    content: ""; position: absolute; top: 50%; left: 0; right: 0;
    border-top: 1px dashed #ccc; z-index: 0;
}
.price-badge{
    background: #fff; border: 1px solid #ccc; border-radius: 12px;
    padding: 2px 12px; font-size: 12px; font-weight: 700; color: #555;
    z-index: 1;
}
.metric-row{
    display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; align-items: center;
}
.metric-row .badge{
    background: #f0f2f6; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 500;
}
.metric-row .price-badge-header{
    font-size: 16px; font-weight: 700; margin-right: 10px;
}
/* Table Styling */
div[data-testid="stDataFrame"] div[data-testid="stTable"] { font-size: 0.85rem; }
/* Hide default hamburger */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. NAVIGATION SIDEBAR
# ==========================================
with st.sidebar:
    st.header("💎 Mr. Darcy's Manor")
    
    app_mode = st.radio("Select App:", [
        "Database",
        "Rankings",
        "Pivot Tables", 
        "Strike Zones",
        "Market Sentiment",
        "Sector Rotation",
        "System Health"
    ])
    
    st.info("Last Updated: Jan 2026")

# ==========================================
# 3. GLOBAL DATA LOADING
# ==========================================
# Only load heavy options data if we are in an "Options" app
OPTIONS_APPS = ["Database", "Rankings", "Pivot Tables", "Strike Zones", "Market Sentiment"]

if app_mode in OPTIONS_APPS:
    # 1. Load Options Data
    df_options = ud.load_parquet_and_clean("URL_OPTIONS_PARQUET", date_col="Trade Date")
    
    if df_options.empty:
        st.error("⚠️ Options Data could not be loaded. Please check 'System Health'.")
        # Initialize empty to prevent crashes
        df_options = pd.DataFrame(columns=["Trade Date", "Symbol", "Expiry_DT", "Strike", "Dollars", "Order Type"])
else:
    df_options = pd.DataFrame()

# ==========================================
# 4. APP ROUTING
# ==========================================

if app_mode == "Database":
    main_options.run_database_app(df_options)

elif app_mode == "Rankings":
    main_options.run_rankings_app(df_options)

elif app_mode == "Pivot Tables":
    main_options.run_pivot_tables_app(df_options)

elif app_mode == "Strike Zones":
    main_options.run_strike_zones_app(df_options)

elif app_mode == "Market Sentiment":
    main_darcy.run_sentiment_app(df_options)

elif app_mode == "Sector Rotation":
    # Sector app manages its own data loading internally
    main_sector.run_sector_rotation_app()

elif app_mode == "System Health":
    st.title("🛠 System Health")
    st.markdown("### Connection Status")
    
    # Check Secrets
    tm_key = "URL_TICKER_MAP"
    tm_url = st.secrets.get(tm_key, "")
    
    if not tm_url:
        st.markdown(f"❌ **Ticker Map**: Secret Missing")
    elif "drive.google.com" not in tm_url:
        st.markdown(f"⚠️ **Ticker Map**: Invalid URL")
    else:
         st.markdown(f"✅ **Ticker Map**: Connected")

    # Check Parquet Files (using Darcy Utils config)
    health_config = ud.get_parquet_config()
    all_good = True
    
    for name, key in health_config.items():
        url = st.secrets.get(key, "")
        
        if not url:
            st.markdown(f"❌ **{name}**: Secret Missing")
            all_good = False
        elif "drive.google.com" not in url:
             st.markdown(f"⚠️ **{name}**: Invalid URL Format")
             all_good = False
        else:
            status_icon = "✅"
            note = ""
            if "usp=drive_link" in url:
                status_icon = "⚠️" 
                note = "(drive_link)"
            st.markdown(f"{status_icon} **{name}**: Linked {note}")
    
    if all_good and tm_url:
        st.caption("All configurations look valid.")
    else:
        st.error("Configuration errors detected.")

# ==========================================
# 5. FOOTER
# ==========================================
st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("© 2026 Mr. Darcy's Manor")