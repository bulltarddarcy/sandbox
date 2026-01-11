import pandas as pd
import utils_darcy
# Assuming these are your specific imports based on our previous discussions
from config import PARQUET_SECTOR_ROTATION
from data_loader import fetch_and_process_universe 

def run_sector_analysis():
    print(f"Loading Sector Rotation data from: {PARQUET_SECTOR_ROTATION}")
    
    # 1. Load the data
    # This currently sets 'Date' as the Index
    sector_df = fetch_and_process_universe(PARQUET_SECTOR_ROTATION)
    
    if sector_df is None or sector_df.empty:
        print("Error: No data loaded for Sector Rotation.")
        return

    # ==============================================================================
    # THE FIX: Reset Index to make Date a column
    # ==============================================================================
    # Check if Date is in the index (not a column) and reset it
    if 'Date' not in sector_df.columns and 'DATE' not in sector_df.columns:
        print(" > Fix applied: Resetting index to move Date into columns...")
        sector_df = sector_df.reset_index()

    # Standardize the column name to 'DATE' for utils_darcy compatibility
    # This handles cases where reset_index() creates a column named 'Date' or 'index'
    sector_df.rename(columns={'Date': 'DATE', 'index': 'DATE'}, inplace=True)
    # ==============================================================================

    print(f" > Data prepared. Columns: {list(sector_df.columns)}")

    # 2. Run the Analysis
    # Now utils_darcy will find the 'DATE' column it needs
    try:
        divergence_data = utils_darcy.prepare_data(sector_df)
        
        if divergence_data is None:
            print("Warning: prepare_data returned None (divergence check failed).")
        else:
            print(" > Divergence analysis complete.")
            # ... rest of your logic to process/save results ...
            
    except Exception as e:
        print(f"Critical Error during analysis: {e}")

if __name__ == "__main__":
    run_sector_analysis()
