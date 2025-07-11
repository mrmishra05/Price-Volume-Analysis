import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import io
import base64

# Try to import plotly with error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError as e:
    st.error(f"⚠️ Plotly not available: {e}")
    st.info("📊 Using Streamlit's built-in charts instead")
    PLOTLY_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="NSE Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 700;
    color: #1f2937;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #6b7280;
    margin-bottom: 2rem;
}
.metric-card {
    background: white;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #3b82f6;
    margin-bottom: 1rem;
}
/* Signal classes for styling table cells */
.signal-price-increasing-delivery-volume-increasing {
    background-color: #10b981; /* Strong Buy Green */
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.signal-price-not-increasing-delivery-volume-increasing {
    background-color: #3b82f6; /* Accumulation Blue */
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.signal-price-decreasing-volume-increasing {
    background-color: #8b5cf6; /* Bottom Fishing Purple */
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.signal-price-decreasing-volume-decreasing {
    background-color: #ef4444; /* Weak Sell Red */
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.signal-price-increasing-volume-not-increasing {
    background-color: #f59e0b; /* Weak Buy Orange */
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.signal-neutral-hold {
    background-color: #6b7280; /* Hold Grey */
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_google_sheets(sheet_url):
    """
    Load raw stock data from Google Sheets using the public CSV export URL.
    This function now handles raw column names and calculates daily deltas.
    """
    try:
        # Extract sheet ID and GID from the URL
        if '/d/' in sheet_url:
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            
            # Extract GID if present
            if 'gid=' in sheet_url:
                gid = sheet_url.split('gid=')[1].split('&')[0].split('#')[0]
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            else:
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        else:
            csv_url = sheet_url # Assume it's already a direct CSV export link
            
        st.sidebar.info(f"Attempting to load raw data from: {csv_url}")
        
        # Read the CSV data with proper error handling
        df = pd.read_csv(csv_url, encoding='utf-8')
        
        # --- Initial DataFrame checks ---
        if df.empty:
            st.error("❌ The CSV file loaded is empty. Please ensure your Google Sheet contains data.")
            return None

        st.sidebar.write("Debug: DataFrame loaded. Is empty?", df.empty)
        st.sidebar.write("Debug: DataFrame columns (raw):", df.columns.tolist())
        # Debug: Display dtypes more robustly
        st.sidebar.write("Debug: DataFrame dtypes (raw):")
        for col, dtype in df.dtypes.items():
            st.sidebar.write(f"- {col}: {dtype}")

        # Store original column names for debugging
        original_columns = df.columns.tolist()
        st.sidebar.write("**Original Columns Found:**")
        st.sidebar.write(original_columns)
        
        # Clean column names (lowercase, trim, replace spaces with underscores)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Check if columns are empty after cleaning (e.g., if CSV was truly malformed)
        if df.columns.empty:
            st.error("❌ The CSV file appears to be empty or malformed (no columns found after cleaning).")
            return None

        cleaned_columns = df.columns.tolist()
        st.sidebar.write("**Cleaned Columns:**")
        st.sidebar.write(cleaned_columns)
        
        # --- Column Mapping and Type Conversion ---
        
        # Date column
        date_candidates = ['date', 'datetime', 'timestamp', 'day', 'trading_date', 'trade_date', 'date1']
        date_column = None
        for candidate in date_candidates:
            if candidate in df.columns:
                date_column = candidate
                break
        if date_column:
            df['date'] = pd.to_datetime(df[date_column], errors='coerce')
            df = df.dropna(subset=['date']) # Remove rows with invalid dates
            if df['date'].empty: # Check if all dates became NaT after coercion
                st.error("❌ All dates in the 'date' column are invalid or missing. Please check date format in your sheet.")
                return None
        else:
            st.sidebar.error("❌ No date column found! Please ensure your sheet has a column with dates named one of: " + ", ".join(date_candidates))
            return None
        
        # Symbol column
        symbol_candidates = ['symbol', 'stock_symbol', 'ticker', 'scrip', 'instrument']
        symbol_column = None
        for candidate in symbol_candidates:
            if candidate in df.columns:
                symbol_column = candidate
                break
        if symbol_column and symbol_column != 'symbol':
            df = df.rename(columns={symbol_column: 'symbol'})
        elif 'symbol' not in df.columns:
            st.sidebar.error("❌ No symbol column found! Please ensure your sheet has a column with symbols named one of: " + ", ".join(symbol_candidates))
            return None
        
        # Series column (Crucial for uniqueness with Symbol)
        series_candidates = ['series', 'stock_series']
        series_column = None
        for candidate in series_candidates:
            if candidate in df.columns:
                series_column = candidate
                break
        if series_column and series_column != 'series':
            df = df.rename(columns={series_column: 'series'})
        elif 'series' not in df.columns:
            # If no series column, create a dummy one to avoid errors in drop_duplicates
            df['series'] = 'EQ' # Default to EQ if not found
            st.sidebar.warning("⚠️ 'SERIES' column not found. Assuming all stocks are 'EQ' series for uniqueness.")

        # Numeric columns mapping and conversion
        numeric_cols_map = {
            'prev_close': ['prev_close', 'previous_close', 'prev_close_price'],
            'close_price': ['close_price', 'close', 'closing_price', 'last_price', 'ltp'],
            'volume': ['volume', 'ttl_trd_qnty', 'traded_quantity', 'qty'],
            'deliv_qty': ['deliv_qty', 'delivery_quantity', 'delivery_qty', 'del_qty'],
            'deliv_per': ['deliv_per', 'delivery_percentage', 'delivery_per'] # Directly map DELIV_PER
        }
        
        for standard_name, candidates in numeric_cols_map.items():
            found_column = None
            for candidate in candidates:
                if candidate in df.columns:
                    found_column = candidate
                    break
            if found_column:
                df[standard_name] = pd.to_numeric(df[found_column], errors='coerce')
                if found_column != standard_name:
                    df = df.rename(columns={found_column: standard_name})
            else:
                df[standard_name] = np.nan # Ensure column exists even if empty
                st.sidebar.warning(f"⚠️ Column '{standard_name}' (or its candidates) not found. Setting to NaN.")

        # Final validation for essential columns
        required_columns = ['symbol', 'date', 'close_price', 'volume', 'deliv_qty'] 
        
        # Check for missing columns and entirely NaN columns
        for col in required_columns:
            if col not in df.columns:
                st.error(f"❌ Missing required column: '{col}'. Please check your raw data headers.")
                return None
            # IMPORTANT: Check if the Series is empty before checking isnull().all()
            if df[col].empty:
                st.error(f"❌ Required column '{col}' is empty after initial processing. Cannot proceed.")
                return None
            if df[col].isnull().all():
                st.error(f"❌ Required column '{col}' is entirely empty (all NaN). Please check your raw data.")
                return None
        
        st.sidebar.write("Debug: All required columns are present and not entirely NaN.")

        # --- CRITICAL FIX: Ensure symbol and series are clean strings before unique_id creation ---
        # This addresses the "truth value of a Series is ambiguous" error
        
        try:
            # Clean symbol column - convert to string and handle NaN values
            df['symbol'] = df['symbol'].astype(str).str.strip()
            # Replace 'nan' strings with empty strings and filter out empty ones
            df['symbol'] = df['symbol'].replace('nan', '').replace('', np.nan)
            
            # Clean series column - convert to string and handle NaN values
            df['series'] = df['series'].astype(str).str.strip()
            df['series'] = df['series'].replace('nan', '').replace('', np.nan)
            
            # Remove rows where symbol is NaN/empty as they're not useful
            df = df.dropna(subset=['symbol'])
            
            # For series, if NaN, default to 'EQ'
            df['series'] = df['series'].fillna('EQ')
            
            # Final check: ensure we have data after cleaning
            if df.empty:
                st.error("❌ No valid data remains after cleaning symbol and series columns.")
                return None
            
            # Now create unique_id safely
            df['unique_id'] = df['symbol'] + '-' + df['series']
            
            st.sidebar.write("Debug: Symbol and Series columns processed to clean strings.")
            st.sidebar.write("Debug: Sample symbols (cleaned):", df['symbol'].head().tolist())
            st.sidebar.write("Debug: Sample series (cleaned):", df['series'].head().tolist())
            st.sidebar.write("Debug: Unique ID created. Sample unique_ids:", df['unique_id'].head().tolist())
            
        except Exception as e:
            st.error(f"Error during symbol/series cleaning or unique_id creation: {e}")
            st.write(f"Debug: Error type: {type(e)}, Message: {e}")
            return None

        # Ensure unique (symbol, series, date) pairs and sort data
        df = df.sort_values(by=['unique_id', 'date']).drop_duplicates(subset=['unique_id', 'date'], keep='last')
        st.sidebar.success(f"✅ Data processed to ensure unique (Symbol, Series, Date) pairs and sorted.")
        st.sidebar.write("Debug: DataFrame after de-duplication and sorting. Head:", df.head())

        # --- Calculate Daily Deltas (Price, Volume, Delivery Quantity) ---
        # These are calculated for every row relative to the previous trading day for that unique stock-series combination
        
        # Price Change
        df['daily_price_change_abs'] = df.groupby('unique_id')['close_price'].diff()
        # Handle division by zero/inf for percentage calculation
        price_prev = df.groupby('unique_id')['close_price'].shift(1)
        df['daily_price_change_pct'] = (df['daily_price_change_abs'] / price_prev) * 100
        df['daily_price_change_pct'] = df['daily_price_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Volume Change
        df['daily_volume_change_abs'] = df.groupby('unique_id')['volume'].diff()
        volume_prev = df.groupby('unique_id')['volume'].shift(1)
        df['daily_volume_change_pct'] = (df['daily_volume_change_abs'] / volume_prev) * 100
        df['daily_volume_change_pct'] = df['daily_volume_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Delivery Quantity Change
        df['daily_deliv_change_abs'] = df.groupby('unique_id')['deliv_qty'].diff()
        deliv_qty_prev = df.groupby('unique_id')['deliv_qty'].shift(1)
        df['daily_deliv_change_pct'] = (df['daily_deliv_change_abs'] / deliv_qty_prev) * 100
        df['daily_deliv_change_pct'] = df['daily_deliv_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)

        st.sidebar.success(f"✅ Daily deltas calculated. Total rows: {len(df)}")
        return df
    
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        st.write("**Debug Info:**")
        st.write(f"CSV URL: {csv_url if 'csv_url' in locals() else 'Not generated'}")
        st.write(f"Error type: {type(e)}")
        st.write(f"Error message: {e}")
        return None

def calculate_price_changes(df, timeframe, selected_date, start_date=None, end_date=None):
    """
    Calculates price and volume/delivery changes based on the selected timeframe.
    This function now uses the pre-calculated daily deltas for 'Today vs Yesterday'
    and calculates other timeframes dynamically.
    """
    df = df.copy()
    
    # Ensure 'date' column is datetime type for comparisons
    df['date'] = pd.to_datetime(df['date'])

    if timeframe == 'Today vs Yesterday':
        # For 'Today vs Yesterday', use the pre-calculated daily deltas
        # Filter for the selected date
        current_day_data = df[df['date'] == selected_date].drop_duplicates(subset=['unique_id'])
        
        # Rename the daily delta columns to the generic names expected by categorize_stocks
        if not current_day_data.empty:
            current_day_data = current_day_data.rename(columns={
                'daily_price_change_abs': 'price_change',
                'daily_price_change_pct': 'price_change_pct',
                'daily_volume_change_abs': 'volume_change', 
                'daily_deliv_change_abs': 'deliv_change',
                'daily_deliv_change_pct': 'deliv_change_pct'
            })
            # Select relevant columns for return
            return current_day_data[['symbol', 'series', 'date', 'close_price', 'volume', 'deliv_qty', 'price_change', 'price_change_pct', 'deliv_change', 'deliv_change_pct']]
        else:
            return pd.DataFrame() # Return empty if no data for selected date
    
    else: # For Weekly, Monthly, Custom Range, calculate dynamically
        # Determine the comparison date based on timeframe
        if timeframe == 'Weekly':
            compare_date = selected_date - timedelta(days=7)
            end_date_for_period = selected_date
        elif timeframe == 'Monthly':
            compare_date = selected_date - timedelta(days=30)
            end_date_for_period = selected_date
        elif timeframe == 'Custom Range':
            compare_date = start_date 
            end_date_for_period = end_date

        # Get data for the selected end date of the period
        current_period_data = df[df['date'] == end_date_for_period].drop_duplicates(subset=['unique_id'])

        # Find the most recent data on or before the comparison date for each unique_id
        past_period_data_candidates = df[df['date'] <= compare_date].sort_values('date').groupby('unique_id').last().reset_index()
        past_period_data = past_period_data_candidates.drop_duplicates(subset=['unique_id'])

        if not current_period_data.empty and not past_period_data.empty:
            merged = current_period_data.merge(past_period_data, on='unique_id', suffixes=('', '_prev_period'))
            merged = merged.drop_duplicates(subset=['unique_id']) # Final check for merge-induced duplicates

            # Calculate Price Change for the period
            merged['price_change'] = merged['close_price'] - merged['close_price_prev_period']
            price_prev_period = merged['close_price_prev_period'].replace(0, np.nan) # Avoid division by zero
            merged['price_change_pct'] = (merged['price_change'] / price_prev_period) * 100
            merged['price_change_pct'] = merged['price_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0) 

            # Calculate Volume Change for the period
            if 'volume' in merged.columns and 'volume_prev_period' in merged.columns:
                merged['volume_change'] = merged['volume'] - merged['volume_prev_period']
                volume_prev_period = merged['volume_prev_period'].replace(0, np.nan) # Avoid division by zero
                merged['volume_change_pct'] = (merged['volume_change'] / volume_prev_period) * 100
                merged['volume_change_pct'] = merged['volume_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                merged['volume_change'] = np.nan
                merged['volume_change_pct'] = np.nan
            
            # Calculate Delivery Quantity Change for the period
            if 'deliv_qty' in merged.columns and 'deliv_qty_prev_period' in merged.columns:
                merged['deliv_change'] = merged['deliv_qty'] - merged['deliv_qty_prev_period']
                deliv_qty_prev_period = merged['deliv_qty_prev_period'].replace(0, np.nan) # Avoid division by zero
                merged['deliv_change_pct'] = (merged['deliv_change'] / deliv_qty_prev_period) * 100
                merged['deliv_change_pct'] = merged['deliv_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                merged['deliv_change'] = np.nan
                merged['deliv_change_pct'] = np.nan

            return merged[['symbol', 'series', 'date', 'close_price', 'volume', 'deliv_qty', 'price_change', 'price_change_pct', 'deliv_change', 'deliv_change_pct']]
        else:
            return pd.DataFrame() # Return empty DataFrame if no data for the period
    
    return pd.DataFrame() # Fallback return

def categorize_stocks(df):
    """
    Categorize stocks based on price and volume change patterns
    as per the refined definitions.
    """
    df = df.copy()
    
    # Ensure required columns for categorization exist
    required_cols = ['price_change_pct', 'deliv_change_pct']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns for categorization: {missing_cols}")
        df['category'] = 'N/A'
        df['signal_strength'] = 'N/A'
        df['category_color'] = '#6B7280' # Default grey
        return df
    
    def get_category(row):
        price_change = row.get('price_change_pct', np.nan)
        deliv_change = row.get('deliv_change_pct', np.nan)
        
        # Handle NaN values gracefully, categorize as Neutral if data is missing
        if pd.isna(price_change) or pd.isna(deliv_change):
            return 'Neutral / Hold', 'NEUTRAL / HOLD', '#6B7280' # Grey

        # 1. Price increasing, Delivery volume increasing
        if price_change > 0 and deliv_change > 0:
            return 'Price Inc, Del Vol Inc', 'STRONG BUY', '#10B981' # Green
        
        # 2. Price not increasing, Delivery volume increasing
        # (Price change is zero or negative, Delivery volume is increasing)
        elif price_change <= 0 and deliv_change > 0:
            return 'Price Not Inc, Del Vol Inc', 'ACCUMULATION', '#3B82F6' # Blue
            
        # 3. Price decreasing, Volume increasing
        # (Price change is negative, Delivery volume is increasing)
        elif price_change < 0 and deliv_change > 0:
            return 'Price Dec, Vol Inc', 'BOTTOM FISHING', '#8B5CF6' # Purple
            
        # 4. Price decreasing, Volume decreasing
        # (Price change is negative, Delivery volume is decreasing)
        elif price_change < 0 and deliv_change < 0:
            return 'Price Dec, Vol Dec', 'WEAK SELL', '#EF4444' # Red
            
        # 5. Price increasing, Volume not increasing
        # (Price change is positive, Delivery volume is zero or negative)
        elif price_change > 0 and deliv_change <= 0:
            return 'Price Inc, Vol Not Inc', 'WEAK BUY', '#F59E0B' # Orange
            
        else:
            return 'Neutral / Hold', 'NEUTRAL / HOLD', '#6B7280' # Grey
    
    categories = df.apply(get_category, axis=1)
    df['category'] = [cat[0] for cat in categories]
    df['signal_strength'] = [cat[1] for cat in categories]
    df['category_color'] = [cat[2] for cat in categories]
    
    return df

def create_category_pie_chart(df):
    """Create pie chart for category distribution"""
    # Ensure category column exists before counting
    if 'category' not in df.columns:
        st.warning("Category column not found for pie chart.")
        return None

    category_counts = df['category'].value_counts()
    
    if PLOTLY_AVAILABLE:
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Stock Category Distribution",
            color_discrete_map={
                'Price Inc, Del Vol Inc': '#10B981',
                'Price Not Inc, Del Vol Inc': '#3B82F6',
                'Price Dec, Vol Inc': '#8B5CF6',
                'Price Dec, Vol Dec': '#EF4444',
                'Price Inc, Vol Not Inc': '#F59E0B',
                'Neutral / Hold': '#6B7280'
            }
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True) # Ensure legend is shown
        return fig
    else:
        # Fallback to Streamlit native chart
        st.subheader("Stock Category Distribution")
        st.bar_chart(category_counts)
        return None

# Main app
def main():
    st.markdown('<h1 class="main-header">📈 NSE Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze price and delivery volume movement patterns for strategic insights</p>', unsafe_allow_html=True)
    
    # Display plotly status
    if PLOTLY_AVAILABLE:
        st.success("✅ Plotly charts enabled")
    else:
        st.warning("⚠️ Using Streamlit native charts (Plotly not available)")
    
    # Sidebar for inputs
    st.sidebar.title("Configuration")
    
    # Google Sheets URL input
    sheet_url = st.sidebar.text_input(
        "Google Sheets URL (Raw Data)",
        placeholder="Paste your Google Sheets URL here (e.g., your 'NSE_2025' sheet)",
        help="Make sure your Google Sheet is publicly accessible. This should be the URL of your raw data sheet."
    )
    
    # Check if URL is provided
    if not sheet_url:
        st.warning("Please provide a Google Sheets URL to load data")
        st.info("📝 **Instructions:**\n1. Make your raw data Google Sheet (e.g., 'NSE_2025') publicly accessible.\n2. Copy the full URL from your browser.\n3. Paste it in the sidebar.\n\n**Expected Raw Column Names (any of these variants will work):**\n- **Date**: Date, Date1, DateTime, Timestamp, Day, Trading_Date\n- **Symbol**: Symbol, Stock_Symbol, Ticker, Scrip, Instrument\n- **Series**: SERIES, Stock_Series\n- **Close Price**: Close_Price, Close, Closing_Price, LAST_PRICE, LTP\n- **Previous Close**: PREV_CLOSE, Previous_Close, Prev_Close_Price\n- **Volume**: Volume, TTL_TRD_QNTY, Traded_Quantity, Qty\n- **Delivery Quantity**: DELIV_QTY, Delivery_Quantity, Del_Qty\n- **Delivery Percentage**: DELIV_PER, Delivery_Percentage, Delivery_Per")
        return
    
    # Load data
    df = load_data_from_google_sheets(sheet_url)
    if df is None or df.empty:
        st.error("Failed to load or process data from Google Sheets")
        st.info("**Troubleshooting:**\n1. Check if the sheet is publicly accessible.\n2. Verify the URL is correct.\n3. Ensure the sheet contains the expected raw columns and valid data.\n4. Check the column names match expected formats (case-insensitive, spaces replaced by underscores).")
        return
    
    # Display data info
    st.sidebar.success(f"✅ Raw data loaded and pre-processed: {len(df)} rows")
    
    # Show data preview
    with st.sidebar.expander("Processed Data Preview (First 5 rows)"):
        st.write("**Data after initial processing and daily delta calculation:**")
        st.dataframe(df.head())
    
    # Date and timeframe selection
    st.sidebar.subheader("Analysis Parameters")
    
    # Get available dates and remove NaT values
    available_dates = df['date'].dropna().dt.date.unique()
    available_dates = sorted([d for d in available_dates if pd.notna(d)])
    
    if len(available_dates) == 0:
        st.error("No valid dates found in the data after processing.")
        return
    
    max_date = max(available_dates)
    min_date = min(available_dates)
    
    # Date selection
    selected_date = st.sidebar.date_input(
        "Select Date for Analysis",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )
    selected_date = pd.to_datetime(selected_date)
    
    # Timeframe selection
    timeframe = st.sidebar.selectbox(
        "Comparison Timeframe",
        ["Today vs Yesterday", "Weekly", "Monthly", "Custom Range"]
    )
    
    # Custom date range for custom timeframe
    start_date_custom = None
    end_date_custom = None
    if timeframe == "Custom Range":
        start_date_custom = st.sidebar.date_input("Start Date", value=max_date - timedelta(days=30), min_value=min_date, max_value=max_date)
        end_date_custom = st.sidebar.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        start_date_custom = pd.to_datetime(start_date_custom)
        end_date_custom = pd.to_datetime(end_date_custom)
    
    # Process data based on timeframe
    # This step now calculates the deltas for the *selected* timeframe,
    # or uses the pre-calculated daily deltas for "Today vs Yesterday".
    processed_df = calculate_price_changes(df, timeframe, selected_date, start_date_custom, end_date_custom)
    
    if processed_df.empty:
        st.warning(f"No data available for the selected timeframe: {timeframe}")
        st.info("Try selecting a different date or timeframe.")
        return
    
    # Categorize stocks based on price and delivery volume patterns
    categorized_df = categorize_stocks(processed_df)
    
    # Main dashboard content
    st.subheader(f"Analysis for {timeframe}")
    st.write(f"**Selected Date:** {selected_date.strftime('%Y-%m-%d')}")
    if timeframe == "Custom Range":
        st.write(f"**Comparison Period:** {start_date_custom.strftime('%Y-%m-%d')} to {end_date_custom.strftime('%Y-%m-%d')}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_stocks = len(categorized_df)
        st.metric("Total Stocks", total_stocks)
    
    with col2:
        strong_buy_count = len(categorized_df[categorized_df['signal_strength'] == 'STRONG BUY'])
        st.metric("Strong Buy", strong_buy_count)
    
    with col3:
        accumulation_count = len(categorized_df[categorized_df['signal_strength'] == 'ACCUMULATION'])
        st.metric("Accumulation", accumulation_count)
    
    with col4:
        avg_price_change = categorized_df['price_change_pct'].mean()
        st.metric("Avg Price Change %", f"{avg_price_change:.2f}%")
    
    # Charts section
    st.subheader("Visual Analysis")
    
    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Category distribution pie chart
        pie_chart = create_category_pie_chart(categorized_df)
        if pie_chart:
            st.plotly_chart(pie_chart, use_container_width=True)
    
    with chart_col2:
        # Price change distribution histogram
        if PLOTLY_AVAILABLE:
            fig_hist = px.histogram(
                categorized_df,
                x='price_change_pct',
                nbins=20,
                title="Price Change Distribution",
                labels={'price_change_pct': 'Price Change %', 'count': 'Number of Stocks'}
            )
            fig_hist.update_layout(showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.subheader("Price Change Distribution")
            st.bar_chart(categorized_df['price_change_pct'])
    
    # Detailed results table
    st.subheader("Detailed Stock Analysis")
    
    # Filter options
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # Category filter
        categories = ['All'] + list(categorized_df['category'].unique())
        selected_category = st.selectbox("Filter by Category", categories)
    
    with filter_col2:
        # Signal strength filter
        signals = ['All'] + list(categorized_df['signal_strength'].unique())
        selected_signal = st.selectbox("Filter by Signal", signals)
    
    # Apply filters
    filtered_df = categorized_df.copy()
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    if selected_signal != 'All':
        filtered_df = filtered_df[filtered_df['signal_strength'] == selected_signal]
    
    # Sort by price change percentage (descending)
    filtered_df = filtered_df.sort_values('price_change_pct', ascending=False)
    
    # Display results
    st.write(f"Showing {len(filtered_df)} stocks out of {len(categorized_df)} total stocks")
    
    if not filtered_df.empty:
        # Prepare display dataframe
        display_df = filtered_df[[
            'symbol', 'series', 'close_price', 'volume', 'deliv_qty', 
            'price_change_pct', 'deliv_change_pct', 'signal_strength'
        ]].copy()
        
        # Round numeric columns
        display_df['close_price'] = display_df['close_price'].round(2)
        display_df['volume'] = display_df['volume'].astype(int)
        display_df['deliv_qty'] = display_df['deliv_qty'].astype(int)
        display_df['price_change_pct'] = display_df['price_change_pct'].round(2)
        display_df['deliv_change_pct'] = display_df['deliv_change_pct'].round(2)
        
        # Rename columns for better display
        display_df.columns = [
            'Symbol', 'Series', 'Close Price', 'Volume', 'Delivery Qty',
            'Price Change %', 'Delivery Change %', 'Signal'
        ]
        
        # Display the table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button for filtered results
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Filtered Results as CSV",
            data=csv,
            file_name=f"nse_analysis_{timeframe.lower().replace(' ', '_')}_{selected_date.strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No stocks match the selected filters.")
    
    # Summary insights
    st.subheader("Key Insights")
    
    if not categorized_df.empty:
        insights = []
        
        # Top performers
        top_gainers = categorized_df.nlargest(3, 'price_change_pct')
        if not top_gainers.empty:
            insights.append(f"🚀 **Top Gainers:** {', '.join(top_gainers['symbol'].tolist())}")
        
        # Strong buy signals
        strong_buys = categorized_df[categorized_df['signal_strength'] == 'STRONG BUY']
        if not strong_buys.empty:
            insights.append(f"💚 **Strong Buy Signals:** {len(strong_buys)} stocks showing price increase with delivery volume increase")
        
        # Accumulation signals
        accumulation = categorized_df[categorized_df['signal_strength'] == 'ACCUMULATION']
        if not accumulation.empty:
            insights.append(f"🔵 **Accumulation Signals:** {len(accumulation)} stocks showing delivery volume increase despite price not increasing")
        
        # Bottom fishing opportunities
        bottom_fishing = categorized_df[categorized_df['signal_strength'] == 'BOTTOM FISHING']
        if not bottom_fishing.empty:
            insights.append(f"🟣 **Bottom Fishing:** {len(bottom_fishing)} stocks showing volume increase despite price decrease")
        
        # Display insights
        for insight in insights:
            st.write(insight)
    
    # Trading strategy recommendations
    st.subheader("Trading Strategy Recommendations")
    
    strategy_recommendations = {
        'STRONG BUY': "Consider buying - both price and delivery volume are increasing, indicating strong momentum.",
        'ACCUMULATION': "Consider accumulating - increased delivery volume despite stable/declining price suggests institutional interest.",
        'BOTTOM FISHING': "Potential contrarian opportunity - increased volume during price decline might indicate bottoming out.",
        'WEAK BUY': "Cautious optimism - price is increasing but delivery volume is not supporting the move.",
        'WEAK SELL': "Consider selling - both price and delivery volume are decreasing, indicating weak momentum.",
        'NEUTRAL / HOLD': "Hold current positions - no clear directional signals."
    }
    
    for signal, recommendation in strategy_recommendations.items():
        count = len(categorized_df[categorized_df['signal_strength'] == signal])
        if count > 0:
            st.write(f"**{signal}** ({count} stocks): {recommendation}")
    
    # Disclaimer
    st.markdown("---")
    st.caption("⚠️ **Disclaimer:** This analysis is for educational purposes only. Always do your own research and consider consulting with a financial advisor before making investment decisions.")

if __name__ == "__main__":
    main()
