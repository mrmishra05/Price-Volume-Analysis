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
        
        # Store original column names for debugging
        original_columns = df.columns.tolist()
        st.sidebar.write("**Original Columns Found:**")
        st.sidebar.write(original_columns)
        
        # Clean column names (lowercase, trim, replace spaces with underscores)
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Store cleaned column names for debugging
        cleaned_columns = df.columns.tolist()
        st.sidebar.write("**Cleaned Columns:**")
        st.sidebar.write(cleaned_columns)
        
        # Try to find date column with various possible names (including 'date1' from sample)
        date_candidates = ['date', 'datetime', 'timestamp', 'day', 'trading_date', 'trade_date', 'date1']
        date_column = None
        
        for candidate in date_candidates:
            if candidate in df.columns:
                date_column = candidate
                break
        
        if date_column:
            st.sidebar.success(f"✅ Found date column: '{date_column}'")
            df['date'] = pd.to_datetime(df[date_column], errors='coerce')
            # Remove rows with invalid dates
            df = df.dropna(subset=['date'])
        else:
            st.sidebar.error("❌ No date column found!")
            st.sidebar.write("Please ensure your sheet has a column with dates named one of:")
            st.sidebar.write(date_candidates)
            return None
        
        # Convert numeric columns - mapping from raw names to standardized names
        numeric_candidates = {
            'prev_close': ['prev_close', 'previous_close', 'prev_close_price'],
            'close_price': ['close_price', 'close', 'closing_price', 'last_price', 'ltp'], # Added 'last_price'
            'volume': ['volume', 'ttl_trd_qnty', 'traded_quantity', 'qty'], # Added 'ttl_trd_qnty'
            'deliv_qty': ['deliv_qty', 'delivery_quantity', 'delivery_qty', 'del_qty']
        }
        
        for standard_name, candidates in numeric_candidates.items():
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
                # If a key numeric column is not found, set it to NaN column to avoid errors
                df[standard_name] = np.nan 
                st.sidebar.warning(f"⚠️ Column '{standard_name}' (or its candidates) not found. Setting to NaN.")

        # Try to find symbol column
        symbol_candidates = ['symbol', 'stock_symbol', 'ticker', 'scrip', 'instrument']
        symbol_column = None
        
        for candidate in symbol_candidates:
            if candidate in df.columns:
                symbol_column = candidate
                break
        
        if symbol_column and symbol_column != 'symbol':
            df = df.rename(columns={symbol_column: 'symbol'})
        elif 'symbol' not in df.columns:
            st.sidebar.error("❌ No symbol column found!")
            st.sidebar.write("Please ensure your sheet has a column with symbols named one of:")
            st.sidebar.write(symbol_candidates)
            return None

        # Final validation for essential columns
        required_columns = ['symbol', 'date', 'close_price', 'volume', 'deliv_qty'] # prev_close is optional for daily calculation if close_price is available
        missing_columns = [col for col in required_columns if col not in df.columns or df[col].isnull().all()]
        
        if missing_columns:
            st.error(f"❌ Missing or entirely empty required columns: {missing_columns}. Please check your raw data.")
            st.write("**Available columns:**", df.columns.tolist())
            return None
        
        # Ensure unique symbol-date pairs and sort data for accurate diff calculations
        if 'symbol' in df.columns and 'date' in df.columns:
            df = df.sort_values(by=['symbol', 'date']).drop_duplicates(subset=['symbol', 'date'], keep='last')
            st.sidebar.success(f"✅ Data processed to ensure unique (Symbol, Date) pairs and sorted.")
        
        # --- Calculate Daily Deltas (Price, Volume, Delivery Quantity) ---
        # These are calculated for every row relative to the previous trading day for that symbol
        
        # Price Change
        df['daily_price_change_abs'] = df.groupby('symbol')['close_price'].diff()
        df['daily_price_change_pct'] = (df['daily_price_change_abs'] / df.groupby('symbol')['close_price'].shift(1)) * 100
        
        # Volume Change
        df['daily_volume_change_abs'] = df.groupby('symbol')['volume'].diff()
        df['daily_volume_change_pct'] = (df['daily_volume_change_abs'] / df.groupby('symbol')['volume'].shift(1)) * 100

        # Delivery Quantity Change
        df['daily_deliv_change_abs'] = df.groupby('symbol')['deliv_qty'].diff()
        df['daily_deliv_change_pct'] = (df['daily_deliv_change_abs'] / df.groupby('symbol')['deliv_qty'].shift(1)) * 100

        # Fill NaN values for percentage changes where denominator was zero or first entry
        df['daily_price_change_pct'] = df['daily_price_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['daily_volume_change_pct'] = df['daily_volume_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)
        df['daily_deliv_change_pct'] = df['daily_deliv_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)

        st.sidebar.success(f"✅ Daily deltas calculated. Total rows: {len(df)}")
        return df
    
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        st.write("**Debug Info:**")
        st.write(f"CSV URL: {csv_url if 'csv_url' in locals() else 'Not generated'}")
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
        current_day_data = df[df['date'] == selected_date].drop_duplicates(subset=['symbol'])
        
        # Rename the daily delta columns to the generic names expected by categorize_stocks
        if not current_day_data.empty:
            current_day_data = current_day_data.rename(columns={
                'daily_price_change_abs': 'price_change',
                'daily_price_change_pct': 'price_change_pct',
                'daily_volume_change_abs': 'volume_change', # Using volume for deliv_change here as per previous logic
                'daily_deliv_change_pct': 'deliv_change_pct' # Use actual daily delivery change
            })
            # Ensure 'deliv_change' is also present if needed, though not directly used by categorize_stocks
            current_day_data['deliv_change'] = current_day_data['daily_deliv_change_abs']
            return current_day_data[['symbol', 'date', 'close_price', 'volume', 'deliv_qty', 'price_change', 'price_change_pct', 'deliv_change', 'deliv_change_pct']]
        else:
            return pd.DataFrame() # Return empty if no data for selected date
    
    else: # For Weekly, Monthly, Custom Range, calculate dynamically
        if timeframe == 'Weekly':
            compare_date = selected_date - timedelta(days=7)
        elif timeframe == 'Monthly':
            compare_date = selected_date - timedelta(days=30)
        elif timeframe == 'Custom Range':
            compare_date = start_date # Use start_date for comparison base

        # Get data for the selected end date
        current_period_data = df[df['date'] == selected_date].drop_duplicates(subset=['symbol'])
        if timeframe == 'Custom Range':
            current_period_data = df[df['date'] == end_date].drop_duplicates(subset=['symbol'])

        # Find the most recent data on or before the comparison date for each symbol
        past_period_data_candidates = df[df['date'] <= compare_date].sort_values('date').groupby('symbol').last().reset_index()
        past_period_data = past_period_data_candidates.drop_duplicates(subset=['symbol'])

        if not current_period_data.empty and not past_period_data.empty:
            merged = current_period_data.merge(past_period_data, on='symbol', suffixes=('', '_prev_period'))
            merged = merged.drop_duplicates(subset=['symbol']) # Final check for merge-induced duplicates

            # Calculate Price Change for the period
            merged['price_change'] = merged['close_price'] - merged['close_price_prev_period']
            merged['price_change_pct'] = (merged['price_change'] / merged['close_price_prev_period']) * 100
            merged['price_change_pct'] = merged['price_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0) # Handle division by zero

            # Calculate Volume Change for the period
            if 'volume' in merged.columns and 'volume_prev_period' in merged.columns:
                merged['volume_change'] = merged['volume'] - merged['volume_prev_period']
                merged['volume_change_pct'] = (merged['volume_change'] / merged['volume_prev_period']) * 100
                merged['volume_change_pct'] = merged['volume_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                merged['volume_change'] = np.nan
                merged['volume_change_pct'] = np.nan
            
            # Calculate Delivery Quantity Change for the period
            if 'deliv_qty' in merged.columns and 'deliv_qty_prev_period' in merged.columns:
                merged['deliv_change'] = merged['deliv_qty'] - merged['deliv_qty_prev_period']
                merged['deliv_change_pct'] = (merged['deliv_change'] / merged['deliv_qty_prev_period']) * 100
                merged['deliv_change_pct'] = merged['deliv_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)
            else:
                merged['deliv_change'] = np.nan
                merged['deliv_change_pct'] = np.nan

            return merged[['symbol', 'date', 'close_price', 'volume', 'deliv_qty', 'price_change', 'price_change_pct', 'deliv_change', 'deliv_change_pct']]
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
        st.info("📝 **Instructions:**\n1. Make your raw data Google Sheet (e.g., 'NSE_2025') publicly accessible.\n2. Copy the full URL from your browser.\n3. Paste it in the sidebar.\n\n**Expected Raw Column Names (any of these variants will work):**\n- **Date**: Date, Date1, DateTime, Timestamp, Day, Trading_Date\n- **Symbol**: Symbol, Stock_Symbol, Ticker, Scrip, Instrument\n- **Close Price**: Close_Price, Close, Closing_Price, LAST_PRICE, LTP\n- **Previous Close**: PREV_CLOSE, Previous_Close, Prev_Close_Price\n- **Volume**: Volume, TTL_TRD_QNTY, Traded_Quantity, Qty\n- **Delivery**: DELIV_QTY, Delivery_Quantity, Del_Qty")
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
    processed_df = calculate_price_changes(df, timeframe, selected_date, start_date=start_date_custom, end_date=end_date_custom)
    
    if processed_df is None or processed_df.empty:
        st.error("No data available for the selected parameters after calculating changes.")
        st.info("Try selecting a different date or timeframe. Ensure data exists for both start and end points of the comparison.")
        return
    
    # Categorize stocks based on the calculated changes for the selected timeframe
    categorized_df = categorize_stocks(processed_df)
    
    # Filtering options
    st.sidebar.subheader("Filters")
    
    # Search filter
    search_term = st.sidebar.text_input("Search Stocks", "")
    if search_term:
        categorized_df = categorized_df[categorized_df['symbol'].str.contains(search_term, case=False, na=False)]
    
    # Category filter
    categories = ['All'] + sorted(list(categorized_df['category'].unique()))
    selected_category = st.sidebar.selectbox("Filter by Category", categories)
    if selected_category != 'All':
        categorized_df = categorized_df[categorized_df['category'] == selected_category]
    
    # Significant moves only
    significant_only = st.sidebar.checkbox("Show Significant Moves Only (>2% price change)")
    if significant_only:
        # Check if 'price_change_pct' exists before filtering
        if 'price_change_pct' in categorized_df.columns:
            categorized_df = categorized_df[abs(categorized_df['price_change_pct'].fillna(0)) > 2]
        else:
            st.warning("Cannot filter by significant moves: 'price_change_pct' column not found.")

    # Main dashboard
    if not categorized_df.empty:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            strong_buy_count = len(categorized_df[categorized_df['signal_strength'] == 'STRONG BUY'])
            st.metric("Price Inc, Del Vol Inc", strong_buy_count)
        
        with col2:
            accumulation_count = len(categorized_df[categorized_df['signal_strength'] == 'ACCUMULATION'])
            st.metric("Price Not Inc, Del Vol Inc", accumulation_count)
        
        with col3:
            # Ensure price_change_pct exists before calculating mean
            if 'price_change_pct' in categorized_df.columns:
                avg_price_change = categorized_df['price_change_pct'].mean()
                st.metric("Avg Price Change", f"{avg_price_change:.2f}%" if pd.notna(avg_price_change) else "N/A")
            else:
                st.metric("Avg Price Change", "N/A")
        
        with col4:
            total_stocks = len(categorized_df)
            st.metric("Total Stocks", total_stocks)
        
        # Charts
        col1, col2 = st.columns(2) # Keep two columns for potential future charts if needed
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig_pie = create_category_pie_chart(categorized_df)
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                create_category_pie_chart(categorized_df)
        
        with col2:
            st.info("Additional charts can be added here if needed.") # Placeholder
            
        # Top movers
        st.subheader("Top Movers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Gainers**")
            # Ensure price_change_pct exists before sorting
            if 'price_change_pct' in categorized_df.columns:
                top_gainers = categorized_df.nlargest(5, 'price_change_pct')[['symbol', 'close_price', 'price_change_pct', 'signal_strength']]
                st.dataframe(top_gainers, use_container_width=True, hide_index=True)
            else:
                st.info("Price change data not available for Top Gainers.")
        
        with col2:
            st.write("**Top Losers**")
            # Ensure price_change_pct exists before sorting
            if 'price_change_pct' in categorized_df.columns:
                top_losers = categorized_df.nsmallest(5, 'price_change_pct')[['symbol', 'close_price', 'price_change_pct', 'signal_strength']]
                st.dataframe(top_losers, use_container_width=True, hide_index=True)
            else:
                st.info("Price change data not available for Top Losers.")
        
        # Detailed table
        st.subheader(f"Detailed Analysis ({len(categorized_df)} stocks)")
        
        # Sort options
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            # Include 'price_change_pct' and 'deliv_change_pct' for sorting
            available_sort_cols = [col for col in ['price_change_pct', 'deliv_change_pct', 'close_price', 'volume', 'deliv_qty'] if col in categorized_df.columns]
            if available_sort_cols:
                sort_by = st.selectbox("Sort by", available_sort_cols)
            else:
                sort_by = 'symbol' # Fallback if no numeric columns
        with sort_col2:
            sort_order = st.selectbox("Sort order", ['Descending', 'Ascending'])
        
        # Sort dataframe
        ascending = sort_order == 'Ascending'
        # Handle potential NaNs in sort column by filling with 0 or a large number for sorting
        display_df = categorized_df.sort_values(sort_by, ascending=ascending, na_position='last')
        
        # Display table with custom formatting
        display_columns = ['symbol', 'close_price', 'price_change_pct', 'deliv_change_pct', 'volume', 'deliv_qty', 'category', 'signal_strength']
        available_display_cols = [col for col in display_columns if col in display_df.columns]
        
        if available_display_cols:
            display_df_formatted = display_df[available_display_cols].copy()
            
            # Format columns if they exist
            if 'close_price' in display_df_formatted.columns:
                display_df_formatted['close_price'] = display_df_formatted['close_price'].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "N/A")
            if 'price_change_pct' in display_df_formatted.columns:
                display_df_formatted['price_change_pct'] = display_df_formatted['price_change_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            if 'deliv_change_pct' in display_df_formatted.columns:
                display_df_formatted['deliv_change_pct'] = display_df_formatted['deliv_change_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            if 'volume' in display_df_formatted.columns:
                display_df_formatted['volume'] = display_df_formatted['volume'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A") # Format as integer
            if 'deliv_qty' in display_df_formatted.columns:
                display_df_formatted['deliv_qty'] = display_df_formatted['deliv_qty'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A") # Format as integer
            
            # Apply CSS styling to the 'signal_strength' column
            def color_signal(val):
                if val == 'STRONG BUY': return f'<span class="signal-price-increasing-delivery-volume-increasing">{val}</span>'
                elif val == 'ACCUMULATION': return f'<span class="signal-price-not-increasing-delivery-volume-increasing">{val}</span>'
                elif val == 'BOTTOM FISHING': return f'<span class="signal-price-decreasing-volume-increasing">{val}</span>'
                elif val == 'WEAK SELL': return f'<span class="signal-price-decreasing-volume-decreasing">{val}</span>'
                elif val == 'WEAK BUY': return f'<span class="signal-price-increasing-volume-not-increasing">{val}</span>'
                elif val == 'NEUTRAL / HOLD': return f'<span class="signal-neutral-hold">{val}</span>'
                return val

            if 'signal_strength' in display_df_formatted.columns:
                display_df_html = display_df_formatted.to_html(escape=False, index=False)
                # Replace plain text signal strength with styled HTML
                for original_signal in categorized_df['signal_strength'].unique():
                    if original_signal: # Avoid None/NaN
                        styled_signal_html = color_signal(original_signal)
                        # Replace only the specific cell content, not general text
                        display_df_html = display_df_html.replace(f'<td>{original_signal}</td>', f'<td>{styled_signal_html}</td>')
                st.markdown(display_df_html, unsafe_allow_html=True)
            else:
                st.dataframe(display_df_formatted, use_container_width=True, hide_index=True)
        else:
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Export functionality
        st.subheader("Export Data")
        if st.button("Generate CSV Download"):
            csv = categorized_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"nse_analysis_{timeframe.lower().replace(' ', '_')}_{selected_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("No data available after applying filters")

if __name__ == "__main__":
    main()
