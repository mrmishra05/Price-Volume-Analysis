import streamlit as st
import pandas as pd
import requests
import zipfile
import io
from datetime import datetime, timedelta

# For visualizations
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuration ---
# URL for daily Bhavcopy ZIP files from NSE archives
# Note: NSE URL structures can change. If you encounter 404s, verify the latest URL on NSE's site.
NSE_BHAVCOPY_URL = "https://archives.nseindia.com/content/historical/EQUITIES/{YEAR}/{MON}/cm{DD}{MON}{YEAR}bhav.csv.zip"

# Mapping month numbers to NSE's three-letter abbreviations (e.g., 1 -> JAN)
MONTH_MAP = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
}

# Partial list of NSE Trading Holidays for 2025 (you should maintain this for accuracy)
# Source: Refer to NSE India's official holiday calendar for the most up-to-date list.
NSE_HOLIDAYS_2025 = [
    datetime(2025, 1, 26).date(),  # Republic Day (Sunday)
    datetime(2025, 2, 26).date(),  # Mahashivratri
    datetime(2025, 3, 14).date(),  # Holi
    datetime(2025, 3, 31).date(),  # Eid-Ul-Fitr
    datetime(2025, 4, 10).date(),  # Shri Mahavir Jayanti
    datetime(2025, 4, 14).date(),  # Dr. Baba Saheb Ambedkar Jayanti
    datetime(2025, 4, 18).date(),  # Good Friday
    datetime(2025, 5, 1).date(),   # Maharashtra Day
    datetime(2025, 6, 7).date(),   # Bakra Eid (Saturday)
    datetime(2025, 7, 6).date(),   # Muharram (Sunday)
    datetime(2025, 8, 15).date(),  # Independence Day
    datetime(2025, 8, 27).date(),  # Ganesh Chaturthi
    datetime(2025, 10, 2).date(), # Mahatma Gandhi Jayanti/Dussehra
    datetime(2025, 10, 21).date(), # Diwali Laxmi Pujan (Muhurat Trading only)
    datetime(2025, 10, 22).date(), # Diwali-Balipratipada
    datetime(2025, 11, 5).date(),  # Prakash Gurpurb Sri Guru Nanak Dev
    datetime(2025, 12, 25).date()  # Christmas
]

# --- Data Fetching and Processing ---

@st.cache_data(ttl=3600) # Cache each daily download for 1 hour
def fetch_single_day_bhavcopy(date: datetime.date):
    """
    Fetches and processes Bhavcopy for a single specific date from NSE archives.
    Returns a pandas DataFrame for that day, or None if not found/error.
    """
    year = date.year
    month_abbr = MONTH_MAP[date.month]
    day = date.day
    
    url = NSE_BHAVCOPY_URL.format(
        YEAR=year,
        MON=month_abbr,
        DD=f"{day:02d}"  # Ensures day is two digits (e.g., 01, 05)
    )
    
    # Use a common User-Agent to mimic a browser, as NSE might block simple requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36)'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10) # Added timeout for network requests
        response.raise_for_status() # Raises HTTPError for 4xx/5xx responses

        # Check if the content is actually a zip file. NSE might return a small HTML error page.
        if not zipfile.is_zipfile(io.BytesIO(response.content)):
            return None # Not a valid zip, likely no data for the day

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            # Find the CSV file within the zip (usually only one)
            csv_file_name = [name for name in z.namelist() if name.endswith('.csv')][0]
            with z.open(csv_file_name) as f:
                df = pd.read_csv(f)
                
                # --- Standardize and Clean Column Names ---
                df.columns = [col.strip().upper() for col in df.columns]

                # Map inconsistent NSE column names to standard internal names
                # This ensures consistent naming regardless of subtle NSE changes in headers
                column_rename_map = {
                    "TIMESTAMP": "DATE", # Standardize any date/timestamp column to 'DATE'
                    "DATE1": "DATE",     # Common alternative for date column
                    "PREV_CLOSE": "PREVCLOSE",
                    "OPEN_PRICE": "OPEN",
                    "HIGH_PRICE": "HIGH",
                    "LOW_PRICE": "LOW",
                    "LAST_PRICE": "LAST",
                    "CLOSE_PRICE": "CLOSE",
                    "TOTTRDVOL": "TOTTRDQTY", # Total Traded Volume -> Total Traded Quantity
                    "TOTTRDVAL": "TOTTRDVAL", # Total Traded Value (already good)
                    "NO_OF_TRADES": "TOTALTRADES",
                    "AVG_PRICE": "AVG_PRICE",
                    "DELIV_QTY": "DELIV_QTY",
                    "DELIV_PER": "DELIV_PER"
                }
                df.rename(columns=column_rename_map, inplace=True)

                # Filter for 'EQ' (Equity) series only if 'SERIES' column exists
                if 'SERIES' in df.columns:
                    df = df[df['SERIES'] == 'EQ']
                
                # Convert essential columns to numeric, coercing errors to NaN and filling with 0
                numeric_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES', 'AVG_PRICE', 'DELIV_QTY', 'DELIV_PER']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 
                    
                # Add a consistent 'TIMESTAMP' column for all data points (important for time series)
                df['TIMESTAMP'] = pd.to_datetime(date)
                
                # Drop rows with NaN in critical columns needed for analysis
                df.dropna(subset=['SYMBOL', 'TIMESTAMP', 'CLOSE', 'TOTTRDQTY'], inplace=True)
                
                return df
    except requests.exceptions.RequestException as e:
        # Log error for debugging, but return None to allow the loop to continue for other dates
        print(f"Network error or 404 for {date.strftime('%d-%b-%Y')}. Skipping: {e}")
        return None
    except Exception as e:
        print(f"Error processing bhavcopy for {date.strftime('%d-%b-%Y')}: {e}. Skipping.")
        return None

@st.cache_data(ttl=3600) # Cache the full historical data fetch for 1 hour
def load_historical_bhavcopy(start_date: datetime.date, end_date: datetime.date):
    """
    Loads Bhavcopy data for a given date range by iterating through days.
    Calculates daily price and volume changes.
    """
    all_dfs = []
    current_date = start_date
    
    # Progress bar for better user experience during longer data fetches
    total_days = (end_date - start_date).days + 1
    progress_bar_text = st.empty()
    progress_bar = st.progress(0)
    processed_count = 0

    while current_date <= end_date:
        processed_count += 1
        progress_value = processed_count / total_days
        progress_bar.progress(progress_value)
        progress_bar_text.text(f"Fetching data for {current_date.strftime('%d-%b-%Y')}... ({int(progress_value*100)}%)")

        # Skip weekends and known holidays
        if current_date.weekday() >= 5: # 5 is Saturday, 6 is Sunday
            st.info(f"Skipping {current_date.strftime('%d-%b-%Y')} (Weekend).")
            current_date += timedelta(days=1)
            continue
        if current_date in NSE_HOLIDAYS_2025:
            st.info(f"Skipping {current_date.strftime('%d-%b-%Y')} (NSE Holiday).")
            current_date += timedelta(days=1)
            continue

        df = fetch_single_day_bhavcopy(current_date)
        if df is not None and not df.empty:
            all_dfs.append(df)
        else:
            # This 'else' catches 404s or other fetch errors for valid trading days not in holiday list
            st.warning(f"Could not fetch data for {current_date.strftime('%d-%b-%Y')}. It might be a non-trading day not in the holiday list, or a temporary NSE issue.")
        
        current_date += timedelta(days=1)
    
    progress_bar.empty() # Remove progress bar after completion
    progress_bar_text.empty()

    if not all_dfs:
        st.error("No data found for the selected date range. Please try a different range or check the NSE holiday calendar.")
        return pd.DataFrame() # Return empty DataFrame if no data was loaded

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.sort_values(by=['SYMBOL', 'TIMESTAMP'], inplace=True)

    # --- Calculate Daily Price and Volume Change (Shifted by 1 trading day per symbol) ---
    # Group by symbol to calculate changes relative to the previous trading day for that specific stock
    full_df['PREV_CLOSE_LAG'] = full_df.groupby('SYMBOL')['CLOSE'].shift(1)
    full_df['DAILY_PRICE_CHANGE'] = full_df['CLOSE'] - full_df['PREV_CLOSE_LAG']
    full_df['DAILY_PRICE_CHANGE_PCT'] = (full_df['DAILY_PRICE_CHANGE'] / full_df['PREV_CLOSE_LAG']) * 100
    # Handle infinite values (division by zero if PREV_CLOSE_LAG was 0) and NaNs (first entry for a symbol)
    full_df['DAILY_PRICE_CHANGE_PCT'].replace([float('inf'), -float('inf')], 0, inplace=True)
    full_df['DAILY_PRICE_CHANGE_PCT'].fillna(0, inplace=True) 

    full_df['PREV_VOLUME_LAG'] = full_df.groupby('SYMBOL')['TOTTRDQTY'].shift(1)
    full_df['DAILY_VOLUME_CHANGE'] = full_df['TOTTRDQTY'] - full_df['PREV_VOLUME_LAG']
    full_df['DAILY_VOLUME_CHANGE_PCT'] = (full_df['DAILY_VOLUME_CHANGE'] / full_df['PREV_VOLUME_LAG']) * 100
    full_df['DAILY_VOLUME_CHANGE_PCT'].replace([float('inf'), -float('inf')], 0, inplace=True)
    full_df['DAILY_VOLUME_CHANGE_PCT'].fillna(0, inplace=True) 

    return full_df

# --- Trend Calculation Functions ---

def get_trend_data(df: pd.DataFrame, period: str):
    """
    Aggregates data for price and volume trends based on the specified period (Daily, Weekly, Monthly).
    """
    if df.empty:
        return pd.DataFrame()

    # Ensure TIMESTAMP is datetime and set as index for resampling
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
    df_indexed = df.set_index('TIMESTAMP')

    df_agg = pd.DataFrame() # Initialize empty DataFrame

    if period == "Daily":
        # For daily, we just return the relevant daily change columns
        # Filter out rows where daily changes cannot be calculated (first day of stock in range)
        df_agg = df_indexed.copy()
        df_agg.dropna(subset=['DAILY_PRICE_CHANGE_PCT', 'DAILY_VOLUME_CHANGE_PCT'], inplace=True)
        # Select and reorder columns for daily display
        return df_agg[['SYMBOL', 'CLOSE', 'TOTTRDQTY', 'DAILY_PRICE_CHANGE_PCT', 'DAILY_VOLUME_CHANGE_PCT']]
    
    elif period == "Weekly":
        # Group by symbol and then resample weekly ('W' for week-ending Sunday)
        df_agg = df_indexed.groupby('SYMBOL').resample('W').agg(
            Open=('OPEN', 'first'),        # Open of the first trading day of the week
            Close=('CLOSE', 'last'),       # Close of the last trading day of the week
            High=('HIGH', 'max'),
            Low=('LOW', 'min'),
            TotalVolume=('TOTTRDQTY', 'sum'), # Sum of volume for the week
            NumTradingDays=('SYMBOL', 'count') # Count actual trading days in the week
        ).reset_index()
        df_agg.rename(columns={'TIMESTAMP': 'WEEK_END_DATE'}, inplace=True)

    elif period == "Monthly":
        # Group by symbol and then resample monthly ('M' for month-ending last day)
        df_agg = df_indexed.groupby('SYMBOL').resample('M').agg(
            Open=('OPEN', 'first'),
            Close=('CLOSE', 'last'),
            High=('HIGH', 'max'),
            Low=('LOW', 'min'),
            TotalVolume=('TOTTRDQTY', 'sum'),
            NumTradingDays=('SYMBOL', 'count')
        ).reset_index()
        df_agg.rename(columns={'TIMESTAMP': 'MONTH_END_DATE'}, inplace=True)
    
    # Calculate period-over-period price and volume change for Weekly/Monthly
    if not df_agg.empty:
        # Sort to ensure correct shift for period-over-period change
        period_date_col = 'WEEK_END_DATE' if period == "Weekly" else 'MONTH_END_DATE'
        df_agg.sort_values(by=['SYMBOL', period_date_col], inplace=True)
        
        # Price change: (Current Period Close - Previous Period Close) / Previous Period Close
        df_agg['PREV_PERIOD_CLOSE'] = df_agg.groupby('SYMBOL')['Close'].shift(1)
        df_agg['PERIOD_PRICE_CHANGE_PCT'] = ( (df_agg['Close'] - df_agg['PREV_PERIOD_CLOSE']) / df_agg['PREV_PERIOD_CLOSE'] ) * 100
        df_agg['PERIOD_PRICE_CHANGE_PCT'].replace([float('inf'), -float('inf')], 0, inplace=True)
        df_agg['PERIOD_PRICE_CHANGE_PCT'].fillna(0, inplace=True)

        # Volume change: (Current Period Total Volume - Previous Period Total Volume) / Previous Period Total Volume
        df_agg['PREV_PERIOD_VOLUME'] = df_agg.groupby('SYMBOL')['TotalVolume'].shift(1)
        df_agg['PERIOD_VOLUME_CHANGE_PCT'] = ( (df_agg['TotalVolume'] - df_agg['PREV_PERIOD_VOLUME']) / df_agg['PREV_PERIOD_VOLUME'] ) * 100
        df_agg['PERIOD_VOLUME_CHANGE_PCT'].replace([float('inf'), -float('inf')], 0, inplace=True)
        df_agg['PERIOD_VOLUME_CHANGE_PCT'].fillna(0, inplace=True)
        
        # Filter out NaN rows that arise from the first period for each stock (no previous period to compare)
        df_agg.dropna(subset=['PERIOD_PRICE_CHANGE_PCT', 'PERIOD_VOLUME_CHANGE_PCT'], inplace=True)
        
        # Select and reorder columns for display
        return df_agg[['SYMBOL', period_date_col, 'Open', 'Close', 'TotalVolume', 
                       'PERIOD_PRICE_CHANGE_PCT', 'PERIOD_VOLUME_CHANGE_PCT', 'NumTradingDays']]
    
    return df_agg # Return empty if df_agg is empty

# --- Streamlit App Layout and Logic ---

def main():
    st.set_page_config(layout="wide", page_title="NSE Historical Data Analyzer")

    st.title("📊 NSE Historical Data Analyzer")
    st.write("Analyze daily, weekly, and monthly trends for NSE equities directly from NSE archives.")

    st.sidebar.header("Data Fetching Options")
    today = datetime.now().date()
    # Default to a range ending yesterday, starting 30 days prior
    default_end_date = today - timedelta(days=1)
    default_start_date = default_end_date - timedelta(days=30) 

    # Date range picker in the sidebar
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(default_start_date, default_end_date),
        max_value=today, # Cannot select future dates
        key="date_range_selector"
    )

    # Ensure start_date and end_date are correctly extracted from the tuple
    start_date = date_range[0]
    end_date = date_range[1] if len(date_range) == 2 else start_date

    st.sidebar.markdown("---")
    st.sidebar.header("Analysis Options")
    # Radio buttons for selecting the trend period
    trend_period = st.sidebar.radio(
        "Select Trend Period",
        ("Daily", "Weekly", "Monthly"),
        index=0 # Default to Daily
    )

    if st.sidebar.button("Fetch & Analyze Data"):
        st.subheader(f"Analyzing Data from {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")
        
        # Load historical data for the selected range
        full_bhavcopy_df = load_historical_bhavcopy(start_date, end_date)

        if full_bhavcopy_df.empty:
            st.error("No data could be loaded for the selected range. This might be due to all days being non-trading days, network issues, or changes in NSE's data provision.")
            return # Stop execution if no data

        st.success(f"Successfully loaded data for {full_bhavcopy_df['SYMBOL'].nunique()} unique symbols across {full_bhavcopy_df['TIMESTAMP'].nunique()} trading days.")

        # --- Individual Stock Analysis ---
        st.header("Individual Stock Analysis")
        all_symbols = sorted(full_bhavcopy_df['SYMBOL'].unique().tolist())
        selected_symbol = st.selectbox(
            "Select a Stock Symbol",
            [''] + all_symbols, # Add an empty option at the top
            index=0
        )

        if selected_symbol:
            symbol_df = full_bhavcopy_df[full_bhavcopy_df['SYMBOL'] == selected_symbol].sort_values('TIMESTAMP').reset_index(drop=True)
            
            if not symbol_df.empty:
                # --- Total Change over Selected Range for Individual Stock ---
                st.subheader(f"Summary for {selected_symbol} over Selected Range")
                
                first_close = symbol_df['CLOSE'].iloc[0]
                last_close = symbol_df['CLOSE'].iloc[-1]
                total_price_change_abs = last_close - first_close
                total_price_change_pct = (total_price_change_abs / first_close) * 100 if first_close != 0 else 0

                first_volume = symbol_df['TOTTRDQTY'].iloc[0]
                last_volume = symbol_df['TOTTRDQTY'].iloc[-1]
                total_volume_change_abs = last_volume - first_volume
                total_volume_change_pct = (total_volume_change_abs / first_volume) * 100 if first_volume != 0 else 0

                col_summary1, col_summary2 = st.columns(2)
                with col_summary1:
                    st.metric("Total Price Change (Abs)", f"₹{total_price_change_abs:.2f}")
                    st.metric("Total Price Change (%)", f"{total_price_change_pct:.2f}%")
                with col_summary2:
                    st.metric("Total Volume Change (Abs)", f"{total_volume_change_abs:,.0f}")
                    st.metric("Total Volume Change (%)", f"{total_volume_change_pct:.2f}%")

                st.markdown("---")
                st.subheader(f"Daily Data for {selected_symbol} ({start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')})")
                # Display relevant daily columns in a dataframe
                st.dataframe(symbol_df[['TIMESTAMP', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'TOTTRDQTY', 'TOTALTRADES', 'DAILY_PRICE_CHANGE_PCT', 'DAILY_VOLUME_CHANGE_PCT']], use_container_width=True)

                st.markdown("---")
                st.subheader(f"Visualizing Trends for {selected_symbol}")

                # --- Plotly Price Trend Chart ---
                fig_price = px.line(symbol_df, x='TIMESTAMP', y='CLOSE', 
                                    title=f'{selected_symbol} Closing Price Trend',
                                    labels={'CLOSE': 'Closing Price (₹)', 'TIMESTAMP': 'Date'},
                                    hover_data={'OPEN': ':.2f', 'HIGH': ':.2f', 'LOW': ':.2f', 'CLOSE': ':.2f', 'TIMESTAMP': '|%Y-%m-%d'})
                fig_price.update_xaxes(rangeslider_visible=True, rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ))
                fig_price.update_layout(hovermode="x unified", title_x=0.5)
                st.plotly_chart(fig_price, use_container_width=True)

                # --- Plotly Volume Trend Chart ---
                fig_volume = px.bar(symbol_df, x='TIMESTAMP', y='TOTTRDQTY', 
                                    title=f'{selected_symbol} Volume Traded',
                                    labels={'TOTTRDQTY': 'Volume', 'TIMESTAMP': 'Date'},
                                    color_discrete_sequence=['#1f77b4'], # Consistent blue color for bars
                                    hover_data={'TOTTRDQTY': ':,0f', 'TIMESTAMP': '|%Y-%m-%d'})
                fig_volume.update_xaxes(rangeslider_visible=True, rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all")
                    ])
                ))
                fig_volume.update_layout(hovermode="x unified", title_x=0.5)
                st.plotly_chart(fig_volume, use_container_width=True)

                st.markdown("---")
                # Display trend analysis for the selected stock based on the chosen period
                st.subheader(f"Trend Analysis for {selected_symbol} ({trend_period} Comparison)")
                # Pass a copy to avoid unintended modifications by get_trend_data
                trend_df = get_trend_data(symbol_df.copy(), trend_period) 
                
                if not trend_df.empty:
                    st.dataframe(trend_df, use_container_width=True)
                else:
                    st.info(f"No {trend_period.lower()} trend data available for {selected_symbol} in the selected range (might be too short for the period).")
            else:
                st.warning(f"No daily data found for {selected_symbol} in the selected range.")

        # --- Overall Trend Analysis ---
        if not full_bhavcopy_df.empty:
            st.header(f"Overall {trend_period} Trends (Top 20 by Total Volume in Period)")
            
            # Get trend data for all stocks across the selected period
            overall_trend_df = get_trend_data(full_bhavcopy_df.copy(), trend_period)

            if not overall_trend_df.empty:
                # Identify top 20 symbols by their *total volume over the entire period*
                # This ensures we pick consistently high-volume stocks for overall trends
                top_volume_symbols = overall_trend_df.groupby('SYMBOL')['TotalVolume'].sum().nlargest(20).index.tolist()
                
                # Filter the overall trend data to show only these top symbols
                overall_trend_df_filtered = overall_trend_df[overall_trend_df['SYMBOL'].isin(top_volume_symbols)].copy()
                
                # Sort for display (most recent period first, then by volume)
                if trend_period == "Weekly":
                    overall_trend_df_filtered.sort_values(by=['WEEK_END_DATE', 'TotalVolume'], ascending=[False, False], inplace=True)
                elif trend_period == "Monthly":
                    overall_trend_df_filtered.sort_values(by=['MONTH_END_DATE', 'TotalVolume'], ascending=[False, False], inplace=True)
                else: # Daily
                     overall_trend_df_filtered.sort_values(by=['TIMESTAMP', 'TOTTRDQTY'], ascending=[False, False], inplace=True)

                st.info(f"Displaying {trend_period.lower()} trends for top 20 symbols by total volume in the selected range (sorted by recent date and volume).")
                st.dataframe(overall_trend_df_filtered, use_container_width=True)
            else:
                st.info(f"No overall {trend_period.lower()} trend data available for the selected range. Try a longer date range.")

        st.markdown("---")
        st.caption("Developed by Gemini AI. Data sourced from NSE India (via archives.nseindia.com).")

if __name__ == "__main__":
    main()
