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
.signal-strong-buy {
    background-color: #10b981;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.signal-weak-buy {
    background-color: #f59e0b;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.signal-accumulation {
    background-color: #3b82f6;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.signal-bottom-fishing {
    background-color: #8b5cf6;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.signal-weak-sell {
    background-color: #ef4444;
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
    font-size: 0.75rem;
    font-weight: 600;
}
.signal-hold {
    background-color: #6b7280;
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
    """Load data from Google Sheets using the public CSV export URL"""
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
            csv_url = sheet_url
        
        # Read the CSV data with proper error handling
        df = pd.read_csv(csv_url, encoding='utf-8')
        
        # Store original column names for debugging
        original_columns = df.columns.tolist()
        st.sidebar.write("**Original Columns Found:**")
        st.sidebar.write(original_columns)
        
        # Clean column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Store cleaned column names for debugging
        cleaned_columns = df.columns.tolist()
        st.sidebar.write("**Cleaned Columns:**")
        st.sidebar.write(cleaned_columns)
        
        # Try to find date column with various possible names
        date_candidates = ['date', 'datetime', 'timestamp', 'day', 'trading_date', 'trade_date']
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
        
        # Convert numeric columns
        numeric_candidates = {
            'prev_close': ['prev_close', 'previous_close', 'prev_close_price'],
            'close_price': ['close_price', 'close', 'closing_price', 'ltp'],
            'volume': ['volume', 'traded_quantity', 'qty'],
            'deliv_qty': ['deliv_qty', 'delivery_quantity', 'delivery_qty'],
            'deliv_per': ['deliv_per', 'delivery_percentage', 'delivery_per']
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
                    # Rename the column to standard name
                    df = df.rename(columns={found_column: standard_name})
        
        # Try to find symbol column
        symbol_candidates = ['symbol', 'stock_symbol', 'ticker', 'scrip', 'instrument']
        symbol_column = None
        
        for candidate in symbol_candidates:
            if candidate in df.columns:
                symbol_column = candidate
                break
        
        if symbol_column and symbol_column != 'symbol':
            df = df.rename(columns={symbol_column: 'symbol'})
        
        # Final validation
        required_columns = ['symbol', 'date', 'close_price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.write("**Available columns:**", df.columns.tolist())
            return None
        
        st.sidebar.success(f"✅ Data loaded successfully: {len(df)} rows")
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("**Debug Info:**")
        st.write(f"CSV URL: {csv_url if 'csv_url' in locals() else 'Not generated'}")
        return None

def calculate_price_changes(df, timeframe, selected_date):
    """Calculate price changes based on timeframe"""
    df = df.copy()
    
    if timeframe == 'Today vs Yesterday':
        # Filter for selected date and previous date
        current_data = df[df['date'] == selected_date]
        prev_date = selected_date - timedelta(days=1)
        # Find the most recent data before the selected date
        prev_data = df[df['date'] < selected_date].sort_values('date').groupby('symbol').last().reset_index()
        
        if not current_data.empty and not prev_data.empty:
            merged = current_data.merge(prev_data, on='symbol', suffixes=('', '_prev'))
            merged['price_change'] = merged['close_price'] - merged['close_price_prev']
            merged['price_change_pct'] = (merged['price_change'] / merged['close_price_prev']) * 100
            return merged
    
    elif timeframe == 'Weekly':
        # Compare with 7 days ago
        current_data = df[df['date'] == selected_date]
        week_ago_date = selected_date - timedelta(days=7)
        week_ago_data = df[df['date'] <= week_ago_date].sort_values('date').groupby('symbol').last().reset_index()
        
        if not current_data.empty and not week_ago_data.empty:
            merged = current_data.merge(week_ago_data, on='symbol', suffixes=('', '_week_ago'))
            merged['price_change'] = merged['close_price'] - merged['close_price_week_ago']
            merged['price_change_pct'] = (merged['price_change'] / merged['close_price_week_ago']) * 100
            return merged
    
    elif timeframe == 'Monthly':
        # Compare with 30 days ago
        current_data = df[df['date'] == selected_date]
        month_ago_date = selected_date - timedelta(days=30)
        month_ago_data = df[df['date'] <= month_ago_date].sort_values('date').groupby('symbol').last().reset_index()
        
        if not current_data.empty and not month_ago_data.empty:
            merged = current_data.merge(month_ago_data, on='symbol', suffixes=('', '_month_ago'))
            merged['price_change'] = merged['close_price'] - merged['close_price_month_ago']
            merged['price_change_pct'] = (merged['price_change'] / merged['close_price_month_ago']) * 100
            return merged
    
    elif timeframe == 'Custom Range':
        # This will be handled separately with date range inputs
        return df
    
    return df

def categorize_stocks(df):
    """Categorize stocks based on price and delivery volume patterns"""
    df = df.copy()
    
    # Check for required columns
    required_cols = ['price_change_pct']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return df
    
    # Define thresholds
    PRICE_THRESHOLD = 0.5  # 0.5%
    DELIV_THRESHOLD = 5    # 5%
    
    # Calculate delivery volume changes (if available)
    deliv_prev_cols = [col for col in df.columns if 'deliv_qty' in col and any(suffix in col for suffix in ['prev', 'week_ago', 'month_ago', 'start'])]
    if deliv_prev_cols and 'deliv_qty' in df.columns:
        # Avoid division by zero
        prev_col = deliv_prev_cols[0]
        df['deliv_change_pct'] = df.apply(
            lambda row: ((row['deliv_qty'] - row[prev_col]) / row[prev_col]) * 100 
            if pd.notna(row[prev_col]) and row[prev_col] != 0 else 0, 
            axis=1
        )
    else:
        df['deliv_change_pct'] = 0
    
    # Categorize based on price and delivery movements
    def get_category(row):
        # Handle NaN values
        price_change = row.get('price_change_pct', 0)
        deliv_change = row.get('deliv_change_pct', 0)
        
        if pd.isna(price_change):
            price_change = 0
        if pd.isna(deliv_change):
            deliv_change = 0
        
        price_up = price_change > PRICE_THRESHOLD
        price_down = price_change < -PRICE_THRESHOLD
        deliv_up = deliv_change > DELIV_THRESHOLD
        deliv_down = deliv_change < -DELIV_THRESHOLD
        
        if price_up and deliv_up:
            return 'Price Up + Delivery Up', 'STRONG BUY', '#10B981'
        elif not price_up and deliv_up:
            return 'Price Flat/Down + Delivery Up', 'ACCUMULATION', '#3B82F6'
        elif price_down and deliv_up:
            return 'Price Down + Delivery Up', 'BOTTOM FISHING', '#8B5CF6'
        elif price_down and deliv_down:
            return 'Price Down + Delivery Down', 'WEAK SELL', '#EF4444'
        elif price_up and not deliv_up:
            return 'Price Up + Delivery Flat/Down', 'WEAK BUY', '#F59E0B'
        else:
            return 'Neutral', 'HOLD', '#6B7280'
    
    categories = df.apply(get_category, axis=1)
    df['category'] = [cat[0] for cat in categories]
    df['signal_strength'] = [cat[1] for cat in categories]
    df['category_color'] = [cat[2] for cat in categories]
    
    return df

def create_category_pie_chart(df):
    """Create pie chart for category distribution"""
    category_counts = df['category'].value_counts()
    
    if PLOTLY_AVAILABLE:
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Stock Category Distribution",
            color_discrete_map={
                'Price Up + Delivery Up': '#10B981',
                'Price Flat/Down + Delivery Up': '#3B82F6',
                'Price Down + Delivery Up': '#8B5CF6',
                'Price Down + Delivery Down': '#EF4444',
                'Price Up + Delivery Flat/Down': '#F59E0B',
                'Neutral': '#6B7280'
            }
        )
        return fig
    else:
        # Fallback to Streamlit native chart
        st.subheader("Stock Category Distribution")
        st.bar_chart(category_counts)
        return None

def create_scatter_plot(df):
    """Create scatter plot for price vs delivery changes"""
    if PLOTLY_AVAILABLE:
        fig = px.scatter(
            df, 
            x='price_change_pct', 
            y='deliv_change_pct',
            hover_data=['symbol', 'close_price'],
            color='category',
            title="Price Change vs Delivery Volume Change",
            labels={
                'price_change_pct': 'Price Change (%)',
                'deliv_change_pct': 'Delivery Volume Change (%)'
            },
            color_discrete_map={
                'Price Up + Delivery Up': '#10B981',
                'Price Flat/Down + Delivery Up': '#3B82F6',
                'Price Down + Delivery Up': '#8B5CF6',
                'Price Down + Delivery Down': '#EF4444',
                'Price Up + Delivery Flat/Down': '#F59E0B',
                'Neutral': '#6B7280'
            }
        )
        return fig
    else:
        # Fallback to Streamlit native chart
        st.subheader("Price Change vs Delivery Volume Change")
        if 'deliv_change_pct' in df.columns:
            chart_data = df[['price_change_pct', 'deliv_change_pct']].copy()
            st.scatter_chart(chart_data.rename(columns={
                'price_change_pct': 'Price Change (%)',
                'deliv_change_pct': 'Delivery Volume Change (%)'
            }))
        else:
            st.info("Delivery volume change data not available for scatter plot")
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
        "Google Sheets URL",
        placeholder="Paste your Google Sheets URL here",
        help="Make sure your Google Sheet is publicly accessible"
    )
    
    # Check if URL is provided
    if not sheet_url:
        st.warning("Please provide a Google Sheets URL to load data")
        st.info("📝 **Instructions:**\n1. Make sure your Google Sheet is publicly accessible\n2. Copy the full URL from your browser\n3. Paste it in the sidebar\n4. Your sheet should have columns like: Date, Symbol, Close Price, Volume, etc.")
        st.write("**Expected Column Names (any of these variants will work):**")
        st.write("- **Date**: Date, DateTime, Timestamp, Day, Trading_Date")
        st.write("- **Symbol**: Symbol, Stock_Symbol, Ticker, Scrip, Instrument")
        st.write("- **Price**: Close_Price, Close, Closing_Price, LTP")
        st.write("- **Volume**: Volume, Traded_Quantity, Qty")
        st.write("- **Delivery**: Deliv_Qty, Delivery_Quantity, Delivery_Qty")
        return
    
    # Load data
    df = load_data_from_google_sheets(sheet_url)
    if df is None:
        st.error("Failed to load data from Google Sheets")
        st.info("**Troubleshooting:**\n1. Check if the sheet is publicly accessible\n2. Verify the URL is correct\n3. Ensure the sheet contains the required columns\n4. Check the column names match expected formats")
        return
    
    # Validate data
    if df.empty:
        st.error("No data available in the sheet")
        return
    
    # Display data info
    st.sidebar.success(f"✅ Data loaded: {len(df)} rows")
    
    # Show data preview
    with st.sidebar.expander("Data Preview"):
        st.write("**First 5 rows:**")
        st.dataframe(df.head())
    
    # Date and timeframe selection
    st.sidebar.subheader("Analysis Parameters")
    
    # Get available dates and remove NaT values
    available_dates = df['date'].dropna().dt.date.unique()
    available_dates = sorted([d for d in available_dates if pd.notna(d)])
    
    if len(available_dates) == 0:
        st.error("No valid dates found in the data")
        return
    
    max_date = max(available_dates)
    min_date = min(available_dates)
    
    # Date selection
    selected_date = st.sidebar.date_input(
        "Select Date",
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
    if timeframe == "Custom Range":
        start_date = st.sidebar.date_input("Start Date", value=max_date - timedelta(days=30))
        end_date = st.sidebar.date_input("End Date", value=max_date)
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
    
    # Process data based on timeframe
    if timeframe == "Custom Range":
        processed_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
        # For custom range, calculate change from start to end
        start_data = processed_df[processed_df['date'] == start_date]
        end_data = processed_df[processed_df['date'] == end_date]
        
        if not start_data.empty and not end_data.empty:
            merged = end_data.merge(start_data, on='symbol', suffixes=('', '_start'))
            merged['price_change'] = merged['close_price'] - merged['close_price_start']
            merged['price_change_pct'] = (merged['price_change'] / merged['close_price_start']) * 100
            processed_df = merged
        else:
            st.error("No data available for selected date range")
            return
    else:
        processed_df = calculate_price_changes(df, timeframe, selected_date)
    
    if processed_df is None or processed_df.empty:
        st.error("No data available for the selected parameters")
        st.info("Try selecting a different date or timeframe")
        return
    
    # Categorize stocks
    categorized_df = categorize_stocks(processed_df)
    
    # Filtering options
    st.sidebar.subheader("Filters")
    
    # Search filter
    search_term = st.sidebar.text_input("Search Stocks", "")
    if search_term:
        categorized_df = categorized_df[categorized_df['symbol'].str.contains(search_term, case=False, na=False)]
    
    # Category filter
    categories = ['All'] + list(categorized_df['category'].unique())
    selected_category = st.sidebar.selectbox("Filter by Category", categories)
    if selected_category != 'All':
        categorized_df = categorized_df[categorized_df['category'] == selected_category]
    
    # Significant moves only
    significant_only = st.sidebar.checkbox("Show Significant Moves Only (>2% price change)")
    if significant_only:
        categorized_df = categorized_df[abs(categorized_df['price_change_pct']) > 2]
    
    # Main dashboard
    if not categorized_df.empty:
        # Check for required columns before creating metrics
        required_metrics_cols = ['signal_strength', 'price_change_pct']
        if all(col in categorized_df.columns for col in required_metrics_cols):
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                strong_buy_count = len(categorized_df[categorized_df['signal_strength'] == 'STRONG BUY'])
                st.metric("Strong Buy Signals", strong_buy_count)
            
            with col2:
                accumulation_count = len(categorized_df[categorized_df['signal_strength'] == 'ACCUMULATION'])
                st.metric("Accumulation Signals", accumulation_count)
            
            with col3:
                avg_price_change = categorized_df['price_change_pct'].mean()
                st.metric("Avg Price Change", f"{avg_price_change:.2f}%")
            
            with col4:
                total_stocks = len(categorized_df)
                st.metric("Total Stocks", total_stocks)
        else:
            st.error("Missing required columns for metrics analysis")
            return
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                fig_pie = create_category_pie_chart(categorized_df)
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
            else:
                create_category_pie_chart(categorized_df)
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig_scatter = create_scatter_plot(categorized_df)
                if fig_scatter:
                    st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                create_scatter_plot(categorized_df)
        
        # Top movers
        st.subheader("Top Movers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Gainers**")
            top_gainers = categorized_df.nlargest(5, 'price_change_pct')[['symbol', 'close_price', 'price_change_pct', 'signal_strength']]
            st.dataframe(top_gainers, use_container_width=True)
        
        with col2:
            st.write("**Top Losers**")
            top_losers = categorized_df.nsmallest(5, 'price_change_pct')[['symbol', 'close_price', 'price_change_pct', 'signal_strength']]
            st.dataframe(top_losers, use_container_width=True)
        
        # Detailed table
        st.subheader(f"Detailed Analysis ({len(categorized_df)} stocks)")
        
        # Sort options
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            available_sort_cols = [col for col in ['price_change_pct', 'close_price', 'volume', 'deliv_qty'] if col in categorized_df.columns]
            if available_sort_cols:
                sort_by = st.selectbox("Sort by", available_sort_cols)
            else:
                sort_by = 'price_change_pct'
        with sort_col2:
            sort_order = st.selectbox("Sort order", ['Descending', 'Ascending'])
        
        # Sort dataframe
        ascending = sort_order == 'Ascending'
        display_df = categorized_df.sort_values(sort_by, ascending=ascending)
        
        # Display table with custom formatting
        display_columns = ['symbol', 'close_price', 'price_change_pct', 'volume', 'deliv_qty', 'category', 'signal_strength']
        available_display_cols = [col for col in display_columns if col in display_df.columns]
        
        if available_display_cols:
            display_df_formatted = display_df[available_display_cols].copy()
            
            # Format columns if they exist
            if 'close_price' in display_df_formatted.columns:
                display_df_formatted['close_price'] = display_df_formatted['close_price'].apply(lambda x: f"₹{x:.2f}" if pd.notna(x) else "N/A")
            if 'price_change_pct' in display_df_formatted.columns:
                display_df_formatted['price_change_pct'] = display_df_formatted['price_change_pct'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
            if 'volume' in display_df_formatted.columns:
                display_df_formatted['volume'] = display_df_formatted['volume'].apply(lambda x: f"{x:,}" if pd.notna(x) else "N/A")
            if 'deliv_qty' in display_df_formatted.columns:
                display_df_formatted['deliv_qty'] = display_df_formatted['deliv_qty'].apply(lambda x: f"{x:,}" if pd.notna(x) else "N/A")
            
            st.dataframe(display_df_formatted, use_container_width=True)
        else:
            st.dataframe(display_df, use_container_width=True)
        
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
