import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import io
import base64

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
        # Convert Google Sheets URL to CSV export URL
        if '/edit' in sheet_url:
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            csv_url = f"https://docs.google.com/spreadsheets/d/1rCqDMaUwrT2mHKeHGjyWAA6vZ5qel-AVg7Atk1ef68Y/edit?gid=988176658#gid=988176658"
        else:
            csv_url = sheet_url
        
        # Read the CSV data
        df = pd.read_csv(csv_url)
        
        # Clean column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Convert date column to datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Convert numeric columns
        numeric_cols = ['prev_close', 'close_price', 'volume', 'deliv_qty', 'deliv_per']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def calculate_price_changes(df, timeframe, selected_date):
    """Calculate price changes based on timeframe"""
    df = df.copy()
    
    if timeframe == 'Today vs Yesterday':
        # Filter for selected date and previous date
        current_data = df[df['date'] == selected_date]
        prev_date = selected_date - timedelta(days=1)
        prev_data = df[df['date'] == prev_date]
        
        if not current_data.empty and not prev_data.empty:
            merged = current_data.merge(prev_data, on='symbol', suffixes=('', '_prev'))
            merged['price_change'] = merged['close_price'] - merged['close_price_prev']
            merged['price_change_pct'] = (merged['price_change'] / merged['close_price_prev']) * 100
            return merged
    
    elif timeframe == 'Weekly':
        # Compare with 7 days ago
        current_data = df[df['date'] == selected_date]
        week_ago_date = selected_date - timedelta(days=7)
        week_ago_data = df[df['date'] <= week_ago_date].groupby('symbol').last().reset_index()
        
        if not current_data.empty and not week_ago_data.empty:
            merged = current_data.merge(week_ago_data, on='symbol', suffixes=('', '_week_ago'))
            merged['price_change'] = merged['close_price'] - merged['close_price_week_ago']
            merged['price_change_pct'] = (merged['price_change'] / merged['close_price_week_ago']) * 100
            return merged
    
    elif timeframe == 'Monthly':
        # Compare with 30 days ago
        current_data = df[df['date'] == selected_date]
        month_ago_date = selected_date - timedelta(days=30)
        month_ago_data = df[df['date'] <= month_ago_date].groupby('symbol').last().reset_index()
        
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
    
    # Define thresholds
    PRICE_THRESHOLD = 0.5  # 0.5%
    DELIV_THRESHOLD = 5    # 5%
    
    # Calculate delivery volume changes (if available)
    if 'deliv_qty_prev' in df.columns or 'deliv_qty_week_ago' in df.columns or 'deliv_qty_month_ago' in df.columns:
        deliv_prev_col = [col for col in df.columns if 'deliv_qty' in col and ('prev' in col or 'week_ago' in col or 'month_ago' in col)]
        if deliv_prev_col:
            df['deliv_change_pct'] = ((df['deliv_qty'] - df[deliv_prev_col[0]]) / df[deliv_prev_col[0]]) * 100
        else:
            df['deliv_change_pct'] = 0
    else:
        df['deliv_change_pct'] = 0
    
    # Categorize based on price and delivery movements
    def get_category(row):
        price_up = row['price_change_pct'] > PRICE_THRESHOLD
        price_down = row['price_change_pct'] < -PRICE_THRESHOLD
        deliv_up = row['deliv_change_pct'] > DELIV_THRESHOLD
        deliv_down = row['deliv_change_pct'] < -DELIV_THRESHOLD
        
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

def create_scatter_plot(df):
    """Create scatter plot for price vs delivery changes"""
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

def get_signal_html(signal):
    """Get HTML for signal badge"""
    signal_classes = {
        'STRONG BUY': 'signal-strong-buy',
        'WEAK BUY': 'signal-weak-buy',
        'ACCUMULATION': 'signal-accumulation',
        'BOTTOM FISHING': 'signal-bottom-fishing',
        'WEAK SELL': 'signal-weak-sell',
        'HOLD': 'signal-hold'
    }
    css_class = signal_classes.get(signal, 'signal-hold')
    return f'<span class="{css_class}">{signal}</span>'

def download_csv(df, filename):
    """Create download link for CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# Main app
def main():
    st.markdown('<h1 class="main-header">📈 NSE Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze price and delivery volume movement patterns for strategic insights</p>', unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.title("Configuration")
    
    # Google Sheets URL input
    sheet_url = st.sidebar.text_input(
        "Google Sheets URL",
        placeholder="Paste your Google Sheets URL here",
        help="Make sure your Google Sheet is publicly accessible"
    )
    
    # Sample data option
    use_sample_data = st.sidebar.checkbox("Use Sample Data", value=True)
    
    if use_sample_data:
        # Create sample data
        sample_data = {
            'symbol': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN'],
            'date': ['2025-01-10'] * 6,
            'prev_close': [1250.5, 3420.8, 1580.2, 1680.3, 1245.6, 820.4],
            'close_price': [1275.3, 3445.2, 1595.8, 1672.1, 1258.9, 825.7],
            'volume': [2500000, 1200000, 1800000, 1500000, 2200000, 3500000],
            'deliv_qty': [1200000, 800000, 900000, 750000, 1100000, 1400000],
            'deliv_per': [48.0, 66.7, 50.0, 50.0, 50.0, 40.0]
        }
        df = pd.DataFrame(sample_data)
        df['date'] = pd.to_datetime(df['date'])
    
    elif sheet_url:
        df = load_data_from_google_sheets(sheet_url)
        if df is None:
            st.error("Failed to load data from Google Sheets")
            return
    else:
        st.warning("Please provide a Google Sheets URL or use sample data")
        return
    
    # Date and timeframe selection
    st.sidebar.subheader("Analysis Parameters")
    
    # Get available dates
    available_dates = sorted(df['date'].unique())
    max_date = max(available_dates)
    
    # Date selection
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=max_date,
        min_value=min(available_dates),
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
        return
    
    # Categorize stocks
    categorized_df = categorize_stocks(processed_df)
    
    # Filtering options
    st.sidebar.subheader("Filters")
    
    # Search filter
    search_term = st.sidebar.text_input("Search Stocks", "")
    if search_term:
        categorized_df = categorized_df[categorized_df['symbol'].str.contains(search_term, case=False)]
    
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
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = create_category_pie_chart(categorized_df)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_scatter = create_scatter_plot(categorized_df)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Top movers
        st.subheader("Top Movers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Gainers**")
            top_gainers = categorized_df.nlargest(5, 'price_change_pct')[['symbol', 'close_price', 'price_change_pct', 'signal_strength']]
            st.dataframe(top_gainers)
        
        with col2:
            st.write("**Top Losers**")
            top_losers = categorized_df.nsmallest(5, 'price_change_pct')[['symbol', 'close_price', 'price_change_pct', 'signal_strength']]
            st.dataframe(top_losers)
        
        # Detailed table
        st.subheader(f"Detailed Analysis ({len(categorized_df)} stocks)")
        
        # Sort options
        sort_col1, sort_col2 = st.columns(2)
        with sort_col1:
            sort_by = st.selectbox("Sort by", ['price_change_pct', 'close_price', 'volume', 'deliv_qty'])
        with sort_col2:
            sort_order = st.selectbox("Sort order", ['Descending', 'Ascending'])
        
        # Sort dataframe
        ascending = sort_order == 'Ascending'
        display_df = categorized_df.sort_values(sort_by, ascending=ascending)
        
        # Display table with custom formatting
        display_columns = ['symbol', 'close_price', 'price_change_pct', 'volume', 'deliv_qty', 'category', 'signal_strength']
        display_df_formatted = display_df[display_columns].copy()
        
        # Format columns
        display_df_formatted['close_price'] = display_df_formatted['close_price'].apply(lambda x: f"₹{x:.2f}")
        display_df_formatted['price_change_pct'] = display_df_formatted['price_change_pct'].apply(lambda x: f"{x:.2f}%")
        display_df_formatted['volume'] = display_df_formatted['volume'].apply(lambda x: f"{x:,}")
        display_df_formatted['deliv_qty'] = display_df_formatted['deliv_qty'].apply(lambda x: f"{x:,}")
        
        st.dataframe(display_df_formatted, use_container_width=True)
        
        # Export functionality
        st.subheader("Export Data")
        if st.button("Download CSV"):
            csv = categorized_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"nse_analysis_{timeframe.lower().replace(' ', '_')}_{selected_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("No data available after applying filters")

if __name__ == "__main__":
    main()
