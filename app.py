import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
import io
import base64

warnings.filterwarnings('ignore')

# --- Dependency Check (from previous fix) ---
def check_dependencies():
    missing_deps = []
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        globals()['px'] = px
        globals()['go'] = go
        globals()['make_subplots'] = make_subplots
    except ImportError:
        missing_deps.append("plotly")
    
    try:
        import openpyxl
        globals()['openpyxl'] = openpyxl
    except ImportError:
        missing_deps.append("openpyxl")
    
    if missing_deps:
        st.error(f"""
        🚨 **Missing Required Dependencies**
        
        The following packages are not installed:
        {', '.join(missing_deps)}
        
        **To fix this:**
        
        1. **If using Streamlit Cloud:**
           - Create a `requirements.txt` file in your repo with:
           ```
           streamlit>=1.28.0
           pandas>=1.5.0
           numpy>=1.24.0
           plotly>=5.15.0
           requests>=2.31.0
           openpyxl>=3.1.0
           ```
           
        2. **If running locally:**
           ```bash
           pip install {" ".join(missing_deps)}
           ```
           
        3. **Then restart your app**
        """)
        st.stop() 

check_dependencies()

# --- Page Configuration and Custom CSS ---
st.set_page_config(
    page_title="NSE Price & Volume Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .alert-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions (from previous fix and new additions) ---

def generate_stock_insights(stock_data):
    insights = []
    if stock_data['PRICE_CHANGE_PCT'] > 5:
        insights.append(f"Significant price increase of {stock_data['PRICE_CHANGE_PCT']:.1f}%.")
    if stock_data['TOTTRDQTY'] > 1000000 and stock_data['VOLUME_CHANGE_PCT'] > 50:
        insights.append(f"High trading volume ({stock_data['TOTTRDQTY']:,.0f}) with a {stock_data['VOLUME_CHANGE_PCT']:.1f}% increase.")
    if stock_data['PV_SIGNAL'] == 'STRONG_BUY':
        insights.append("Strong buy signal detected based on price and volume.")
    elif stock_data['PV_SIGNAL'] == 'STRONG_SELL':
        insights.append("Strong sell signal detected based on price and volume.")
    
    if not insights:
        insights.append("No specific insights detected based on current metrics.")
    return insights

def generate_chat_response(prompt, df):
    """Generate intelligent responses based on the data"""
    prompt_lower = prompt.lower()
    
    try:
        if 'strong buy' in prompt_lower or 'buy signal' in prompt_lower:
            strong_buys = df[df['PV_SIGNAL'] == 'STRONG_BUY']
            if len(strong_buys) > 0:
                top_buys = strong_buys.nlargest(5, 'PRICE_CHANGE_PCT')
                response = f"📈 **{len(strong_buys)} stocks** showing strong buy signals:\n\n"
                for _, stock in top_buys.iterrows():
                    response += f"• **{stock['SYMBOL']}**: {stock['PRICE_CHANGE_PCT']:.1f}% ↑, Volume: {stock['TOTTRDQTY']:,.0f}\n"
                return response
            else:
                return "No strong buy signals found in the current data."
        
        elif 'volume' in prompt_lower and ('high' in prompt_lower or 'unusual' in prompt_lower):
            if 'VOLUME_CHANGE_PCT' in df.columns:
                high_volume = df[df['VOLUME_CHANGE_PCT'] > 50]
                if len(high_volume) > 0:
                    top_volume = high_volume.nlargest(5, 'VOLUME_CHANGE_PCT')
                    response = f"📊 **{len(high_volume)} stocks** with unusual volume activity:\n\n"
                    for _, stock in top_volume.iterrows():
                        response += f"• **{stock['SYMBOL']}**: {stock['VOLUME_CHANGE_PCT']:.1f}% volume increase, Price: {stock['PRICE_CHANGE_PCT']:.1f}%\n"
                    return response
                else:
                    return "No unusual volume activity detected in the current data."
            else:
                return "Volume change percentage data is not available to perform this analysis."
        
        elif 'gainer' in prompt_lower or 'top performer' in prompt_lower:
            top_gainers = df.nlargest(5, 'PRICE_CHANGE_PCT')
            response = "🔥 **Top 5 Gainers Today:**\n\n"
            for _, stock in top_gainers.iterrows():
                response += f"• **{stock['SYMBOL']}**: {stock['PRICE_CHANGE_PCT']:.1f}% ↑, Signal: {stock['PV_SIGNAL']}\n"
            return response
        
        elif 'loser' in prompt_lower or 'worst performer' in prompt_lower:
            top_losers = df.nsmallest(5, 'PRICE_CHANGE_PCT')
            response = "📉 **Top 5 Losers Today:**\n\n"
            for _, stock in top_losers.iterrows():
                response += f"• **{stock['SYMBOL']}**: {stock['PRICE_CHANGE_PCT']:.1f}% ↓, Signal: {stock['PV_SIGNAL']}\n"
            return response
        
        elif any(symbol in prompt.upper() for symbol in df['SYMBOL'].values):
            matched_symbol = None
            for symbol_val in df['SYMBOL'].values:
                if symbol_val in prompt.upper():
                    matched_symbol = symbol_val
                    break

            if matched_symbol:
                stock_data = df[df['SYMBOL'] == matched_symbol].iloc[0]
                insights = generate_stock_insights(stock_data)
                response = f"📊 **Analysis for {matched_symbol}:**\n\n"
                response += f"• Price: ₹{stock_data['CLOSE']:.2f} ({stock_data['PRICE_CHANGE_PCT']:.1f}%)\n"
                response += f"• Volume: {stock_data['TOTTRDQTY']:,.0f}\n"
                response += f"• Signal: {stock_data['PV_SIGNAL']}\n\n"
                response += "**Key Insights:**\n"
                for insight in insights:
                    response += f"• {insight}\n"
                return response
            else:
                return "Could not find the specified stock symbol in the data."
        
        elif 'market' in prompt_lower and 'summary' in prompt_lower:
            total_stocks = len(df)
            gainers = len(df[df['PRICE_CHANGE_PCT'] > 0])
            losers = len(df[df['PRICE_CHANGE_PCT'] < 0])
            avg_change = df['PRICE_CHANGE_PCT'].mean()
            
            response = f"📊 **Market Summary:**\n\n"
            response += f"• Total Stocks: {total_stocks}\n"
            response += f"• Gainers: {gainers} ({gainers/total_stocks*100:.1f}%)\n"
            response += f"• Losers: {losers} ({losers/total_stocks*100:.1f}%)\n"
            response += f"• Average Change: {avg_change:.2f}%\n"
            response += f"• Strong Signals: {len(df[df['PV_SIGNAL'].isin(['STRONG_BUY', 'STRONG_SELL'])])}\n"
            return response
        
        else:
            return """I can help you analyze the NSE data! Try asking:
            
• "Which stocks have strong buy signals?"
• "Show me stocks with unusual volume activity"
• "Top gainers today"
• "Market summary"
• "Analysis for [STOCK_SYMBOL]"

What would you like to know about today's market data?"""
    
    except Exception as e:
        return f"I encountered an error analyzing the data: {str(e)}. Please try rephrasing your question."

@st.cache_data(ttl=300)
def load_nse_data(url=None, uploaded_file=None):
    """Load NSE bhavcopy data from Google Sheets or uploaded file."""
    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("CSV file uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
            return None
    elif url:
        try:
            df = pd.read_csv(url)
            st.success("Data loaded successfully from Google Sheets!")
        except requests.exceptions.RequestException as e:
            st.error(f"Network error loading data from URL: {e}. Please check the URL or your internet connection.")
            return None
        except Exception as e:
            st.error(f"Error loading data from Google Sheet URL: {e}. Ensure it's a direct CSV export link.")
            return None
    else:
        st.info("Please upload a CSV file or enter a Google Sheet URL to proceed.")
        return None

    if df is not None:
        # Clean column names
        df.columns = df.columns.str.strip().str.upper()
        
        # Convert numeric columns
        numeric_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES']
        for col in numeric_cols:
            if col in df.columns:
                # Use errors='coerce' to turn non-numeric values into NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows where essential numeric columns are NaN after conversion
        df.dropna(subset=['CLOSE', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL'], inplace=True)

        if df.empty:
            st.warning("The loaded data is empty or became empty after cleaning. Please check your CSV file.")
            return None

        # Calculate derived metrics
        df['PRICE_CHANGE'] = df['CLOSE'] - df['PREVCLOSE']
        df['PRICE_CHANGE_PCT'] = (df['PRICE_CHANGE'] / df['PREVCLOSE']) * 100
        
        # Handle division by zero for VOLUME_CHANGE_PCT if TOTTRDQTY.shift(1) is 0
        # For a single day's bhavcopy, 'PREV_VOLUME' isn't available.
        # If you need historical volume change, you need multi-day data.
        # For now, calculate change from the previous row in the loaded data.
        df['PREV_TOTTRDQTY'] = df['TOTTRDQTY'].shift(1)
        df['VOLUME_CHANGE_PCT'] = ((df['TOTTRDQTY'] - df['PREV_TOTTRDQTY']) / df['PREV_TOTTRDQTY']) * 100
        df['VOLUME_CHANGE_PCT'] = df['VOLUME_CHANGE_PCT'].fillna(0).replace([np.inf, -np.inf], 0) # Handle inf values

        df['VWAP'] = df['TOTTRDVAL'] / df['TOTTRDQTY']  # Volume Weighted Average Price
        df['TURNOVER_RATIO'] = df['TOTTRDVAL'] / 1000000  # in millions
        
        # Price Volume Analysis
        df['PV_SIGNAL'] = 'NEUTRAL'
        df.loc[(df['PRICE_CHANGE_PCT'] > 2) & (df['VOLUME_CHANGE_PCT'] > 20), 'PV_SIGNAL'] = 'STRONG_BUY'
        df.loc[(df['PRICE_CHANGE_PCT'] > 0) & (df['VOLUME_CHANGE_PCT'] > 10), 'PV_SIGNAL'] = 'BUY'
        df.loc[(df['PRICE_CHANGE_PCT'] < -2) & (df['VOLUME_CHANGE_PCT'] > 20), 'PV_SIGNAL'] = 'STRONG_SELL'
        df.loc[(df['PRICE_CHANGE_PCT'] < 0) & (df['VOLUME_CHANGE_PCT'] > 10), 'PV_SIGNAL'] = 'SELL'
        
    return df

# --- Additional Features and Enhancements ---

def create_download_link(df, filename, file_format='csv'):
    """Create a download link for the filtered data (Deprecated - using st.download_button directly)"""
    # This function is not strictly needed anymore as st.download_button is used directly below.
    # Keeping it commented out for reference if you had older logic using this.
    pass

def calculate_technical_indicators(df):
    """Calculate additional technical indicators"""
    # Ensure df is not empty to avoid errors
    if df.empty:
        return df

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    
    # RSI calculation (simplified) - requires a series for 'prices'
    # RSI requires multiple data points, so if the DataFrame is small, RSI will be NaN
    # For daily bhavcopy, you'd apply this per stock across historical data, not row-wise.
    # For a single day's bhavcopy, RSI calculation won't be meaningful unless you have historical data per stock in your CSV.
    # If this is for a single day, consider removing or adapting this.
    # For now, applying a dummy RSI if the data represents multiple days for a single stock, or per stock if grouped.
    # If your CSV is bhavcopy for ONE day, this RSI calculation for a single row per SYMBOL will result in NaNs.
    # A proper RSI needs time-series data for each stock.
    # I'm removing the inner function and just assigning NaN for now as it won't work correctly for single-day bhavcopy.
    df_copy['RSI'] = np.nan # Placeholder for proper multi-day RSI calculation
    
    # Moving averages
    # These will also be NaN if you only have one day's data for a stock.
    df_copy['SMA_5'] = df_copy['CLOSE'].rolling(window=5, min_periods=1).mean()
    df_copy['SMA_20'] = df_copy['CLOSE'].rolling(window=20, min_periods=1).mean()
    
    # Bollinger Bands
    df_copy['BB_STD'] = df_copy['CLOSE'].rolling(window=20, min_periods=1).std()
    df_copy['BB_UPPER'] = df_copy['SMA_20'] + (df_copy['BB_STD'] * 2)
    df_copy['BB_LOWER'] = df_copy['SMA_20'] - (df_copy['BB_STD'] * 2)
    
    # Price Position
    df_copy['PRICE_POSITION'] = ((df_copy['CLOSE'] - df_copy['LOW']) / (df_copy['HIGH'] - df_copy['LOW'])) * 100
    df_copy['PRICE_POSITION'].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero if High=Low
    
    return df_copy

def calculate_risk_metrics(df):
    """Calculate risk assessment metrics"""
    if df.empty:
        return df

    df_copy = df.copy()
    
    # Volatility - using daily range relative to close
    df_copy['VOLATILITY'] = ((df_copy['HIGH'] - df_copy['LOW']) / df_copy['CLOSE']) * 100
    df_copy['VOLATILITY'].replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero
    df_copy['VOLATILITY'].fillna(0, inplace=True) # Fill any remaining NaNs with 0

    # Risk Score (0-100) - Normalize contributions and sum
    # Ensure all components are numeric and handle NaNs before calculation
    price_change_contrib = df_copy['PRICE_CHANGE_PCT'].abs().fillna(0) * 0.3
    volume_change_contrib = df_copy['VOLUME_CHANGE_PCT'].abs().fillna(0) * 0.3
    volatility_contrib = df_copy['VOLATILITY'].fillna(0) * 0.4
    
    df_copy['RISK_SCORE'] = (price_change_contrib + volume_change_contrib + volatility_contrib).clip(0, 100)
    
    # Risk Categories
    df_copy['RISK_LEVEL'] = pd.cut(
        df_copy['RISK_SCORE'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['LOW', 'MODERATE', 'MEDIUM', 'HIGH', 'VERY HIGH'],
        right=True, # Include the rightmost bin edge
        include_lowest=True # Include the lowest bin edge (0)
    )
    
    return df_copy

def generate_portfolio_suggestions(df, budget=100000, risk_tolerance='MODERATE'):
    """Generate portfolio suggestions based on analysis"""
    suggestions = []
    
    # Filter based on risk tolerance
    suitable_stocks = pd.DataFrame()
    if risk_tolerance == 'LOW':
        suitable_stocks = df[df['RISK_LEVEL'].isin(['LOW', 'MODERATE'])]
    elif risk_tolerance == 'MODERATE':
        suitable_stocks = df[df['RISK_LEVEL'].isin(['LOW', 'MODERATE', 'MEDIUM'])]
    elif risk_tolerance == 'HIGH': 
        suitable_stocks = df[df['RISK_LEVEL'].isin(['LOW', 'MODERATE', 'MEDIUM', 'HIGH', 'VERY HIGH'])]
    
    if suitable_stocks.empty:
        return pd.DataFrame()

    # Get strong buy signals among suitable stocks
    strong_buys = suitable_stocks[suitable_stocks['PV_SIGNAL'] == 'STRONG_BUY'].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    if len(strong_buys) > 0:
        # Normalize scores to 0-1 range before combining, if necessary, or ensure they are on similar scales
        # Here, I'll assume they are somewhat comparable, or you might need normalization logic
        strong_buys['SCORE'] = (
            strong_buys['PRICE_CHANGE_PCT'].fillna(0) * 0.3 +
            strong_buys['VOLUME_CHANGE_PCT'].fillna(0) * 0.2 +
            (100 - strong_buys['RISK_SCORE'].fillna(100)) * 0.5 # Lower risk = higher score
        )
        
        # Ensure 'CLOSE' is not zero or NaN before calculating shares
        strong_buys = strong_buys[strong_buys['CLOSE'] > 0].copy()

        if strong_buys.empty:
            return pd.DataFrame()

        top_picks = strong_buys.nlargest(min(5, len(strong_buys)), 'SCORE') # Take max 5, or fewer if less are available
        
        for _, stock in top_picks.iterrows():
            allocation_per_stock = budget / min(5, len(top_picks)) # Distribute budget among top picks
            shares = int(allocation_per_stock / stock['CLOSE']) if stock['CLOSE'] > 0 else 0
            
            # Only add if shares can be bought
            if shares > 0:
                suggestions.append({
                    'Symbol': stock['SYMBOL'],
                    'Price': stock['CLOSE'],
                    'Allocation': shares * stock['CLOSE'], # Actual allocation based on shares
                    'Shares': shares,
                    'Signal': stock['PV_SIGNAL'],
                    'Risk': stock['RISK_LEVEL'],
                    'Score': stock['SCORE']
                })
    
    return pd.DataFrame(suggestions)

def analyze_market_sentiment(df):
    """Analyze overall market sentiment"""
    if df.empty:
        return {
            'sentiment': "NO DATA", 'score': 0, 'gainers': 0, 'strong_gainers': 0, 
            'losers': 0, 'strong_losers': 0, 'high_volume': 0, 
            'buy_signals': 0, 'sell_signals': 0
        }

    total_stocks = len(df)
    
    gainers = len(df[df['PRICE_CHANGE_PCT'] > 0])
    strong_gainers = len(df[df['PRICE_CHANGE_PCT'] > 3])
    losers = len(df[df['PRICE_CHANGE_PCT'] < 0])
    strong_losers = len(df[df['PRICE_CHANGE_PCT'] < -3])
    
    high_volume = len(df[df['VOLUME_CHANGE_PCT'] > 20])
    
    buy_signals = len(df[df['PV_SIGNAL'].isin(['BUY', 'STRONG_BUY'])])
    sell_signals = len(df[df['PV_SIGNAL'].isin(['SELL', 'STRONG_SELL'])])
    
    # Calculate sentiment score - ensure no division by zero if total_stocks is 0
    sentiment_score = 0
    if total_stocks > 0:
        sentiment_score = (
            (gainers / total_stocks) * 40 +
            (strong_gainers / total_stocks) * 30 +
            (buy_signals / total_stocks) * 30 -
            (strong_losers / total_stocks) * 20 -
            (sell_signals / total_stocks) * 20
        )
    
    if sentiment_score > 60:
        sentiment = "VERY BULLISH 🚀"
    elif sentiment_score > 30:
        sentiment = "BULLISH 📈"
    elif sentiment_score > -30:
        sentiment = "NEUTRAL 😐"
    elif sentiment_score > -60:
        sentiment = "BEARISH 📉"
    else:
        sentiment = "VERY BEARISH 💥"
    
    return {
        'sentiment': sentiment,
        'score': sentiment_score,
        'gainers': gainers,
        'strong_gainers': strong_gainers,
        'losers': losers,
        'strong_losers': strong_losers,
        'high_volume': high_volume,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals
    }

def generate_alerts(df):
    """Generate trading alerts"""
    alerts = []
    if df.empty:
        return alerts

    # Price alerts
    big_movers = df[abs(df['PRICE_CHANGE_PCT']) > 5]
    for _, stock in big_movers.iterrows():
        direction = "📈 UP" if stock['PRICE_CHANGE_PCT'] > 0 else "📉 DOWN"
        alerts.append({
            'type': 'PRICE_ALERT',
            'message': f"{stock['SYMBOL']} moved {direction} {abs(stock['PRICE_CHANGE_PCT']):.1f}%",
            'severity': 'HIGH' if abs(stock['PRICE_CHANGE_PCT']) > 10 else 'MEDIUM'
        })
    
    # Volume alerts
    volume_spikes = df[df['VOLUME_CHANGE_PCT'] > 100]
    for _, stock in volume_spikes.iterrows():
        alerts.append({
            'type': 'VOLUME_ALERT',
            'message': f"{stock['SYMBOL']} volume spike: {stock['VOLUME_CHANGE_PCT']:.1f}%",
            'severity': 'HIGH' if stock['VOLUME_CHANGE_PCT'] > 200 else 'MEDIUM'
        })
    
    # Breakout alerts (closing near high)
    breakouts = df[df['HIGH'] > 0.001] # Avoid division by zero
    breakouts = breakouts[df['CLOSE'] >= df['HIGH'] * 0.99]
    for _, stock in breakouts.iterrows():
        alerts.append({
            'type': 'BREAKOUT_ALERT',
            'message': f"{stock['SYMBOL']} potential breakout - closing near high (₹{stock['CLOSE']:.2f})",
            'severity': 'MEDIUM'
        })
    
    return alerts

# --- Main Streamlit Application Logic ---

st.markdown('<h1 class="main-header">📊 NSE Price & Volume Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Dynamic insights from NSE Bhavcopy data with intelligent analysis")

# --- Data Loading Section (Sidebar) ---
st.sidebar.header("📥 Data Source Selection")

uploaded_file = st.sidebar.file_uploader("Upload Bhavcopy CSV File", type=["csv"])
st.sidebar.markdown("---")
st.sidebar.write("Or enter a Google Sheet CSV Export URL:")
google_sheet_url = st.sidebar.text_input(
    "Google Sheet URL", 
    value="", 
    help="Publish your Google Sheet to web as CSV and paste the URL here. Example: https://docs.google.com/spreadsheets/d/.../gviz/tq?tvd=excel&tqx=out:csv&gid=..."
)

df = None
if uploaded_file or google_sheet_url:
    with st.spinner("Loading and processing data..."):
        df = load_nse_data(url=google_sheet_url, uploaded_file=uploaded_file)

# Placeholder for filtered_df (you'd implement actual filtering logic here)
# For now, filtered_df is just df, but you can add filters later.
filtered_df = pd.DataFrame() 

if df is not None and not df.empty:
    st.sidebar.success(f"Data loaded with {len(df)} records.")
    st.sidebar.markdown(f"Last updated: **{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}**")

    # Apply additional calculations to the loaded DataFrame
    df = calculate_technical_indicators(df)
    df = calculate_risk_metrics(df)
    
    # --- Filtering and Display (Basic placeholder) ---
    st.subheader("🔍 Data Overview & Filters")

    # Basic filter for equity stocks (if 'SERIES' column exists)
    if 'SERIES' in df.columns:
        df_eq = df[df['SERIES'] == 'EQ'].copy()
        if not df_eq.empty:
            filtered_df = df_eq # Use equity data for analysis if available
        else:
            st.warning("No 'EQ' series found in the data. Displaying all available data.")
            filtered_df = df.copy()
    else:
        st.info("No 'SERIES' column found. Displaying all available data.")
        filtered_df = df.copy()
    
    # Symbol search
    all_symbols = ['ALL'] + sorted(filtered_df['SYMBOL'].unique().tolist())
    selected_symbol = st.selectbox("Select Stock Symbol (or ALL)", all_symbols)

    if selected_symbol != 'ALL':
        filtered_df = filtered_df[filtered_df['SYMBOL'] == selected_symbol].copy()

    if filtered_df.empty:
        st.warning("No data matches the current filters. Please adjust your selections.")
    else:
        st.dataframe(filtered_df.head(10), use_container_width=True)

        # --- Dashboard Metrics (Example) ---
        st.markdown("---")
        st.header("📈 Key Performance Indicators")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_stocks = len(filtered_df)
            st.markdown(f'<div class="metric-card"><h4>Total Stocks</h4><p style="font-size: 2rem; color: #1f77b4;">{total_stocks}</p></div>', unsafe_allow_html=True)
        with col2:
            gainers_count = len(filtered_df[filtered_df['PRICE_CHANGE_PCT'] > 0])
            st.markdown(f'<div class="metric-card"><h4>Gainers</h4><p style="font-size: 2rem; color: #28a745;">{gainers_count}</p></div>', unsafe_allow_html=True)
        with col3:
            losers_count = len(filtered_df[filtered_df['PRICE_CHANGE_PCT'] < 0])
            st.markdown(f'<div class="metric-card"><h4>Losers</h4><p style="font-size: 2rem; color: #dc3545;">{losers_count}</p></div>', unsafe_allow_html=True)
        with col4:
            avg_change_pct = filtered_df['PRICE_CHANGE_PCT'].mean()
            st.markdown(f'<div class="metric-card"><h4>Avg. Price Change</h4><p style="font-size: 2rem; color: {"#28a745" if avg_change_pct >= 0 else "#dc3545"};">{avg_change_pct:.2f}%</p></div>', unsafe_allow_html=True)
        
        # --- Chat Interface ---
        st.markdown("---")
        st.header("💬 AI Assistant for Data Insights")
        chat_input = st.text_input("Ask me about the filtered NSE data:", key="chat_input")
        if chat_input:
            with st.spinner("Thinking..."):
                response = generate_chat_response(chat_input, filtered_df)
                st.info(response)

        # --- Additional Features Tabs ---
        st.markdown("---")
        st.header("🔮 Advanced Analytics")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Market Sentiment", 
            "Risk Analysis", 
            "Portfolio Suggestions", 
            "Trading Alerts", 
            "Export Data"
        ])
        
        with tab1:
            st.subheader("📊 Market Sentiment Analysis")
            sentiment_data = analyze_market_sentiment(filtered_df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Market Sentiment", sentiment_data['sentiment'])
                st.metric("Sentiment Score", f"{sentiment_data['score']:.1f}/100")
            with col2:
                st.metric("Gainers", f"{sentiment_data['gainers']}/{len(filtered_df)}")
                st.metric("Strong Gainers", sentiment_data['strong_gainers'])
            with col3:
                st.metric("Buy Signals", sentiment_data['buy_signals'])
                st.metric("High Volume (Vol Change > 20%)", sentiment_data['high_volume'])
            
            sentiment_chart_data = {
                'Category': ['Gainers', 'Losers', 'Neutral'],
                'Count': [
                    sentiment_data['gainers'],
                    sentiment_data['losers'],
                    len(filtered_df) - sentiment_data['gainers'] - sentiment_data['losers']
                ]
            }
            fig_sentiment = px.pie(
                sentiment_chart_data,
                values='Count',
                names='Category',
                title="Market Participation",
                color_discrete_map={'Gainers': 'green', 'Losers': 'red', 'Neutral': 'gray'}
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        with tab2:
            st.subheader("⚠️ Risk Analysis")
            risk_distribution = filtered_df['RISK_LEVEL'].value_counts().reindex(['LOW', 'MODERATE', 'MEDIUM', 'HIGH', 'VERY HIGH'], fill_value=0)
            
            fig_risk = px.bar(
                x=risk_distribution.index,
                y=risk_distribution.values,
                title="Risk Distribution Across Stocks",
                labels={'x': 'Risk Level', 'y': 'Number of Stocks'},
                color=risk_distribution.values,
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            st.subheader("🚨 High Risk Stocks (Top 10)")
            high_risk_stocks = filtered_df[filtered_df['RISK_LEVEL'].isin(['HIGH', 'VERY HIGH'])]
            if not high_risk_stocks.empty:
                high_risk_display = high_risk_stocks[['SYMBOL', 'CLOSE', 'PRICE_CHANGE_PCT', 'VOLATILITY', 'RISK_LEVEL', 'RISK_SCORE']].sort_values('RISK_SCORE', ascending=False)
                st.dataframe(high_risk_display.head(10), use_container_width=True)
            else:
                st.info("No high-risk stocks found in the current filter.")
        
        with tab3:
            st.subheader("💼 Portfolio Suggestions")
            col1, col2 = st.columns(2)
            with col1:
                portfolio_budget = st.number_input("Investment Budget (₹)", min_value=10000, value=100000, step=10000)
            with col2:
                risk_tolerance = st.selectbox("Risk Tolerance", ['LOW', 'MODERATE', 'HIGH'])
            
            portfolio_suggestions = generate_portfolio_suggestions(filtered_df, portfolio_budget, risk_tolerance)
            
            if not portfolio_suggestions.empty:
                st.subheader("🎯 Recommended Portfolio")
                st.dataframe(portfolio_suggestions, use_container_width=True)
                
                fig_allocation = px.pie(
                    portfolio_suggestions,
                    values='Allocation',
                    names='Symbol',
                    title="Suggested Portfolio Allocation"
                )
                st.plotly_chart(fig_allocation, use_container_width=True)
                
                total_allocation = portfolio_suggestions['Allocation'].sum()
                total_shares = portfolio_suggestions['Shares'].sum()
                
                st.markdown(f"""
                **Portfolio Summary:**
                - Total Investment: ₹{total_allocation:,.0f}
                - Total Shares: {total_shares:,}
                - Number of Stocks: {len(portfolio_suggestions)}
                - Risk Level: {risk_tolerance}
                """)
            else:
                st.warning("No suitable stocks found for the selected criteria or insufficient data.")
        
        with tab4:
            st.subheader("🚨 Trading Alerts")
            alerts = generate_alerts(filtered_df)
            
            if alerts:
                high_alerts = [a for a in alerts if a['severity'] == 'HIGH']
                medium_alerts = [a for a in alerts if a['severity'] == 'MEDIUM']
                
                if high_alerts:
                    st.markdown("### 🔴 High Priority Alerts")
                    for alert in high_alerts:
                        st.markdown(f'<div class="alert-box">🚨 {alert["message"]}</div>', unsafe_allow_html=True)
                
                if medium_alerts:
                    st.markdown("### 🟡 Medium Priority Alerts")
                    for alert in medium_alerts:
                        st.markdown(f'<div class="insight-box">⚠️ {alert["message"]}</div>', unsafe_allow_html=True)
                
                st.markdown("### 📊 Alert Summary")
                alert_summary = {
                    'Alert Type': [a['type'] for a in alerts],
                    'Severity': [a['severity'] for a in alerts]
                }
                alert_df = pd.DataFrame(alert_summary)
                alert_counts = alert_df.groupby(['Alert Type', 'Severity']).size().reset_index(name='Count')
                
                fig_alerts = px.bar(
                    alert_counts,
                    x='Alert Type',
                    y='Count',
                    color='Severity',
                    title="Alert Distribution by Type and Severity"
                )
                st.plotly_chart(fig_alerts, use_container_width=True)
            else:
                st.info("No alerts generated for the current data.")
        
        with tab5:
            st.subheader("📥 Export Data")
            export_format = st.selectbox("Export Format", ['CSV', 'Excel'])
            
            col1, col2 = st.columns(2)
            
            with col1:
                filename = f"NSE_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if export_format == 'CSV':
                    csv = filtered_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Filtered Data (CSV)",
                        data=csv,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
                else: # Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        filtered_df.to_excel(writer, sheet_name='NSE_Data', index=False)
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        label="Download Filtered Data (Excel)",
                        data=excel_data,
                        file_name=f"{filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col2:
                if not portfolio_suggestions.empty:
                    portfolio_csv = portfolio_suggestions.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Portfolio Suggestions (CSV)",
                        data=portfolio_csv,
                        file_name=f"Portfolio_Suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No portfolio suggestions to export yet.")
            
            st.markdown("### 📈 Data Summary")
            st.markdown(f"""
            - **Total Records in Filtered Data**: {len(filtered_df):,}
            - **Columns**: {len(filtered_df.columns)}
            - **Data Source**: User Upload / Google Sheet
            """)

else:
    # This block will execute if df is None or empty after load_nse_data
    if not (uploaded_file or google_sheet_url):
        st.info("Please upload a Bhavcopy CSV file or enter a Google Sheet CSV Export URL in the sidebar to begin analysis.")
    st.image("https://www.niftyindices.com/images/nselogosm.gif", width=200) # Example image
    st.markdown("---")
    st.subheader("How to get your NSE Bhavcopy CSV:")
    st.markdown("""
    1.  Go to the official NSE India website (e.g., [www.nseindia.com](https://www.nseindia.com/)).
    2.  Navigate to 'Market Data' -> 'Daily Bhavcopy'.
    3.  Select the desired date and download the Bhavcopy in CSV format.
    4.  Alternatively, if you maintain data in Google Sheets, ensure it's published to the web as a CSV to get a direct download link.
    """)
    

# --- Performance Optimization Tips ---
st.markdown("---")
st.header("⚡ Performance Tips")

with st.expander("🔧 Optimization Suggestions"):
    st.markdown("""
    **For Better Performance:**
    1.  **Limit Data Range**: Use filters to reduce dataset size if available (e.g., specific date ranges).
    2.  **Refresh Strategically**: Data loaded from URL is cached for 5 minutes. Uploaded files are processed on each upload.
    3.  **Export Large Datasets**: Use the export feature for detailed analysis in Excel or other tools.
    4.  **Browser Performance**: Clear browser cache if the dashboard becomes slow.
    5.  **Mobile Usage**: Use desktop/tablet for a better experience with interactive charts.
    
    **Data Refresh:**
    - Data from Google Sheet URL is automatically cached for 5 minutes (`@st.cache_data`).
    - Uploaded files are re-processed on each new upload.
    - Use browser refresh (F5 or Ctrl+R) to force a complete reload of the app and clear cache.
    - Check data timestamp in sidebar for when data was last processed.
    
    **Chart Interactions:**
    - Click and drag to zoom.
    - Double-click to reset zoom.
    - Use legend to filter series.
    - Hover for detailed information.
    """)

# --- API Information ---
st.markdown("---")
st.header("🔗 API & Integration")

with st.expander("📡 API Documentation"):
    st.markdown("""
    **Data Sources:**
    - NSE Bhavcopy (CSV format via upload or Google Sheets URL).
    - Real-time data processing upon load.
    
    **Custom Integration Example (Python):**
    ```python
    import requests
    import pandas as pd
    
    # Load data from your Google Sheets or local CSV
    # For Google Sheets:
    url = "YOUR_GOOGLE_SHEETS_CSV_URL" # Replace with your published CSV URL
    df = pd.read_csv(url)
    
    # For local CSV:
    # df = pd.read_csv("path/to/your/bhavcopy.csv")
    
    # Pre-process columns (similar to what the app does)
    df.columns = df.columns.str.strip().str.upper()
    numeric_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['CLOSE', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL'], inplace=True)
    
    # Calculate derived metrics (example: Price Change PCT)
    df['PRICE_CHANGE_PCT'] = ((df['CLOSE'] - df['PREVCLOSE']) / df['PREVCLOSE']) * 100
    
    # You can then use the functions from this application:
    # from your_app_file import calculate_technical_indicators, calculate_risk_metrics, generate_portfolio_suggestions
    
    # df = calculate_technical_indicators(df)
    # df = calculate_risk_metrics(df)
    # suggestions = generate_portfolio_suggestions(df, budget=50000, risk_tolerance='MODERATE')
    # print(suggestions.head())
    ```
    
    **Available Functions (for external use):**
    - `load_nse_data(url=..., uploaded_file=...)` - Core data loader
    - `calculate_technical_indicators(df)` - Adds SMA, BB, Price Position
    - `calculate_risk_metrics(df)` - Adds Volatility, Risk Score, Risk Level
    - `generate_portfolio_suggestions(df, budget, risk_tolerance)` - Provides stock picks
    - `analyze_market_sentiment(df)` - Gives overall market outlook
    - `generate_alerts(df)` - Identifies specific trading alerts
    - `generate_chat_response(prompt, df)` - AI-driven insights
    """)

st.markdown("---")
st.markdown(f"*Enhanced NSE Analysis Dashboard v2.0 | Built with ❤️ using Streamlit | Last Updated: {datetime.now().strftime('%Y-%m-%d')}*")
