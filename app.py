import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
import io
import base64

warnings.filterwarnings('ignore')

# Check for required dependencies
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
           pip install plotly openpyxl
           ```
           
        3. **Then restart your app**
        """)
        st.stop()

# Run dependency check
check_dependencies()

# Now your original code continues here...
# Page config
st.set_page_config(
    page_title="NSE Price & Volume Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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

# Chat response function (moved to top)
def generate_chat_response(prompt, df):
    """Generate intelligent responses based on the data"""
    prompt_lower = prompt.lower()
    
    try:
        # Strong buy signals
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
        
        # Volume analysis
        elif 'volume' in prompt_lower and ('high' in prompt_lower or 'unusual' in prompt_lower):
            high_volume = df[df['VOLUME_CHANGE_PCT'] > 50]
            if len(high_volume) > 0:
                top_volume = high_volume.nlargest(5, 'VOLUME_CHANGE_PCT')
                response = f"📊 **{len(high_volume)} stocks** with unusual volume activity:\n\n"
                for _, stock in top_volume.iterrows():
                    response += f"• **{stock['SYMBOL']}**: {stock['VOLUME_CHANGE_PCT']:.1f}% volume increase, Price: {stock['PRICE_CHANGE_PCT']:.1f}%\n"
                return response
            else:
                return "No unusual volume activity detected in the current data."
        
        # Top gainers
        elif 'gainer' in prompt_lower or 'top performer' in prompt_lower:
            top_gainers = df.nlargest(5, 'PRICE_CHANGE_PCT')
            response = "🔥 **Top 5 Gainers Today:**\n\n"
            for _, stock in top_gainers.iterrows():
                response += f"• **{stock['SYMBOL']}**: {stock['PRICE_CHANGE_PCT']:.1f}% ↑, Signal: {stock['PV_SIGNAL']}\n"
            return response
        
        # Top losers
        elif 'loser' in prompt_lower or 'worst performer' in prompt_lower:
            top_losers = df.nsmallest(5, 'PRICE_CHANGE_PCT')
            response = "📉 **Top 5 Losers Today:**\n\n"
            for _, stock in top_losers.iterrows():
                response += f"• **{stock['SYMBOL']}**: {stock['PRICE_CHANGE_PCT']:.1f}% ↓, Signal: {stock['PV_SIGNAL']}\n"
            return response
        
        # Specific stock query
        elif any(symbol in prompt.upper() for symbol in df['SYMBOL'].values):
            for symbol in df['SYMBOL'].values:
                if symbol in prompt.upper():
                    stock_data = df[df['SYMBOL'] == symbol].iloc[0]
                    insights = generate_stock_insights(stock_data)
                    response = f"📊 **Analysis for {symbol}:**\n\n"
                    response += f"• Price: ₹{stock_data['CLOSE']:.2f} ({stock_data['PRICE_CHANGE_PCT']:.1f}%)\n"
                    response += f"• Volume: {stock_data['TOTTRDQTY']:,.0f}\n"
                    response += f"• Signal: {stock_data['PV_SIGNAL']}\n\n"
                    response += "**Key Insights:**\n"
                    for insight in insights:
                        response += f"• {insight}\n"
                    return response
        
        # Market summary
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
        
        # Default response
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

# Title
st.markdown('<h1 class="main-header">📊 NSE Price & Volume Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("### Dynamic insights from NSE Bhavcopy data with intelligent analysis")

# Data loading function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_nse_data(url):
    """Load NSE bhavcopy data from Google Sheets"""
    try:
        df = pd.read_csv(url)
        # Clean column names
        df.columns = df.columns.str.strip().str.upper()
        
        # Convert numeric columns
        numeric_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate derived metrics
        df['PRICE_CHANGE'] = df['CLOSE'] - df['PREVCLOSE']
        df['PRICE_CHANGE_PCT'] = (df['PRICE_CHANGE'] / df['PREVCLOSE']) * 100
        df['VOLUME_CHANGE'] = df['TOTTRDQTY'] - df.get('PREV_VOLUME', 0)  # If prev volume available
        df['VOLUME_CHANGE_PCT'] = ((df['TOTTRDQTY'] - df.get('PREV_VOLUME', df['TOTTRDQTY'])) / df.get('PREV_VOLUME', df['TOTTRDQTY'])) * 100
        df['VWAP'] = df['TOTTRDVAL'] / df['TOTTRDQTY']  # Volume Weighted Average Price
        df['TURNOVER_RATIO'] = df['TOTTRDVAL'] / 1000000  # in millions
        
        # Price Volume Analysis
        df['PV_SIGNAL'] = 'NEUTRAL'
        df.loc[(df['PRICE_CHANGE_PCT'] > 2) & (df['VOLUME_CHANGE_PCT'] > 20), 'PV_SIGNAL'] = 'STRONG_BUY'
        df.loc[(df['PRICE_CHANGE_PCT'] > 0) & (df['VOLUME_CHANGE_PCT'] > 10), 'PV_SIGNAL'] = 'BUY'
        df.loc[(df['PRICE_CHANGE_PCT'] < -2) & (df['VOLUME_CHANGE_PCT'] > 20), 'PV_SIGNAL'] = 'STRONG_SELL'
        df.loc[(df['PRICE_CHANGE_PCT'] < 0) & (df['VOLUME_CHANGE_PCT'] > 10), 'PV_SIGNAL'] = 'SELL'
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Continue with the rest of your original code...
# [The rest of your functions go here exactly as they were]
