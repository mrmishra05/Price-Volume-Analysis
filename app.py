import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

# Analysis functions
def analyze_price_volume_relationship(df):
    """Analyze price-volume relationship and generate insights"""
    insights = []
    
    # Strong movers with volume
    strong_movers = df[df['PV_SIGNAL'].isin(['STRONG_BUY', 'STRONG_SELL'])]
    if len(strong_movers) > 0:
        insights.append(f"🔥 **{len(strong_movers)} stocks** showing strong price movements with high volume support")
    
    # Unusual volume activity
    high_volume = df[df['VOLUME_CHANGE_PCT'] > 50]
    if len(high_volume) > 0:
        insights.append(f"📈 **{len(high_volume)} stocks** showing unusual volume activity (>50% increase)")
    
    # Price breakouts
    breakouts = df[df['CLOSE'] > df['HIGH'] * 0.98]  # Near day high
    if len(breakouts) > 0:
        insights.append(f"🚀 **{len(breakouts)} stocks** closing near day highs - potential breakouts")
    
    # Volume without price movement (accumulation/distribution)
    accumulation = df[(df['VOLUME_CHANGE_PCT'] > 20) & (abs(df['PRICE_CHANGE_PCT']) < 1)]
    if len(accumulation) > 0:
        insights.append(f"🔄 **{len(accumulation)} stocks** showing volume accumulation with minimal price change")
    
    return insights

def get_sector_analysis(df):
    """Analyze sector-wise performance"""
    if 'SERIES' in df.columns:
        sector_performance = df.groupby('SERIES').agg({
            'PRICE_CHANGE_PCT': ['mean', 'count'],
            'TOTTRDVAL': 'sum'
        }).round(2)
        return sector_performance
    return None

def generate_stock_insights(stock_data):
    """Generate insights for individual stock"""
    insights = []
    
    price_change = stock_data['PRICE_CHANGE_PCT']
    volume_change = stock_data.get('VOLUME_CHANGE_PCT', 0)
    signal = stock_data['PV_SIGNAL']
    
    # Price analysis
    if price_change > 5:
        insights.append(f"📈 Strong bullish momentum with {price_change:.1f}% price increase")
    elif price_change < -5:
        insights.append(f"📉 Significant bearish pressure with {price_change:.1f}% price decline")
    elif abs(price_change) < 0.5:
        insights.append(f"😐 Consolidation phase with minimal price movement ({price_change:.1f}%)")
    
    # Volume analysis
    if volume_change > 100:
        insights.append(f"🔊 Exceptional volume surge of {volume_change:.1f}% - strong institutional interest")
    elif volume_change > 50:
        insights.append(f"📊 High volume activity with {volume_change:.1f}% increase")
    elif volume_change < -30:
        insights.append(f"📉 Below average volume - lack of conviction")
    
    # Combined analysis
    if signal == 'STRONG_BUY':
        insights.append(f"🚀 **STRONG BUY SIGNAL** - Price up with volume support")
    elif signal == 'STRONG_SELL':
        insights.append(f"⚠️ **STRONG SELL SIGNAL** - Price down with volume confirmation")
    elif signal == 'BUY':
        insights.append(f"✅ **BUY SIGNAL** - Positive price action with volume")
    elif signal == 'SELL':
        insights.append(f"❌ **SELL SIGNAL** - Negative price action with volume")
    
    return insights

# Sidebar controls
st.sidebar.header("📊 Dashboard Controls")

# Data source
data_url = st.sidebar.text_input(
    "Google Sheets CSV URL",
    value="https://docs.google.com/spreadsheets/d/1rCqDMaUwrT2mHKeHGjyWAA6vZ5qel-AVg7Atk1ef68Y/export?format=csv&gid=988176658",
    help="Enter your Google Sheets CSV export URL"
)

# Load data
with st.spinner("Loading NSE Bhavcopy data..."):
    df = load_nse_data(data_url)

if df is not None and not df.empty:
    st.sidebar.success(f"✅ Data loaded: {len(df)} stocks")
    
    # Filters
    st.sidebar.subheader("🔍 Filters")
    
    # Price change filter
    price_change_range = st.sidebar.slider(
        "Price Change % Range",
        min_value=-20.0,
        max_value=20.0,
        value=(-20.0, 20.0),
        step=0.5
    )
    
    # Volume filter
    min_volume = st.sidebar.number_input(
        "Minimum Volume",
        min_value=0,
        value=0,
        step=1000
    )
    
    # Signal filter
    signal_filter = st.sidebar.multiselect(
        "Price-Volume Signals",
        options=['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL'],
        default=['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL']
    )
    
    # Apply filters
    filtered_df = df[
        (df['PRICE_CHANGE_PCT'] >= price_change_range[0]) &
        (df['PRICE_CHANGE_PCT'] <= price_change_range[1]) &
        (df['TOTTRDQTY'] >= min_volume) &
        (df['PV_SIGNAL'].isin(signal_filter))
    ]
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Stocks",
            len(filtered_df),
            delta=f"{len(filtered_df) - len(df)}" if len(filtered_df) != len(df) else None
        )
    
    with col2:
        avg_price_change = filtered_df['PRICE_CHANGE_PCT'].mean()
        st.metric(
            "Avg Price Change %",
            f"{avg_price_change:.2f}%",
            delta=f"{avg_price_change:.2f}%"
        )
    
    with col3:
        total_turnover = filtered_df['TOTTRDVAL'].sum() / 1000000
        st.metric(
            "Total Turnover",
            f"₹{total_turnover:,.0f}M"
        )
    
    with col4:
        strong_signals = len(filtered_df[filtered_df['PV_SIGNAL'].isin(['STRONG_BUY', 'STRONG_SELL'])])
        st.metric(
            "Strong Signals",
            strong_signals
        )
    
    # Market insights
    st.markdown("---")
    st.header("🧠 Market Insights")
    
    insights = analyze_price_volume_relationship(filtered_df)
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
    
    # Interactive charts
    st.markdown("---")
    st.header("📈 Interactive Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Price vs Volume", "Top Movers", "Sector Analysis", "Individual Stock"])
    
    with tab1:
        st.subheader("Price Change vs Volume Analysis")
        
        # Price-Volume scatter plot
        fig = px.scatter(
            filtered_df.head(200),  # Limit for performance
            x='PRICE_CHANGE_PCT',
            y='VOLUME_CHANGE_PCT',
            color='PV_SIGNAL',
            size='TOTTRDVAL',
            hover_data=['SYMBOL', 'CLOSE', 'TOTTRDQTY'],
            title="Price Change vs Volume Change Analysis",
            labels={
                'PRICE_CHANGE_PCT': 'Price Change %',
                'VOLUME_CHANGE_PCT': 'Volume Change %'
            }
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Price-Volume heatmap
        st.subheader("Price-Volume Signal Distribution")
        signal_counts = filtered_df['PV_SIGNAL'].value_counts()
        fig_pie = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="Distribution of Price-Volume Signals"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        st.subheader("Top Movers by Price & Volume")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔥 Biggest Gainers**")
            top_gainers = filtered_df.nlargest(10, 'PRICE_CHANGE_PCT')[['SYMBOL', 'CLOSE', 'PRICE_CHANGE_PCT', 'TOTTRDQTY', 'PV_SIGNAL']]
            st.dataframe(top_gainers, use_container_width=True)
        
        with col2:
            st.write("**📉 Biggest Losers**")
            top_losers = filtered_df.nsmallest(10, 'PRICE_CHANGE_PCT')[['SYMBOL', 'CLOSE', 'PRICE_CHANGE_PCT', 'TOTTRDQTY', 'PV_SIGNAL']]
            st.dataframe(top_losers, use_container_width=True)
        
        st.write("**📊 Highest Volume**")
        high_volume = filtered_df.nlargest(10, 'TOTTRDQTY')[['SYMBOL', 'CLOSE', 'PRICE_CHANGE_PCT', 'TOTTRDQTY', 'PV_SIGNAL']]
        st.dataframe(high_volume, use_container_width=True)
    
    with tab3:
        st.subheader("Sector-wise Performance")
        
        if 'SERIES' in filtered_df.columns:
            sector_data = filtered_df.groupby('SERIES').agg({
                'PRICE_CHANGE_PCT': 'mean',
                'TOTTRDVAL': 'sum',
                'SYMBOL': 'count'
            }).round(2)
            sector_data.columns = ['Avg Price Change %', 'Total Turnover', 'Stock Count']
            sector_data = sector_data.sort_values('Avg Price Change %', ascending=False)
            
            fig_sector = px.bar(
                sector_data.reset_index(),
                x='SERIES',
                y='Avg Price Change %',
                title="Average Price Change by Series",
                color='Avg Price Change %',
                color_continuous_scale='RdYlGn'
            )
            fig_sector.update_layout(height=400)
            st.plotly_chart(fig_sector, use_container_width=True)
            
            st.dataframe(sector_data, use_container_width=True)
        else:
            st.info("Sector data not available in the dataset")
    
    with tab4:
        st.subheader("Individual Stock Analysis")
        
        # Stock selector
        selected_stock = st.selectbox(
            "Select a stock for detailed analysis:",
            filtered_df['SYMBOL'].unique(),
            index=0
        )
        
        if selected_stock:
            stock_data = filtered_df[filtered_df['SYMBOL'] == selected_stock].iloc[0]
            
            # Stock metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Current Price", f"₹{stock_data['CLOSE']:.2f}")
            with col2:
                st.metric("Price Change", f"{stock_data['PRICE_CHANGE_PCT']:.2f}%", 
                         delta=f"{stock_data['PRICE_CHANGE']:.2f}")
            with col3:
                st.metric("Volume", f"{stock_data['TOTTRDQTY']:,.0f}")
            with col4:
                st.metric("Signal", stock_data['PV_SIGNAL'])
            
            # Stock insights
            st.subheader(f"📊 Insights for {selected_stock}")
            stock_insights = generate_stock_insights(stock_data)
            for insight in stock_insights:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
            
            # OHLC Chart
            if all(col in stock_data.index for col in ['OPEN', 'HIGH', 'LOW', 'CLOSE']):
                fig_ohlc = go.Figure(data=go.Candlestick(
                    x=[selected_stock],
                    open=[stock_data['OPEN']],
                    high=[stock_data['HIGH']],
                    low=[stock_data['LOW']],
                    close=[stock_data['CLOSE']]
                ))
                fig_ohlc.update_layout(
                    title=f"{selected_stock} - OHLC",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_ohlc, use_container_width=True)
    
    # Chat interface
    st.markdown("---")
    st.header("💬 Ask About Your Data")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant", 
            "content": "Hi! I'm your NSE data analyst. Ask me about price movements, volume analysis, or any specific stocks. Try asking: 'Which stocks have strong buy signals?' or 'Show me stocks with unusual volume activity.'"
        })
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about the market data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response based on data
        with st.chat_message("assistant"):
            response = generate_chat_response(prompt, filtered_df)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.error("❌ Unable to load data. Please check your Google Sheets URL and ensure it's publicly accessible.")
    st.info("Make sure your Google Sheet is published and the URL is in the correct format.")
# Additional Features and Enhancements

# Export functionality
def create_download_link(df, filename, file_format='csv'):
    """Create a download link for the filtered data"""
    if file_format == 'csv':
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV</a>'
    elif file_format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='NSE_Data', index=False)
        excel_data = output.getvalue()
        b64 = base64.b64encode(excel_data).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}.xlsx">Download Excel</a>'
    return href

# Advanced Technical Analysis
def calculate_technical_indicators(df):
    """Calculate additional technical indicators"""
    df = df.copy()
    
    # RSI calculation (simplified)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    # Moving averages
    df['SMA_5'] = df['CLOSE'].rolling(window=5).mean()
    df['SMA_20'] = df['CLOSE'].rolling(window=20).mean()
    
    # Bollinger Bands
    df['BB_UPPER'] = df['SMA_20'] + (df['CLOSE'].rolling(window=20).std() * 2)
    df['BB_LOWER'] = df['SMA_20'] - (df['CLOSE'].rolling(window=20).std() * 2)
    
    # Price Position
    df['PRICE_POSITION'] = ((df['CLOSE'] - df['LOW']) / (df['HIGH'] - df['LOW'])) * 100
    
    return df

# Risk Assessment
def calculate_risk_metrics(df):
    """Calculate risk assessment metrics"""
    df = df.copy()
    
    # Volatility
    df['VOLATILITY'] = ((df['HIGH'] - df['LOW']) / df['CLOSE']) * 100
    
    # Risk Score (0-100)
    df['RISK_SCORE'] = (
        (df['VOLATILITY'] * 0.4) +
        (abs(df['PRICE_CHANGE_PCT']) * 0.3) +
        (df['VOLUME_CHANGE_PCT'].abs() * 0.3)
    ).clip(0, 100)
    
    # Risk Categories
    df['RISK_LEVEL'] = pd.cut(
        df['RISK_SCORE'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['LOW', 'MODERATE', 'MEDIUM', 'HIGH', 'VERY HIGH']
    )
    
    return df

# Portfolio Suggestion Engine
def generate_portfolio_suggestions(df, budget=100000, risk_tolerance='MODERATE'):
    """Generate portfolio suggestions based on analysis"""
    suggestions = []
    
    # Filter based on risk tolerance
    if risk_tolerance == 'LOW':
        suitable_stocks = df[df['RISK_LEVEL'].isin(['LOW', 'MODERATE'])]
    elif risk_tolerance == 'MODERATE':
        suitable_stocks = df[df['RISK_LEVEL'].isin(['LOW', 'MODERATE', 'MEDIUM'])]
    else:  # HIGH
        suitable_stocks = df
    
    # Get strong buy signals
    strong_buys = suitable_stocks[suitable_stocks['PV_SIGNAL'] == 'STRONG_BUY']
    
    if len(strong_buys) > 0:
        # Sort by a combination of factors
        strong_buys['SCORE'] = (
            strong_buys['PRICE_CHANGE_PCT'] * 0.3 +
            strong_buys['VOLUME_CHANGE_PCT'] * 0.2 +
            (100 - strong_buys['RISK_SCORE']) * 0.5
        )
        
        top_picks = strong_buys.nlargest(5, 'SCORE')
        
        for _, stock in top_picks.iterrows():
            allocation = budget / 5  # Equal allocation
            shares = int(allocation / stock['CLOSE'])
            suggestions.append({
                'Symbol': stock['SYMBOL'],
                'Price': stock['CLOSE'],
                'Allocation': allocation,
                'Shares': shares,
                'Signal': stock['PV_SIGNAL'],
                'Risk': stock['RISK_LEVEL'],
                'Score': stock['SCORE']
            })
    
    return pd.DataFrame(suggestions)

# Market Sentiment Analysis
def analyze_market_sentiment(df):
    """Analyze overall market sentiment"""
    total_stocks = len(df)
    
    # Price sentiment
    gainers = len(df[df['PRICE_CHANGE_PCT'] > 0])
    strong_gainers = len(df[df['PRICE_CHANGE_PCT'] > 3])
    losers = len(df[df['PRICE_CHANGE_PCT'] < 0])
    strong_losers = len(df[df['PRICE_CHANGE_PCT'] < -3])
    
    # Volume sentiment
    high_volume = len(df[df['VOLUME_CHANGE_PCT'] > 20])
    
    # Signal sentiment
    buy_signals = len(df[df['PV_SIGNAL'].isin(['BUY', 'STRONG_BUY'])])
    sell_signals = len(df[df['PV_SIGNAL'].isin(['SELL', 'STRONG_SELL'])])
    
    # Overall sentiment score
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

# Alert System
def generate_alerts(df):
    """Generate trading alerts"""
    alerts = []
    
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
    
    # Breakout alerts
    breakouts = df[df['CLOSE'] >= df['HIGH'] * 0.99]
    for _, stock in breakouts.iterrows():
        alerts.append({
            'type': 'BREAKOUT_ALERT',
            'message': f"{stock['SYMBOL']} potential breakout - closing near high",
            'severity': 'MEDIUM'
        })
    
    return alerts

# Add these features to the existing dashboard
if df is not None and not df.empty:
    
    # Enhanced calculations
    df = calculate_technical_indicators(df)
    df = calculate_risk_metrics(df)
    
    # Additional tabs for new features
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
            st.metric("High Volume", sentiment_data['high_volume'])
        
        # Sentiment visualization
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
        
        # Risk distribution
        risk_distribution = filtered_df['RISK_LEVEL'].value_counts()
        
        fig_risk = px.bar(
            x=risk_distribution.index,
            y=risk_distribution.values,
            title="Risk Distribution Across Stocks",
            labels={'x': 'Risk Level', 'y': 'Number of Stocks'},
            color=risk_distribution.values,
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_risk, use_container_width=True)
        
        # High risk stocks
        st.subheader("🚨 High Risk Stocks")
        high_risk_stocks = filtered_df[filtered_df['RISK_LEVEL'].isin(['HIGH', 'VERY HIGH'])]
        
        if len(high_risk_stocks) > 0:
            high_risk_display = high_risk_stocks[['SYMBOL', 'CLOSE', 'PRICE_CHANGE_PCT', 'VOLATILITY', 'RISK_LEVEL', 'RISK_SCORE']].sort_values('RISK_SCORE', ascending=False)
            st.dataframe(high_risk_display.head(10), use_container_width=True)
        else:
            st.info("No high-risk stocks found in the current filter.")
    
    with tab3:
        st.subheader("💼 Portfolio Suggestions")
        
        # Portfolio parameters
        col1, col2 = st.columns(2)
        
        with col1:
            portfolio_budget = st.number_input("Investment Budget (₹)", min_value=10000, value=100000, step=10000)
        
        with col2:
            risk_tolerance = st.selectbox("Risk Tolerance", ['LOW', 'MODERATE', 'HIGH'])
        
        # Generate suggestions
        portfolio_suggestions = generate_portfolio_suggestions(filtered_df, portfolio_budget, risk_tolerance)
        
        if len(portfolio_suggestions) > 0:
            st.subheader("🎯 Recommended Portfolio")
            
            # Display suggestions
            st.dataframe(portfolio_suggestions, use_container_width=True)
            
            # Portfolio allocation chart
            fig_allocation = px.pie(
                portfolio_suggestions,
                values='Allocation',
                names='Symbol',
                title="Suggested Portfolio Allocation"
            )
            st.plotly_chart(fig_allocation, use_container_width=True)
            
            # Portfolio summary
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
            st.warning("No suitable stocks found for the selected criteria.")
    
    with tab4:
        st.subheader("🚨 Trading Alerts")
        
        alerts = generate_alerts(filtered_df)
        
        if alerts:
            # Group alerts by severity
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
            
            # Alert summary
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
        
        # Export options
        export_format = st.selectbox("Export Format", ['CSV', 'Excel'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Filtered Data"):
                filename = f"NSE_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if export_format == 'CSV':
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"{filename}.csv",
                        mime="text/csv"
                    )
                else:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        filtered_df.to_excel(writer, sheet_name='NSE_Data', index=False)
                    
                    st.download_button(
                        label="Download Excel",
                        data=output.getvalue(),
                        file_name=f"{filename}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            if len(portfolio_suggestions) > 0 and st.button("Export Portfolio Suggestions"):
                portfolio_csv = portfolio_suggestions.to_csv(index=False)
                st.download_button(
                    label="Download Portfolio CSV",
                    data=portfolio_csv,
                    file_name=f"Portfolio_Suggestions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # Data summary
        st.markdown("### 📈 Data Summary")
        st.markdown(f"""
        - **Total Records**: {len(filtered_df):,}
        - **Date Range**: {datetime.now().strftime('%Y-%m-%d')}
        - **Columns**: {len(filtered_df.columns)}
        - **Data Source**: NSE Bhavcopy
        """)

# Performance optimization tips
st.markdown("---")
st.header("⚡ Performance Tips")

with st.expander("🔧 Optimization Suggestions"):
    st.markdown("""
    **For Better Performance:**
    1. **Limit Data Range**: Use filters to reduce dataset size
    2. **Refresh Strategically**: Only refresh when needed (data is cached for 5 minutes)
    3. **Export Large Datasets**: Use export feature for detailed analysis in Excel
    4. **Browser Performance**: Clear cache if dashboard becomes slow
    5. **Mobile Usage**: Use desktop/tablet for better experience with charts
    
    **Data Refresh:**
    - Data is automatically cached for 5 minutes
    - Use browser refresh to force reload
    - Check data timestamp in sidebar
    
    **Chart Interactions:**
    - Click and drag to zoom
    - Double-click to reset zoom
    - Use legend to filter series
    - Hover for detailed information
    """)

# API Information
st.markdown("---")
st.header("🔗 API & Integration")

with st.expander("📡 API Documentation"):
    st.markdown("""
    **Data Sources:**
    - NSE Bhavcopy (CSV format)
    - Google Sheets integration
    - Real-time data processing
    
    **Custom Integration:**
    ```python
    # Example API call structure
    import requests
    import pandas as pd
    
    # Load data from your Google Sheets
    url = "YOUR_GOOGLE_SHEETS_CSV_URL"
    df = pd.read_csv(url)
    
    # Process with our functions
    df = calculate_technical_indicators(df)
    df = calculate_risk_metrics(df)
    ```
    
    **Available Functions:**
    - `calculate_technical_indicators(df)` - Add technical analysis
    - `calculate_risk_metrics(df)` - Risk assessment
    - `generate_portfolio_suggestions(df, budget, risk)` - Portfolio optimization
    - `analyze_market_sentiment(df)` - Market sentiment analysis
    - `generate_alerts(df)` - Trading alerts
    """)
# Footer
st.markdown("---")
st.markdown("### 📝 Additional Features")
st.markdown("""
**Key Features:**
- Real-time NSE data analysis with price-volume insights
- Interactive charts and visualizations
- Intelligent chat interface for data queries
- Customizable filters and analysis parameters
- Sector-wise performance analysis
- Individual stock deep-dive analysis

**Usage Tips:**
- Use the sidebar filters to narrow down your analysis
- Try different combinations of price and volume filters
- Ask specific questions in the chat interface
- Export data by right-clicking on tables
- Refresh the page to reload latest data

**Chat Commands:**
- "Market summary" - Get overall market overview
- "Top gainers" - View best performing stocks
- "Strong buy signals" - Find stocks with buy momentum
- "Unusual volume" - Detect volume anomalies
- "[SYMBOL] analysis" - Get detailed stock analysis
""")

st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit | Data: NSE Bhavcopy*")
