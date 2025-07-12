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
