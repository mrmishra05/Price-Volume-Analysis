# App.py
import streamlit as st
import pandas as pd
import numpy as np
import random
import time
import requests # Import requests for fetching data from URL

# Try to import OpenAI with error handling
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    st.warning("OpenAI module not installed. AI features will be disabled.")

# --- Configuration for API Keys ---
# WARNING: Embedding API keys directly in code is NOT recommended for security reasons.
# For production deployments, always use Streamlit Secrets or environment variables.
OPENAI_API_KEY_DIRECT = "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"

# Initialize OpenAI client
if OPENAI_AVAILABLE:
    try:
        # Using the directly embedded key as requested
        openai_client = OpenAI(api_key=OPENAI_API_KEY_DIRECT)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}. Please check your API key.")
        openai_client = None # Set to None if key is missing to prevent errors
else:
    openai_client = None

# --- Data Loading Function ---
@st.cache_data(ttl=600) # Cache data for 10 minutes
def load_data_from_csv(url):
    try:
        df = pd.read_csv(url)
        # Clean column names: lowercase, replace spaces with underscores
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        return df
    except Exception as e:
        st.error(f"Error loading or processing data: {e}")
        st.error("Failed to load or process data from Google Sheets")
        st.error("Troubleshooting:")
        st.error("Check if the sheet is publicly accessible.")
        st.error("Verify the URL is correct.")
        st.error("Ensure the sheet contains the expected raw columns and valid data.")
        st.error("Check the column names match expected formats (case-insensitive, spaces replaced by underscores)")
        return pd.DataFrame() # Return empty DataFrame on error

# --- Simulate Financial Data Functions (now uses loaded data) ---
def fetch_watchlist_data(df):
    """Fetches data for watchlist stocks from the loaded DataFrame."""
    watchlist = ["RELIANCE", "INFY", "TATAMOTORS", "HDFCBANK", "BTC"]
    watchlist_data = {}
    for symbol in watchlist:
        # Case-insensitive search for symbol
        stock_row = df[df["symbol"].str.lower() == symbol.lower()]
        if not stock_row.empty:
            # Assuming the CSV has columns like 'symbol', 'price', 'change', 'sentiment', 'sector'
            # Adjust column names as per your actual CSV structure
            data = stock_row.iloc[0].to_dict()
            watchlist_data[data["symbol"]] = {
                "price": data.get("price", "N/A"),
                "change": data.get("change", "N/A"),
                "sentiment": data.get("sentiment", "neutral"),
                "sector": data.get("sector", "General")
            }
    return watchlist_data

# --- AI-Powered Content Generation Functions ---

def call_openai_chat_model(messages, model="gpt-3.5-turbo"):
    """Helper function to call OpenAI's chat completion API."""
    if not openai_client:
        return "AI service is not configured. Please check API key."
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7, # Adjust creativity
            max_tokens=300 # Limit response length
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with OpenAI API: {e}")
        return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again later."


def generate_hot_story_with_ai(market_context):
    """
    Generates a "Hot Market Story" using OpenAI's LLM based on given market context.
    """
    prompt = (
        f"Based on the following market context, generate a concise, engaging headline "
        f"and a brief impact summary for a 'Hot Market Story' for retail traders. "
        f"Focus on the 'why' behind the movement and affected assets. "
        f"Context: {market_context}"
        f"\n\nFormat your response as: 'Headline: [Your Headline]\nImpact: [Your Impact]'"
    )
    
    messages = [{"role": "user", "content": prompt}]
    generated_text = call_openai_chat_model(messages)

    # Attempt to parse the generated text
    parts = generated_text.split('\nImpact: ')
    if len(parts) == 2:
        title = parts[0].replace('Headline: ', '').strip()
        impact = parts[1].strip()
        return {"title": title, "impact": impact}
    else:
        # Fallback if AI doesn't follow format
        return {"title": "AI Story Generation Issue", "impact": generated_text}


def get_hot_stories_from_ai():
    """Fetches AI-generated hot stories."""
    hot_stories = []
    contexts = [
        "Recent positive earnings reports from major Indian IT companies and optimistic global tech forecasts.",
        "Anticipated surge in vehicle sales during upcoming Indian festivals and government incentives for EVs.",
        "Sudden increase in global crude oil and metal prices due to geopolitical tensions in the Middle East.",
        "Reserve Bank of India's latest monetary policy announcement, keeping key interest rates unchanged, affecting banking sector.",
        "Recent approvals of Bitcoin ETFs in major global markets and increasing institutional adoption of cryptocurrencies."
    ]
    
    # Generate 3 hot stories for the demo
    for _ in range(3):
        context = random.choice(contexts)
        story = generate_hot_story_with_ai(context)
        hot_stories.append(story)
    return hot_stories

# --- Simulated Premium Interests Data (for "Stories Just For You") ---
PREMIUM_INTERESTS_DATA = {
    "AI & Tech": [
        {"title": "Mid-Cap AI Firm 'TechGen' Secures Major Government Contract", "asset": "TechGen Solutions", "signal": "Unusual insider buying detected."},
        {"title": "Semiconductor Demand Soars Amidst AI Boom: Focus on 'ChipInnovate'", "asset": "ChipInnovate Ltd.", "signal": "Early analyst upgrades."},
    ],
    "Renewable Energy": [
        {"title": "Solar Panel Manufacturing Booms: 'SunPower India' Expands Capacity", "asset": "SunPower India", "signal": "Positive regulatory news overlooked by market."},
        {"title": "Green Hydrogen Policy Boosts 'GreenFuel Corp' Prospects", "asset": "GreenFuel Corp", "signal": "Increased institutional interest."},
    ],
    "Healthcare Innovation": [
        {"title": "Biotech Startup 'GeneCure' Announces Breakthrough Drug Trial Results", "asset": "GeneCure Pharma", "signal": "Strong pre-market buzz."},
        {"title": "Hospital Chain 'MediCare' Adopts AI for Patient Management", "asset": "MediCare Hospitals", "signal": "Partnership with leading tech firm."},
    ]
}

def get_personalized_alerts(is_premium, user_interests=None):
    """Simulates GenAI generating personalized trend alerts."""
    alerts = []
    if is_premium and user_interests:
        for interest in user_interests:
            if interest in PREMIUM_INTERESTS_DATA:
                for item in PREMIUM_INTERESTS_DATA[interest]:
                    alerts.append(f"💡 **New Opportunity in {interest}:** {item['title']} ({item['asset']}). Signal: {item['signal']}")
    return alerts

def generate_asset_story_ai_powered(asset_name, data):
    """
    Generates a concise asset story using OpenAI's LLM.
    This would be for individual asset deep dives, potentially pulling more specific real-time data.
    """
    price_change = data.get('change', 'N/A')
    sentiment = data.get('sentiment', 'neutral')
    sector = data.get('sector', 'General')

    prompt = (
        f"Generate a concise, natural language story for {asset_name}, a {sector} sector asset. "
        f"Its current price change is {price_change} and overall sentiment is {sentiment}. "
        f"Explain the likely reasons for its current performance and what factors a retail trader should watch. "
        f"Keep it brief and easy to understand, like a summary for a busy trader."
    )
    
    messages = [{"role": "user", "content": prompt}]
    return call_openai_chat_model(messages)


def analyze_trade_ai_powered(trade_details):
    """Analyzes a past trade using OpenAI's LLM."""
    asset = trade_details['asset']
    trade_type = trade_details['type']
    outcome = trade_details['outcome']
    entry_price = trade_details['entry_price']
    exit_price = trade_details['exit_price']
    trade_date = trade_details['date']

    prompt = (
        f"Analyze a past trade for a retail trader:\n"
        f"Asset: {asset}\n"
        f"Trade Type: {trade_type}\n"
        f"Outcome: {outcome}\n"
        f"Entry Price: {entry_price}\n"
        f"Exit Price: {exit_price}\n"
        f"Trade Date: {trade_date}\n\n"
        f"Provide a concise debrief. If it was a profit, explain what likely went well. "
        f"If a loss, suggest what might have gone wrong and provide 2-3 actionable learning suggestions "
        f"for the trader to improve. Keep the language supportive and educational."
    )
    
    messages = [{"role": "user", "content": prompt}]
    return call_openai_chat_model(messages)


# --- AI Co-Pilot Chat Functionality ---
def get_ai_chat_response(user_message, chat_history_st, watchlist_data):
    """
    Gets an AI agent's response to a user message using OpenAI's LLM.
    Includes basic simulated tool calling logic.
    """
    lower_message = user_message.lower()

    # --- Basic Tool Calling Simulation (App-side logic) ---
    # This is a simple way to direct the AI to specific app functions.
    # For more advanced tool calling, the LLM itself would output a function call.

    # Check for asset story requests
    for asset_name, data in watchlist_data.items():
        # Check for full name or common short forms (e.g., "Reliance" for "Reliance Industries")
        if asset_name.lower() in lower_message or \
           (asset_name.split()[0].lower() in lower_message and \
            any(kw in lower_message for kw in ["story", "doing", "about", "performance"])):
            return generate_asset_story_ai_powered(asset_name, data)
    
    # Check for hot stories/trends requests
    if any(phrase in lower_message for phrase in ["hot stories", "hot trends", "trending", "market trends", "what's happening"]):
        hot_stories = get_hot_stories_from_ai()
        formatted_stories = "\n".join([f"- **{s['title']}**: {s['impact']}" for s in hot_stories])
        return f"Here are today's hot stories:\n{formatted_stories}"
    
    # If no specific tool call is detected, send to general AI chat
    # Prepare chat history for LLM
    llm_chat_history = [{"role": "assistant" if msg["role"] == "assistant" else "user", "content": msg["message"]} for msg in chat_history_st]
    llm_chat_history.append({"role": "user", "content": user_message})

    return call_openai_chat_model(llm_chat_history)


# --- User Authentication (Simplified for demo) ---
def authenticate_user():
    st.sidebar.header("User Status")
    is_premium = st.sidebar.checkbox("Enable Premium Features (Simulated)", value=False)
    return is_premium

# --- Main Streamlit App Layout ---
def main():
    st.set_page_config(layout="wide", page_title="EZTrade: Your AI Co-Pilot")

    st.title("EZTrade: Your AI Co-Pilot for Smarter Trades")
    st.write("Transforming market noise into clear, actionable narratives.")

    is_premium = authenticate_user()
    
    # Load data from CSV
    csv_url = "https://docs.google.com/spreadsheets/d/1rCqDMaUwrT2mHKeHGjyWAA6vZ5qel-AVg7Atk1ef68Y/export?format=csv&gid=988176658"
    df = load_data_from_csv(csv_url)

    # Only proceed if DataFrame is not empty
    if not df.empty:
        watchlist_data = fetch_watchlist_data(df) # Fetch watchlist data from loaded DataFrame

        st.markdown("---")

        # 1. My Watchlist First
        st.header("My Watchlist Stories")
        st.write("Quick insights for the assets you care about most.")

        cols = st.columns(len(watchlist_data))
        for i, (asset, data) in enumerate(watchlist_data.items()):
            with cols[i]:
                card_title = f"{asset} ({data['change']})"
                if data['sentiment'] == 'positive' or data['sentiment'] == 'bullish':
                    st.markdown(f"**<p style='color:green;'>{card_title} ↑</p>**", unsafe_allow_html=True)
                elif data['sentiment'] == 'negative' or data['sentiment'] == 'bearish':
                    st.markdown(f"**<p style='color:red;'>{card_title} ↓</p>**", unsafe_allow_html=True)
                else:
                    st.markdown(f"**<p style='color:orange;'>{card_title} ↔</p>**", unsafe_allow_html=True)

                if st.button(f"View Story for {asset}", key=f"watchlist_btn_{asset}"):
                    with st.expander(f"**{asset} - The Full Story**", expanded=True):
                        # This now calls the AI-powered story generation
                        st.write(generate_asset_story_ai_powered(asset, data))
                        st.write(f"**Sector:** {data['sector']}")
                        st.write("*(This narrative is generated by a live AI based on simulated data.)*")
                st.markdown("---")

        st.markdown("---")

        # 2. Hot Stories of the Day (Now AI-Generated)
        st.header("🔥 Hot Market Stories Today (AI-Generated)")
        st.write("The biggest narratives moving the overall market, synthesized by AI.")

        hot_stories_ai = get_hot_stories_from_ai()
        for story in hot_stories_ai:
            st.subheader(f"Headline: {story['title']}")
            st.write(f"**Impact:** {story['impact']}")
            st.write("*(These stories are dynamically generated by AI based on simulated market context.)*")
            st.markdown("---")

        st.markdown("---")

        # 3. My Interests, My Stories (Premium Feature)
        st.header("💡 Stories Just For You (Premium Feature)")
        if is_premium:
            st.write("AI-powered insights tailored to your specific trading interests, helping you spot unique opportunities.")
            user_selected_interests = st.multiselect(
                "Select your interests to see personalized stories:",
                list(PREMIUM_INTERESTS_DATA.keys()),
                default=["AI & Tech"]
            )
            if user_selected_interests:
                personalized_alerts = get_personalized_alerts(is_premium, user_selected_interests)
                if personalized_alerts:
                    for alert in personalized_alerts:
                        st.markdown(alert)
                        st.write("*(These are early signals and nuanced insights, often missed by the general market.)*")
                        st.markdown("---")
                else:
                    st.info("No personalized stories found for your selected interests today. Try different interests!")
            else:
                st.info("Select interests above to see personalized stories.")
        else:
            st.warning("Unlock 'Stories Just For You' and other advanced features with EZTrade Premium!")
            if st.button("Learn More about Premium"):
                st.write("*(Imagine a link to your pricing page here)*")

        st.markdown("---")

        # 4. Trade Debrief & Learn (Premium Feature)
        st.header("📈 Trade Debrief & Learn (Premium Feature)")
        if is_premium:
            st.write("Get personalized feedback on your past trades to refine your strategy and avoid common pitfalls.")
            st.info("*(In a real app, you would connect your brokerage account securely for automated debriefs.)*")

            st.subheader("Simulate a Trade Debrief:")
            sim_asset = st.selectbox("Select Asset for Debrief:", list(watchlist_data.keys()), key="sim_asset")
            sim_trade_type = st.radio("Trade Type:", ["Buy", "Sell"], key="sim_trade_type")
            sim_entry_price = st.number_input("Entry Price:", value=100.0, key="sim_entry_price")
            sim_exit_price = st.number_input("Exit Price:", value=105.0, key="sim_exit_price")
            sim_date = st.date_input("Trade Date:", pd.to_datetime('today'), key="sim_date")

            sim_outcome = "Profit" if sim_exit_price > sim_entry_price else "Loss"

            if st.button("Analyze This Trade", key="analyze_trade_btn"):
                with st.spinner("Analyzing trade..."):
                    # This now calls the AI-powered trade analysis
                    ai_debrief_response = analyze_trade_ai_powered(
                        {
                            "asset": sim_asset,
                            "type": sim_trade_type,
                            "entry_price": sim_entry_price,
                            "exit_price": sim_exit_price,
                            "date": sim_date.strftime("%Y-%m-%d"),
                            "outcome": sim_outcome
                        }
                    )
                    st.markdown(ai_debrief_response)
                st.write("\n---")
                st.write("#### Personalized Learning Suggestions (from AI):")
                st.write("- Review modules on 'Sentiment Analysis in Volatile Markets'.")
                st.write("- Explore strategies for 'Early Exit Signals' to protect profits/limit losses.")
                st.write("- Understand 'Sectoral Correlations' to anticipate broader market impact.")

        else:
            st.warning("Unlock 'Trade Debrief & Learn' to get personalized feedback on your trading performance with EZTrade Premium!")

        st.markdown("---")

        # --- AI Co-Pilot Chat Section ---
        st.header("💬 EZTrade AI Co-Pilot Chat")
        st.write("Chat with your AI assistant to get quick answers and insights.")

        # Initialize chat history in session state
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "message": "Hello! I'm your EZTrade AI Co-Pilot. How can I help you today? Try asking for 'story for Reliance' or 'hot stories'."})

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["message"])

        # User input
        if prompt := st.chat_input("Ask me anything about the market..."):
            st.session_state.messages.append({"role": "user", "message": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Pass watchlist_data to the AI for potential tool calling
                    ai_response = get_ai_chat_response(prompt, st.session_state.messages, watchlist_data)
                    st.markdown(ai_response)
                st.session_state.messages.append({"role": "assistant", "message": ai_response})

        st.markdown("---")
        st.sidebar.markdown("---")
        st.sidebar.info("EZTrade: Your intelligent co-pilot for the Indian stock market. Built for clarity, powered by AI.")

    else:
        st.error("Could not load data from the provided CSV URL. Please check the URL and accessibility.")

if __name__ == "__main__":
    main()
