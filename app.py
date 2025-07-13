import streamlit as st
import pandas as pd
import requests
import zipfile
import io
from datetime import datetime, timedelta

# --- Configuration ---
# The URL might need to be dynamic or updated if NSE changes it again.
# Keep the current one, but be aware it might break in the future.
NSE_BHAVCOPY_URL = "https://archives.nseindia.com/content/historical/EQUITIES/{YEAR}/{MON}/cm{DD}{MON}{YEAR}bhav.csv.zip"
MONTH_MAP = {
    1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN",
    7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT", 11: "NOV", 12: "DEC"
}

# Add a list of known holidays for 2025 (you'd need to maintain this)
# This is a partial list, you'd want to get a comprehensive one from NSE or a reliable source.
NSE_HOLIDAYS_2025 = [
    datetime(2025, 2, 26).date(),  # Mahashivratri
    datetime(2025, 3, 14).date(),  # Holi
    datetime(2025, 3, 31).date(),  # Eid-Ul-Fitr
    datetime(2025, 4, 10).date(),  # Shri Mahavir Jayanti
    datetime(2025, 4, 14).date(),  # Dr. Baba Saheb Ambedkar Jayanti
    datetime(2025, 4, 18).date(),  # Good Friday
    datetime(2025, 5, 1).date(),   # Maharashtra Day
    datetime(2025, 8, 15).date(),  # Independence Day
    datetime(2025, 8, 27).date(),  # Ganesh Chaturthi
    datetime(2025, 10, 2).date(), # Mahatma Gandhi Jayanti/Dussehra
    datetime(2025, 10, 21).date(), # Diwali Laxmi Pujan
    datetime(2025, 10, 22).date(), # Diwali-Balipratipada
    datetime(2025, 11, 5).date(),  # Prakash Gurpurb Sri Guru Nanak Dev
    datetime(2025, 12, 25).date()  # Christmas
]

@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid repeated downloads
def fetch_bhavcopy(date: datetime.date):
    """
    Fetches the NSE Bhavcopy for a given date.
    Returns a pandas DataFrame if successful, None otherwise.
    """
    year = date.year
    month_abbr = MONTH_MAP[date.month]
    day = date.day

    url = NSE_BHAVCOPY_URL.format(
        YEAR=year,
        MON=month_abbr,
        DD=f"{day:02d}"
    )

    st.info(f"Attempting to fetch bhavcopy from: {url}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() 

        if not zipfile.is_zipfile(io.BytesIO(response.content)):
            st.error(f"Downloaded file for {date.strftime('%d-%b-%Y')} is not a valid ZIP file. It might be a holiday or data is not yet available.")
            return None

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_file_name = [name for name in z.namelist() if name.endswith('.csv')][0]
            with z.open(csv_file_name) as f:
                df = pd.read_csv(f)
                return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data for {date.strftime('%d-%b-%Y')}: {e}")
        return None
    except IndexError:
        st.error(f"No CSV file found inside the zip for {date.strftime('%d-%b-%Y')}. This might indicate data not being available.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def process_bhavcopy(df: pd.DataFrame):
    if df is None:
        return pd.DataFrame() 

    df.columns = [col.strip().upper() for col in df.columns]

    if 'SERIES' in df.columns:
        df = df[df['SERIES'] == 'EQ']

    required_cols = ['SYMBOL', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'TOTTRDVOL', 'TOTTRDVAL']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.warning(f"Missing expected columns in Bhavcopy data: {', '.join(missing_cols)}. Displaying available data.")

    numeric_cols = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'TOTTRDVOL', 'TOTTRDVAL']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0) 

    if 'PREVCLOSE' in df.columns and 'CLOSE' in df.columns:
        df['CHANGE'] = df['CLOSE'] - df['PREVCLOSE']
        df['% CHANGE'] = (df['CHANGE'] / df['PREVCLOSE']) * 100
    elif 'CLOSE' in df.columns and 'OPEN' in df.columns:
        df['CHANGE'] = df['CLOSE'] - df['OPEN']
        df['% CHANGE'] = (df['CHANGE'] / df['OPEN']) * 100
    else:
        df['CHANGE'] = 0.0
        df['% CHANGE'] = 0.0
        st.warning("Could not calculate daily change due to missing 'PREVCLOSE' or 'OPEN'/'CLOSE' columns.")

    return df

def main():
    st.set_page_config(layout="wide", page_title="NSE Bhavcopy Analyzer")

    st.title("📊 NSE Bhavcopy Data Analyzer")
    st.write("Fetch and analyze daily Bhavcopy data from the National Stock Exchange of India.")

    st.sidebar.header("Settings")
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=datetime.now().date() - timedelta(days=1), # Default to yesterday
        max_value=datetime.now().date() # Cannot select future date
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("How to Use:")
    st.sidebar.info(
        "1. Select a date to fetch Bhavcopy data.\n"
        "2. The app will attempt to download and process the data.\n"
        "3. Explore stock details, top gainers/losers, and high volume stocks."
    )

    if st.button("Fetch & Analyze Bhavcopy"):
        st.subheader(f"Data for {selected_date.strftime('%d %B %Y')}")

        # --- Check for Weekend/Holiday ---
        if selected_date.weekday() >= 5: # 5 is Saturday, 6 is Sunday
            st.warning(f"The selected date, {selected_date.strftime('%d %B %Y')}, is a weekend. NSE is closed on weekends, so no Bhavcopy data is available.")
            st.stop() # Stop further execution

        if selected_date in NSE_HOLIDAYS_2025:
            st.warning(f"The selected date, {selected_date.strftime('%d %B %Y')}, is an NSE trading holiday. No Bhavcopy data is available for this day.")
            st.stop() # Stop further execution

        with st.spinner(f"Fetching Bhavcopy for {selected_date.strftime('%d-%b-%Y')}..."):
            raw_df = fetch_bhavcopy(selected_date)

        if raw_df is not None and not raw_df.empty:
            processed_df = process_bhavcopy(raw_df)

            if not processed_df.empty:
                st.success("Bhavcopy data fetched and processed successfully!")

                st.header("Search Stock by Symbol")
                stock_symbol = st.text_input("Enter Stock Symbol (e.g., RELIANCE, TCS)", "").upper().strip()
                if stock_symbol:
                    found_stock = processed_df[processed_df['SYMBOL'] == stock_symbol]
                    if not found_stock.empty:
                        st.subheader(f"Details for {stock_symbol}")
                        st.dataframe(found_stock)
                    else:
                        st.warning(f"No data found for symbol: {stock_symbol} on {selected_date.strftime('%d-%b-%Y')}")
                else:
                    st.info("Enter a stock symbol to view its details.")

                st.header("Top Gainers and Losers")
                if '% CHANGE' in processed_df.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Top 10 Gainers")
                        top_gainers = processed_df.sort_values(by='% CHANGE', ascending=False).head(10)
                        if not top_gainers.empty:
                            st.dataframe(top_gainers[['SYMBOL', 'CLOSE', 'CHANGE', '% CHANGE', 'TOTTRDVOL']])
                        else:
                            st.info("No gainers found.")

                    with col2:
                        st.subheader("Top 10 Losers")
                        top_losers = processed_df.sort_values(by='% CHANGE', ascending=True).head(10)
                        if not top_losers.empty:
                            st.dataframe(top_losers[['SYMBOL', 'CLOSE', 'CHANGE', '% CHANGE', 'TOTTRDVOL']])
                        else:
                            st.info("No losers found.")
                else:
                    st.info("Cannot determine top gainers/losers due to missing '% CHANGE' data.")

                st.header("Top 20 High Volume Stocks")
                if 'TOTTRDVOL' in processed_df.columns:
                    high_volume_stocks = processed_df.sort_values(by='TOTTRDVOL', ascending=False).head(20)
                    if not high_volume_stocks.empty:
                        st.dataframe(high_volume_stocks[['SYMBOL', 'CLOSE', 'TOTTRDVOL', 'TOTTRDVAL']])
                    else:
                        st.info("No high volume stocks found.")
                else:
                    st.info("Cannot display high volume stocks due to missing 'TOTTRDVOL' data.")

                st.header("Full Bhavcopy Data Table")
                st.dataframe(processed_df, use_container_width=True)

            else:
                st.warning("Processed DataFrame is empty. Please check the selected date or if the data is available.")
        else:
            st.error("Could not fetch or process Bhavcopy data for the selected date. This could be due to a weekend, holiday, or a change in NSE's data availability.")

    st.markdown("---")
    st.caption("Developed by Gemini AI. Data sourced from NSE India.")

if __name__ == "__main__":
    main()
