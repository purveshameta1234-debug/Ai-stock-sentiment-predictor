import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Setup Website Layout Header
st.set_page_config(page_title="AI Stock Direction Predictor", layout="centered")
st.title("🔮 AI Stock Market Trend Predictor")
st.write("Search for **any** Indian or US company name below. Type to filter the dynamic dropdown.")

# 2. Automatically Load Thousands of Stocks (US & India)
@st.cache_data(ttl=86400) # Caches the list for 24 hours so it loads instantly after the first time
def load_all_companies():
    directory = {}
    
    # --- Load Indian NSE Companies ---
    try:
        nse_url = "https://archives.nseindia.com/content/equities/EQUITY_LST.csv"
        nse_df = pd.read_csv(nse_url)
        for _, row in nse_df.iterrows():
            # Match format: "Reliance Industries Ltd. [NSE]" -> Ticker: "RELIANCE.NS"
            name_str = f"{row['NAME OF COMPANY']} [NSE]"
            ticker_str = f"{row['SYMBOL']}.NS"
            directory[name_str] = ticker_str
    except Exception:
        # Fallback list if exchange servers are down or timing out
        directory.update({
            "Reliance Industries Ltd. [NSE]": "RELIANCE.NS",
            "Tata Consultancy Services Ltd. [NSE]": "TCS.NS",
            "HDFC Bank Ltd. [NSE]": "HDFCBANK.NS",
            "Infosys Ltd. [NSE]": "INFY.NS"
        })

    # --- Load US Large-Cap Companies ---
    try:
        # Using a reliable public curated backup list of major NASDAQ/NYSE assets
        us_url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.csv"
        us_df = pd.read_csv(us_url)
        for _, row in us_df.head(2500).iterrows(): # Load top 2,500 US companies to keep things running fast
            if pd.notna(row['Name']) and pd.notna(row['Ticker']):
                name_str = f"{row['Name']} [US]"
                directory[name_str] = str(row['Ticker'])
    except Exception:
        # Fallback list for US stocks
        directory.update({
            "Apple Inc. [US]": "AAPL",
            "Microsoft Corporation [US]": "MSFT",
            "NVIDIA Corporation [US]": "NVDA"
        })
        
    return directory

# Initialize the giant dictionary directory
with st.spinner("Assembling corporate market directory..."):
    COMPANY_DIRECTORY = load_all_companies()

# 3. Add Searchable Dropdown Selection Component
selected_company = st.selectbox(
    "Type to search company name (e.g., Tata, Apple, Reliance):",
    options=sorted(list(COMPANY_DIRECTORY.keys())),
    index=0
)

# Extract the target string ticker identifier based on the user's choice
ticker = COMPANY_DIRECTORY[selected_company]

# 4. Add Timeline Selection Slider
years = st.slider("Select years of training history:", min_value=1, max_value=5, value=3)

if ticker:
    try:
        # Fetch Live Data
        with st.spinner(f"Fetching data for {selected_company}..."):
            df = yf.download(ticker, start=pd.Timestamp.now() - pd.DateOffset(years=years), end=pd.Timestamp.now())
        
        if len(df) < 20:
            st.error("Not enough historical data found for this company.")
        else:
            # Clean yfinance multi-index frames if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
                
            close_prices = df['Close'].squeeze()

            # Pure Pandas Tech Indicators
            # MACD Calculation
            exp1 = close_prices.ewm(span=12, adjust=False).mean()
            exp2 = close_prices.ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            
            # RSI Calculation
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
            rs = gain / (loss + 1e-9)
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Synthetic Sentiment
            np.random.seed(42)
            df['Sentiment_Score'] = np.random.uniform(-0.5, 0.8, len(df))
            
            # Setup Training Target Matrix
            df['Tomorrow_Close'] = df['Close'].shift(-1)
            df['Target'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
            df = df.dropna()

            # Core AI Modeling Framework
            features = ['RSI', 'MACD', 'Sentiment_Score']
            X = df[features]
            y = df['Target']
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)

            # --- Visual UI Dashboard Components ---
            latest_price = float(close_prices.iloc[-1])
            st.metric(label=f"Current Closing Price ({selected_company})", value=f"${latest_price:.2f}")

            # Draw Historical Trend Line Chart 
            st.subheader(f"Historical Closing Price Trend")
            st.line_chart(close_prices)

            # Generate the Predictive Forecast
            latest_features = df[features].tail(1)
            prediction = model.predict(latest_features)[0]

            st.subheader("🤖 AI Next-Day Market Movement Forecast")
            if prediction == 1:
                st.success(f"🔮 **Prediction: UP** — The AI expects {selected_company} to close higher next trading session.")
            else:
                st.warning(f"🔮 **Prediction: DOWN** — The AI expects {selected_company} to close lower next trading session.")
                
            # Print calculated internal engine values
            st.write("### Current Technical Engine Analytics")
            st.write(f"- Current RSI (14-day): `{float(df['RSI'].iloc[-1]):.2f}`")
            st.write(f"- Current MACD Momentum: `{float(df['MACD'].iloc[-1]):.4f}`")

    except Exception as e:
        st.error(f"Could not load ticker symbol. Error profile: {e}")
