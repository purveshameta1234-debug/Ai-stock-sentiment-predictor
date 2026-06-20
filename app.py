import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Setup Website Layout Header
st.set_page_config(page_title="AI Stock Direction Predictor", layout="centered")
st.title("🔮 AI Stock Market Trend Predictor")
st.write("Enter any stock ticker to pull real-time market data and view AI next-day direction forecasts.")

# 2. Add User Text-Input Component 
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, NVDA, TSLA, MSFT):", value="AAPL").upper()

# 3. Add Timeline Selection Slider
years = st.slider("Select years of training history:", min_value=1, max_value=5, value=3)

if ticker:
    try:
        # Fetch Live Data and Company Info
        with st.spinner(f"Fetching data for {ticker}..."):
            ticker_info = yf.Ticker(ticker)
            df = ticker_info.history(start=pd.Timestamp.now() - pd.DateOffset(years=years), end=pd.Timestamp.now())
            
            # Fetch full company name (fallback to ticker if name isn't found)
            company_name = ticker_info.info.get('longName', ticker)
        
        if len(df) < 20:
            st.error("Not enough historical data found for this ticker symbol.")
        else:
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

            # --- Visual UI Dashboard Components (Showing Full Company Name) ---
            latest_price = float(close_prices.iloc[-1])
            st.metric(label=f"Current Closing Price ({company_name})", value=f"${latest_price:.2f}")

            # Draw Historical Trend Line Chart 
            st.subheader(f"Historical Closing Price Trend for {company_name}")
            st.line_chart(close_prices)

            # Generate the Predictive Forecast
            latest_features = df[features].tail(1)
            prediction = model.predict(latest_features)[0]

            st.subheader("🤖 AI Next-Day Market Movement Forecast")
            if prediction == 1:
                st.success(f"🔮 **Prediction: UP** — The AI expects {company_name} to close higher next trading session.")
            else:
                st.warning(f"🔮 **Prediction: DOWN** — The AI expects {company_name} to close lower next trading session.")
                
            # Print calculated internal engine values
            st.write("### Current Technical Engine Analytics")
            st.write(f"- Current RSI (14-day): `{float(df['RSI'].iloc[-1]):.2f}`")
            st.write(f"- Current MACD Momentum: `{float(df['MACD'].iloc[-1]):.4f}`")

    except Exception as e:
        st.error(f"Could not load ticker symbol. Error profile: {e}")
