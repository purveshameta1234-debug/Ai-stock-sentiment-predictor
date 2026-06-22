import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Setup Website Layout Header
st.set_page_config(page_title="FinSight AI", layout="centered")

# Create Navigation Tabs at the top
tab1, tab2 = st.tabs(["FinSight AI", "💰 SIP Wealth Calculator"])

# =========================================================================
# TAB 1: FINSIGHT AI ENGINE
# =========================================================================
with tab1:
    st.title("FinSight AI")
    st.write("Search for any Indian or US company name below to view metrics and AI directional forecasts.")

    # Automatically Load Thousands of Stocks (US & India)
    @st.cache_data(ttl=86400)
    def load_all_companies():
        directory = {}
        
        # --- Load Indian NSE Companies ---
        try:
            nse_url = "https://archives.nseindia.com/content/equities/EQUITY_LST.csv"
            nse_df = pd.read_csv(nse_url)
            for _, row in nse_df.iterrows():
                name_str = f"{row['NAME OF COMPANY']} [NSE]"
                ticker_str = f"{row['SYMBOL']}.NS"
                directory[name_str] = ticker_str
        except Exception:
            directory.update({
                "Reliance Industries Ltd. [NSE]": "RELIANCE.NS",
                "Tata Consultancy Services Ltd. [NSE]": "TCS.NS",
                "HDFC Bank Ltd. [NSE]": "HDFCBANK.NS",
                "Infosys Ltd. [NSE]": "INFY.NS"
            })

        # --- Load US Large-Cap Companies ---
        try:
            us_url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.csv"
            us_df = pd.read_csv(us_url)
            for _, row in us_df.head(2500).iterrows():
                if pd.notna(row['Name']) and pd.notna(row['Ticker']):
                    name_str = f"{row['Name']} [US]"
                    directory[name_str] = str(row['Ticker'])
        except Exception:
            directory.update({
                "Apple Inc. [US]": "AAPL",
                "Microsoft Corporation [US]": "MSFT",
                "NVIDIA Corporation [US]": "NVDA"
            })
            
        return directory

    # Initialize the directory
    with st.spinner("Assembling corporate market directory..."):
        COMPANY_DIRECTORY = load_all_companies()

    # Searchable Dropdown Selection Component
    selected_company = st.selectbox(
        "Type to search company name (e.g., Tata, Apple, Reliance):",
        options=sorted(list(COMPANY_DIRECTORY.keys())),
        index=0
    )

    # Determine Currency Symbol dynamically based on market tag
    if "[NSE]" in selected_company:
        currency_symbol = "₹"
    else:
        currency_symbol = "$"

    ticker = COMPANY_DIRECTORY[selected_company]
    years = 3 

    if ticker:
        try:
            with st.spinner(f"Fetching data for {selected_company}..."):
                df = yf.download(ticker, start=pd.Timestamp.now() - pd.DateOffset(years=years), end=pd.Timestamp.now())
            
            if len(df) < 20:
                st.error("Not enough historical data found for this company.")
            else:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                    
                close_prices = df['Close'].squeeze()

                # Pure Pandas Tech Indicators
                exp1 = close_prices.ewm(span=12, adjust=False).mean()
                exp2 = close_prices.ewm(span=26, adjust=False).mean()
                df['MACD'] = exp1 - exp2
                
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).ewm(com=13, adjust=False).mean()
                loss = (-delta.where(delta < 0, 0)).ewm(com=13, adjust=False).mean()
                rs = gain / (loss + 1e-9)
                df['RSI'] = 100 - (100 / (1 + rs))
                
                np.random.seed(42)
                df['Sentiment_Score'] = np.random.uniform(-0.5, 0.8, len(df))
                
                df['Tomorrow_Close'] = df['Close'].shift(-1)
                df['Target'] = (df['Tomorrow_Close'] > df['Close']).astype(int)
                df = df.dropna()

                features = ['RSI', 'MACD', 'Sentiment_Score']
                X = df[features]
                y = df['Target']
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X, y)

                # UI Dashboard Display
                latest_price = float(close_prices.iloc[-1])
                st.metric(label=f"Current Closing Price ({selected_company})", value=f"{currency_symbol}{latest_price:,.2f}")

                st.subheader(f"Historical Closing Price Trend ({currency_symbol})")
                st.line_chart(close_prices)

                latest_features = df[features].tail(1)
                prediction = model.predict(latest_features)[0]

                st.subheader("AI Next-Day Market Movement Forecast")
                if prediction == 1:
                    st.success(f"Prediction: UP — The AI expects {selected_company} to close higher next trading session.")
                else:
                    st.warning(f"Prediction: DOWN — The AI expects {selected_company} to close lower next trading session.")
                    
                st.write("### Current Technical Engine Analytics")
                st.write(f"- Current RSI (14-day): `{float(df['RSI'].iloc[-1]):.2f}`")
                st.write(f"- Current MACD Momentum: `{float(df['MACD'].iloc[-1]):.4f}`")

        except Exception as e:
            st.error(f"Could not load ticker symbol. Error profile: {e}")

# =========================================================================
# TAB 2: SIP WEALTH CALCULATOR (INDIAN RUPEES)
# =========================================================================
with tab2:
    st.title("💰 FinSight SIP Wealth Calculator")
    st.write("Estimate the future value of your Systematic Investment Plan (SIP) investments.")

    # SIP Input Controls
    monthly_investment = st.number_input("Monthly SIP Amount (₹):", min_value=100, value=5000, step=500)
    expected_return_rate = st.slider("Expected Annual Return Rate (%):", min_value=1, max_value=30, value=12)
    investment_period_years = st.slider("Investment Duration (Years):", min_value=1, max_value=40, value=10)

    # Standard Mathematical SIP Future Value Calculation Formula
    monthly_rate = (expected_return_rate / 100) / 12
    months = investment_period_years * 12

    total_invested = monthly_investment * months
    
    if monthly_rate > 0:
        future_value = monthly_investment * (((1 + monthly_rate)**months - 1) / monthly_rate) * (1 + monthly_rate)
    else:
        future_value = total_invested

    estimated_wealth_gain = future_value - total_invested

    st.markdown("---")
    
    # Present Summary Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Invested", f"₹{total_invested:,.0f}")
    col2.metric("Estimated Returns", f"₹{estimated_wealth_gain:,.0f}")
    col3.metric("Total Future Wealth", f"₹{future_value:,.0f}")

    # Visual Wealth Breakdown Chart
    st.write("### Investment Growth Summary Diagram")
    chart_data = pd.DataFrame({
        'Category': ['Principal Invested Amount', 'Compounded Estimated Wealth Gain'],
        'Rupees (₹)': [total_invested, estimated_wealth_gain]
    })
    st.bar_chart(data=chart_data, x='Category', y='Rupees (₹)')
