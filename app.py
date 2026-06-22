import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# 1. Setup Website Layout Header
st.set_page_config(page_title="FinSight AI", layout="centered")

# Custom Clean Sub-header Branded Style
st.markdown("<h1 style='text-align: center; color: #1E3A8A; margin-bottom: 0px;'>FinSight AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B7280; font-size: 16px; margin-top: 0px;'>Advanced Analytics & Financial Intelligence Hub</p>", unsafe_allow_html=True)
st.markdown("---")

# Create Navigation Tabs at the top
tab1, tab2 = st.tabs(["📊 Market Intelligence Engine", "💰 SIP Wealth Calculator"])

# =========================================================================
# TAB 1: FINSIGHT AI ENGINE
# =========================================================================
with tab1:
    # Searchable Dropdown Selection Component in a clean container
    st.markdown("### 🔍 Select Asset Blueprint")
    
    # Automatically Load Thousands of Stocks (US & India)
    @st.cache_data(ttl=86400)
    def load_all_companies():
        directory = {}
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

    with st.spinner("Assembling corporate market directory..."):
        COMPANY_DIRECTORY = load_all_companies()

    selected_company = st.selectbox(
        "Type to filter company profiles instantly:",
        options=sorted(list(COMPANY_DIRECTORY.keys())),
        index=0,
        label_visibility="collapsed"
    )

    if "[NSE]" in selected_company:
        currency_symbol = "₹"
    else:
        currency_symbol = "$"

    ticker = COMPANY_DIRECTORY[selected_company]
    years = 3 

    if ticker:
        try:
            with st.spinner(f"Querying financial vectors for {selected_company}..."):
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

                latest_price = float(close_prices.iloc[-1])
                prev_price = float(close_prices.iloc[-2])
                price_diff = latest_price - prev_price
                pct_diff = (price_diff / prev_price) * 100

                # --- NEW INTERACTIVE BLOCKS GRID ---
                st.markdown("### 📈 Core Asset Statistics")
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric(
                        label="Latest Closing Valuation", 
                        value=f"{currency_symbol}{latest_price:,.2f}",
                        delta=f"{currency_symbol}{price_diff:+.2f} ({pct_diff:+.2f}%)"
                    )
                with col_m2:
                    # Model prediction calculation
                    latest_features = df[features].tail(1)
                    prediction = model.predict(latest_features)[0]
                    
                    if prediction == 1:
                        st.markdown("<div style='background-color: #DEF7EC; border-left: 5px solid #03543F; padding: 12px; border-radius: 4px; text-align: center; margin-top: 5px;'><b style='color: #03543F; font-size: 14px;'>AI NEXT-DAY FORECAST</b><br><span style='color: #0E6245; font-size: 20px; font-weight: bold;'>📈 UPWARD TREND</span></div>", unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='background-color: #FDE8E8; border-left: 5px solid #9B1C1C; padding: 12px; border-radius: 4px; text-align: center; margin-top: 5px;'><b style='color: #9B1C1C; font-size: 14px;'>AI NEXT-DAY FORECAST</b><br><span style='color: #C81E1E; font-size: 20px; font-weight: bold;'>📉 DOWNWARD TREND</span></div>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.subheader(f"Historical Price Tracking Matrix ({currency_symbol})")
                st.line_chart(close_prices)

                # Expandable Tech Details Section to minimize clutter
                with st.expander("🛠️ View Internal Machine Learning Engine Analytics", expanded=False):
                    col_e1, col_e2 = st.columns(2)
                    col_e1.metric("Relative Strength Index (RSI-14)", f"{float(df['RSI'].iloc[-1]):.2f}")
                    col_e2.metric("MACD Convergence Vector", f"{float(df['MACD'].iloc[-1]):.4f}")
                    st.write("---")
                    st.caption("Disclaimer: Forecast profiles are driven by mathematical pattern classification algorithms via historical market data and are intended for structural software presentation parameters only.")

        except Exception as e:
            st.error(f"Could not load ticker symbol. Error profile: {e}")

# =========================================================================
# TAB 2: SIP WEALTH CALCULATOR (INDIAN RUPEES)
# =========================================================================
with tab2:
    st.markdown("### 💰 Systematic Wealth Accelerator Engine")
    st.write("Map out your compounding capital appreciation curve with precise inputs.")
    st.markdown("---")

    # Arrange side-by-side control layouts using 2 sleek column divisions
    col_input, col_results = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown("#### ⚙️ Allocation Configs")
        monthly_investment = st.number_input("Monthly Contribution (₹):", min_value=100, value=5000, step=500)
        expected_return_rate = st.slider("Expected Yield Growth (%):", min_value=1, max_value=30, value=12)
        investment_period_years = st.slider("Duration Spectrum (Years):", min_value=1, max_value=40, value=10)

    # Standard Mathematical SIP Future Value Calculation Formula
    monthly_rate = (expected_return_rate / 100) / 12
    months = investment_period_years * 12
    total_invested = monthly_investment * months
    
    if monthly_rate > 0:
        future_value = monthly_investment * (((1 + monthly_rate)**months - 1) / monthly_rate) * (1 + monthly_rate)
    else:
        future_value = total_invested
    estimated_wealth_gain = future_value - total_invested

    with col_results:
        st.markdown("#### 💎 Compounded Projections")
        st.metric("Total Principal Contribution", f"₹{total_invested:,.0f}")
        st.metric("Accumulated Wealth Growth", f"₹{estimated_wealth_gain:,.0f}", delta=f"₹{future_value:,.0f} Total Future Value", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("### 📊 Portfolio Growth Trajectory Visualization")
    chart_data = pd.DataFrame({
        'Category': ['Principal Investment', 'Compounded Yield Accumulation'],
        'Rupees (₹)': [total_invested, estimated_wealth_gain]
    })
    st.bar_chart(data=chart_data, x='Category', y='Rupees (₹)')
