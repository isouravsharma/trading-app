import streamlit as st
import pandas as pd
from fetcher import get_ohlcv
from processor import MarketDataProcessor
from charts import plot_stock_signals, plot_stock_subplots
from metrices import PerformanceMetrics
from backtest import backtest_mean_reversion

st.set_page_config(page_title="Stock App", layout="wide")

# Sidebar navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Select Page", ["Process Data", "Charts", "Performance"])

# Process Data Page
if page == "Process Data":
    st.title("🧠 Process Market Data")
    # Use session_state to store/retrieve ticker across pages
    if 'ticker' not in st.session_state:
        st.session_state['ticker'] = ""
    ticker = st.text_input("Enter Stock Ticker:", st.session_state['ticker']).upper()
    st.session_state['ticker'] = ticker
    period = st.selectbox("Period", ["1d", "5d", "1mo", "2mo","3mo", "6mo", "1y","5y","max"], index=4)
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d", "1wk"], index=4)

    if st.button("Fetch & Process Data"):
        raw_data = get_ohlcv(ticker, period, interval)
        if raw_data is not None and not raw_data.empty:
            # Flatten MultiIndex columns if present
            if isinstance(raw_data.columns, pd.MultiIndex):
                raw_data.columns = [col[1] for col in raw_data.columns]
            required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
            if not required_cols.issubset(raw_data.columns):
                st.error(f"Data must contain columns: {required_cols}. Got: {set(raw_data.columns)}")
            else:
                processor = MarketDataProcessor(raw_data)
                processor.cummulative_returns()
                processor.add_rsi().add_macd().add_bollinger_bands().add_vwap().add_atr().add_anomaly_detection().add_historical_volatility().add_half_life_signals().add_entry_price()
                processed_data = processor.get_data()

                st.success("Data processed successfully!")
                filtered = processed_data[processed_data['Label'].isin(['BELOW 2 SD', 'BELOW 3 SD'])]
                st.dataframe(filtered.tail(100))

                # Support/Resistance
                support, resistance = processor.get_support_resistance(order=14)

                # Store in session_state for use in Charts/Performance page
                if 'stock_data_dict' not in st.session_state:
                    st.session_state['stock_data_dict'] = {}
                if 'support_dict' not in st.session_state:
                    st.session_state['support_dict'] = {}
                if 'resistance_dict' not in st.session_state:
                    st.session_state['resistance_dict'] = {}

                st.session_state['stock_data_dict'][ticker] = processed_data
                st.session_state['support_dict'][ticker] = support
                st.session_state['resistance_dict'][ticker] = resistance
        else:
            st.warning("No data fetched.")

    tickers = [
        "NVDA", "MSFT", "AAPL", "AMZN", "META", "AVGO", "GOOGL", "TSLA", "BRK-B", "GOOG",
        "JPM", "LLY", "V", "XOM", "COST", "MA", "UNH", "NFLX", "WMT", "PG",
        "JNJ", "HD", "ABBV", "BAC", "CRM", "ORCL", "CSCO", "GE", "PLTR", "IBM",
        "T", "PFE", "KO", "PM", "QCOM", "ACN", "TMO", "MCD", "VZ", "NKE",
        "LIN", "MRK", "ABT", "AMD", "ADBE", "AMAT", "LOW", "DHR", "TXN", "UPS",
        "NEE", "INTU", "SPGI", "MS", "CAT", "UNP", "CVX", "GS", "RTX", "ISRG",
        "AMGN", "INTC", "BLK", "MDT", "DE", "ADI", "NOW", "SCHW", "LRCX", "C",
        "SYK", "MO", "TGT", "MMC", "CI", "ZTS", "ADP", "ELV", "ETN", "PANW",
        "PYPL", "REGN", "BDX", "WM", "SLB", "APD", "FISV", "VRTX", "CB", "SO",
        "CL", "EOG", "DUK", "PGR", "MNST", "HUM", "FCX", "AON", "PSX", "TRV"
    ]

    if st.button("Batch Process 100 Stocks"):
        batch_results = []
        for ticker in tickers:
            try:
                raw_data = get_ohlcv(ticker, period, interval)
                if raw_data is not None and not raw_data.empty:
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        raw_data.columns = [col[1] for col in raw_data.columns]
                    required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
                    if not required_cols.issubset(raw_data.columns):
                        continue
                    processor = MarketDataProcessor(raw_data)
                    processor.cummulative_returns()
                    processor.add_rsi().add_macd().add_bollinger_bands().add_vwap().add_atr().add_anomaly_detection().add_historical_volatility().add_half_life_signals().add_entry_price()
                    processed_data = processor.get_data()
                    processed_data['Ticker'] = ticker
                    # Filter for BELOW 2 SD and BELOW 3 SD
                    filtered = processed_data[processed_data['Label'].isin(['BELOW 2 SD', 'BELOW 3 SD'])]
                    if not filtered.empty:
                        batch_results.append(filtered)
            except Exception as e:
                st.warning(f"{ticker}: {e}")

        if batch_results:
            combined_df = pd.concat(batch_results)
            # Sort by date descending (assuming index is date or has a date column)
            combined_df = combined_df.sort_index(ascending=False)
            st.success(f"Processed {len(batch_results)} stocks with qualifying signals.")
            st.dataframe(combined_df)
        else:
            st.warning("No stocks met the signal condition.")

# Charts Page
elif page == "Charts":
    st.title("📈 Signal Charts")
    ticker = st.session_state.get('ticker', '')
    stock_name = st.text_input("Enter Stock Name for Chart:", ticker).upper()

    # Retrieve from session_state
    stock_data_dict = st.session_state.get('stock_data_dict', {})
    support_dict = st.session_state.get('support_dict', {})
    resistance_dict = st.session_state.get('resistance_dict', {})

    support = support_dict.get(stock_name, [])
    resistance = resistance_dict.get(stock_name, [])

    if stock_name in stock_data_dict:
        st.subheader("Signal Chart")
        single_stock_data = {stock_name: stock_data_dict[stock_name]}
        fig_signal = plot_stock_signals(single_stock_data, support, resistance, stock_name)
        st.plotly_chart(fig_signal, use_container_width=True)

        st.subheader("Subplots Chart")
        fig_subplots = plot_stock_subplots(single_stock_data)
        st.plotly_chart(fig_subplots, use_container_width=True)
    else:
        st.info("Please process data first and ensure the stock name is correct.")

# Performance Page
elif page == "Performance":
    st.title("📊 Strategy Performance Metrics")
    ticker = st.session_state.get('ticker', '')
    stock_name = st.text_input("Enter Stock Name for Performance:", ticker).upper()

    stock_data_dict = st.session_state.get('stock_data_dict', {})
    if stock_name in stock_data_dict:
        df = stock_data_dict[stock_name]
        st.subheader(f"Backtest Results for {stock_name}")

        # Run the mean reversion backtest
        trades_df, summary = backtest_mean_reversion(df, mean_col='m_half_51', tolerance=0.01)

        # Show summary metrics
        st.table(pd.DataFrame(summary, index=["Value"]).T)

        # Show equity curve
        if not trades_df.empty:
            st.line_chart(trades_df.set_index('Exit_Date')['Cumulative_Return'])
            st.subheader("Trade Log")
            st.dataframe(trades_df)
        else:
            st.info("No trades found for this strategy.")
    else:
        st.info("Please process data first and ensure the stock name is correct.")
