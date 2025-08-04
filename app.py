import streamlit as st
import pandas as pd
from datetime import datetime
from fetcher import get_ohlcv
from processor import MarketDataProcessor
from charts import plot_stock_signals, plot_stock_subplots
from metrices import PerformanceMetrics
from backtest import backtest_mean_reversion, backtest_mean_reversion_tp_sl
from papertrading import PaperTrader

st.set_page_config(page_title="Stock App", layout="wide")

# Sidebar navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Select Page", ["Process Data", "Charts", "Performance", "Backtest Strategy", "Paper Trading"])

# Process Data Page
if page == "Process Data":
    st.title("🧠 Process Market Data")
    # Use session_state to store/retrieve ticker across pages
    if 'ticker' not in st.session_state:
        st.session_state['ticker'] = ""
    ticker = st.text_input("Enter Stock Ticker:", st.session_state['ticker']).upper()
    st.session_state['ticker'] = ticker
    period = st.selectbox("Period", ["1d", "5d", "1mo", "2mo","3mo", "6mo", "1y","5y","max"], index=6)
    interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "4h", "1d", "1wk"], index=5)

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

    top_200_indian_stocks = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "LT.NS", "HINDUNILVR.NS", "SBIN.NS", "ITC.NS", "BHARTIARTL.NS",
        "KOTAKBANK.NS", "AXISBANK.NS", "HCLTECH.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS", "WIPRO.NS", "NTPC.NS", "POWERGRID.NS",
        "ADANIENT.NS", "ONGC.NS", "TITAN.NS", "ULTRACEMCO.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "TECHM.NS", "NESTLEIND.NS", "HDFCLIFE.NS", "DRREDDY.NS",
        "COALINDIA.NS", "BAJAJ-AUTO.NS", "GRASIM.NS", "BPCL.NS", "DIVISLAB.NS", "BRITANNIA.NS", "EICHERMOT.NS", "CIPLA.NS", "HINDALCO.NS", "SBILIFE.NS",
        "BAJAJFINSV.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "UPL.NS", "APOLLOHOSP.NS", "ICICIPRULI.NS", "ADANIPORTS.NS", "TATAMOTORS.NS", "VEDL.NS", "PIDILITIND.NS",
        "GODREJCP.NS", "AMBUJACEM.NS", "DMART.NS", "DLF.NS", "HAVELLS.NS", "ICICIGI.NS", "AUROPHARMA.NS", "SRF.NS", "MUTHOOTFIN.NS", "LTI.NS",
        "SIEMENS.NS", "ABB.NS", "BANKBARODA.NS", "COLPAL.NS", "GAIL.NS", "BERGEPAINT.NS", "M&MFIN.NS", "INDIGO.NS", "ZOMATO.NS", "IDFCFIRSTB.NS",
        "CANBK.NS", "NAUKRI.NS", "TVSMOTOR.NS", "TRENT.NS", "BEL.NS", "CHOLAFIN.NS", "INDUSTOWER.NS", "BOSCHLTD.NS", "MPHASIS.NS", "PAGEIND.NS",
        "PNB.NS", "PETRONET.NS", "IOC.NS", "YESBANK.NS", "IRCTC.NS", "NHPC.NS", "RECLTD.NS", "PFC.NS", "JINDALSTEL.NS", "TATACOMM.NS",
        "VOLTAS.NS", "GLENMARK.NS", "ALKEM.NS", "MOTHERSON.NS", "BHEL.NS", "TATAPOWER.NS", "AMARAJABAT.NS", "ESCORTS.NS", "HINDPETRO.NS", "IDEA.NS",
        "FORTIS.NS", "LUPIN.NS", "CONCOR.NS", "BIOCON.NS", "CROMPTON.NS", "ABFRL.NS", "LICI.NS", "NYKAA.NS", "POLYCAB.NS", "DEEPAKNTR.NS",
        "BALRAMCHIN.NS", "RAIN.NS", "WHIRLPOOL.NS", "INDIAMART.NS", "IRFC.NS", "SJVN.NS", "JSWENERGY.NS", "ASHOKLEY.NS", "AUBANK.NS", "AIAENG.NS",
        "PHOENIXLTD.NS", "SAIL.NS", "TV18BRDCST.NS", "RBLBANK.NS", "FEDERALBNK.NS", "RAMCOCEM.NS", "CESC.NS", "IDBI.NS", "HONAUT.NS", "VGUARD.NS",
        "ADANIGREEN.NS", "ADANITRANS.NS", "LTIM.NS", "ZENSARTECH.NS", "TATAELXSI.NS", "INDIACEM.NS", "LALPATHLAB.NS", "COFORGE.NS", "BALKRISIND.NS", "SPARC.NS",
        "SUNDARMFIN.NS", "IBULHSGFIN.NS", "NATIONALUM.NS", "MCX.NS", "NAVINFLUOR.NS", "KPITTECH.NS", "KEI.NS", "ROUTE.NS", "ASTERDM.NS", "HAPPSTMNDS.NS",
        "FSL.NS", "TATACONSUM.NS", "JUBLFOOD.NS", "IEX.NS", "PERSISTENT.NS", "SOBHA.NS", "AFFLE.NS", "BORORENEW.NS", "INOXWIND.NS", "TRIDENT.NS",
        "NCC.NS", "UBL.NS", "IRB.NS", "JSWINFRA.NS", "KPRMILL.NS", "MAZDOCK.NS", "RVNL.NS", "GRINDWELL.NS", "GREENPANEL.NS", "TIMKEN.NS",
        "KALYANKJIL.NS", "MANKIND.NS", "DEVYANI.NS", "RATNAMANI.NS", "CLEAN.NS", "ANGELONE.NS", "FINEORG.NS", "ICRA.NS", "BATAINDIA.NS", "RENUKA.NS",
        "IIFL.NS", "PRAJIND.NS", "TATAINVEST.NS", "APLLTD.NS", "METROPOLIS.NS", "SFL.NS", "MAHSEAMLES.NS", "CANFINHOME.NS", "CYIENT.NS", "EIDPARRY.NS",
        "UCOBANK.NS", "SUMICHEM.NS", "NLCINDIA.NS", "RPOWER.NS", "LAURUSLABS.NS", "SYNGENE.NS", "DELHIVERY.NS", "JUBLINGREA.NS", "TMB.NS", "KARURVYSYA.NS"
    ]

    if st.button("Batch Process Top 200 Indian Stocks"):
        batch_results = []
        for ticker in top_200_indian_stocks:
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
            combined_df = combined_df.sort_index(ascending=False)
            st.success(f"Processed {len(batch_results)} Indian stocks with qualifying signals.")
            st.dataframe(combined_df)
        else:
            st.warning("No Indian stocks met the signal condition.")

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

# Backtest Strategy Page
elif page == "Backtest Strategy":
    st.title("🔬 Backtest Strategy")

    with st.form("backtest_form"):
        ticker = st.session_state.get('ticker', '')
        stock_name = st.text_input("Enter Stock Name for Backtest:", ticker).upper()
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
        end_date = st.date_input("End Date", value=datetime.today())
        strategy = st.selectbox(
            "Strategy",
            ["Mean Reversion", "Mean Reversion Fixed Risk Reward"]
        )
        run_bt = st.form_submit_button("Run Backtest")

    stock_data_dict = st.session_state.get('stock_data_dict', {})
    if run_bt:
        if stock_name in stock_data_dict:
            df = stock_data_dict[stock_name]
            # Remove timezone for comparison
            if df.index.tz is not None:
                df = df.copy()
                df.index = df.index.tz_convert(None)
            # Filter by date
            df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
            st.write(f"Backtesting {strategy} on {stock_name} from {start_date} to {end_date}")

            if strategy == "Mean Reversion":
                trades_df, summary = backtest_mean_reversion(df, mean_col='m_half_51', tolerance=0.01)
            elif strategy == "Mean Reversion Fixed Risk Reward":
                trades_df, summary = backtest_mean_reversion_tp_sl(df, take_profit=0.04, stop_loss=0.07)
            else:
                trades_df, summary = pd.DataFrame(), {}

            st.subheader("Performance Summary")
            st.table(pd.DataFrame(summary, index=["Value"]).T)
            st.subheader("Trade Log")
            if not trades_df.empty:
                # Only convert if not already datetime, otherwise this is a no-op
                trades_df['Entry_Date'] = pd.to_datetime(trades_df['Entry_Date'])
                trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date'])
                display_cols = [
                    'Entry_Date', 'Entry_Price', 'Exit_Date', 'Exit_Price',
                    'Return', 'Exit_Reason'
                ]
                display_cols = [col for col in display_cols if col in trades_df.columns]
                trades_df = trades_df[display_cols]
                st.dataframe(trades_df)
            else:
                st.info("No trades found for this strategy.")
        else:
            st.warning("Please process data for this stock first.")

# Paper Trading Page
elif page == "Paper Trading":
    st.title("📝 Paper Trading")

    trader = PaperTrader()

    st.subheader("Add New Trade")
    with st.form("add_trade_form"):
        symbol = st.text_input("Ticker (Symbol)").upper()
        side = st.selectbox("Side", ["BUY", "SELL"])
        entry_date = st.date_input("Entry Date", value=datetime.today())
        entry_price = st.number_input("Entry Price", min_value=0.0, format="%.4f")
        quantity = st.number_input("Quantity", min_value=1, step=1)
        submit_trade = st.form_submit_button("Add Trade")

    if submit_trade:
        entry_datetime = datetime.combine(entry_date, datetime.now().time())
        trader.take_trade(symbol, side, entry_price, quantity)
        st.success(f"Trade added: {symbol} {side} {quantity} @ {entry_price}")

    st.subheader("Open Trades")
    open_trades = trader.get_open_trades()
    st.dataframe(open_trades)

    # --- Close Trade Feature ---
    st.subheader("Close Trade")
    open_trade_ids = open_trades['Trade_ID'].tolist() if not open_trades.empty else []
    if open_trade_ids:
        with st.form("close_trade_form"):
            close_id = st.selectbox("Select Trade ID to Close", open_trade_ids)
            exit_price = st.number_input("Exit Price", min_value=0.0, format="%.4f", key="exit_price")
            submit_close = st.form_submit_button("Close Trade")
        if submit_close:
            trader.close_trade(close_id, exit_price)
            st.success(f"Trade {close_id} closed at {exit_price}")

    # --- Edit Trade Feature ---
    st.subheader("Edit Open Trade")
    if open_trade_ids:
        with st.form("edit_trade_form"):
            edit_id = st.selectbox("Select Trade ID to Edit", open_trade_ids, key="edit_id")
            new_entry_price = st.number_input("New Entry Price", min_value=0.0, format="%.4f", key="edit_entry_price")
            new_quantity = st.number_input("New Quantity", min_value=1, step=1, key="edit_quantity")
            submit_edit = st.form_submit_button("Edit Trade")
        if submit_edit:
            with trader._get_connection() as conn:
                conn.execute(
                    "UPDATE trades SET Entry_Price = ?, Quantity = ? WHERE Trade_ID = ? AND Status = 'OPEN'",
                    (new_entry_price, new_quantity, edit_id)
                )
            st.success(f"Trade {edit_id} updated: Entry Price={new_entry_price}, Quantity={new_quantity}")

    st.subheader("Trade Log")
    trade_log = trader.get_trade_log()
    st.dataframe(trade_log)
