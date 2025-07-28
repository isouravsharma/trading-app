import yfinance as yf

def get_ohlcv(ticker, period, interval, multi_level_index=False):
    try:
        data = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            progress=False
        )
        data.dropna(how='any', inplace=True)
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to fetch data: {e}")
