import numpy as np
import pandas as pd
import datetime as dt
from renko import Renko
from scipy.signal import argrelextrema
from scipy.stats import trim_mean

class MarketDataProcessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self._validate()

    def _validate(self):
        """Enhanced data validation"""
        required_cols = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not required_cols.issubset(self.data.columns):
            raise ValueError(f"Data must contain {required_cols}")
        
        # Check for NaN values
        if self.data[list(required_cols)].isnull().any().any():
            raise ValueError("Data contains NaN values in required columns")
        
        # Verify price integrity
        if not (self.data['High'] >= self.data['Low']).all():
            raise ValueError("High prices must be >= Low prices")
        
        # Check for sufficient data
        if len(self.data) < 100:  # Minimum required for calculations
            raise ValueError("Insufficient data points. Need at least 100 bars.")

    def add_rsi(self, n=14):
        delta = self.data['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/n, min_periods=n).mean()
        avg_loss = loss.ewm(alpha=1/n, min_periods=n).mean()
        rs = avg_gain / avg_loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        return self

    def add_macd(self, a=12, b=26, c=9):
        self.data['ma_fast'] = self.data['Close'].ewm(span=a, min_periods=a).mean()
        self.data['ma_slow'] = self.data['Close'].ewm(span=b, min_periods=b).mean()
        self.data['MACD'] = self.data['ma_fast'] - self.data['ma_slow']
        self.data['SIGNAL'] = self.data['MACD'].ewm(span=c, min_periods=c).mean()
        return self

    def add_atr(self, n=14):
        hl = self.data['High'] - self.data['Low']
        hp = (self.data['High'] - self.data['Close'].shift()).abs()
        lp = (self.data['Low'] - self.data['Close'].shift()).abs()
        tr = pd.concat([hl, hp, lp], axis=1).max(axis=1)
        self.data['ATR'] = tr.ewm(span=n, min_periods=n).mean()
        return self

    # def add_adx(self, n=14):
    #     self.add_atr(n)
    #     up_move = self.data['High'].diff()
    #     down_move = self.data['Low'].shift() - self.data['Low']
    #     plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    #     minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    #     plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/n, min_periods=n).mean() / self.data['ATR']
    #     minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/n, min_periods=n).mean() / self.data['ATR']
    #     adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).ewm(span=n, min_periods=n).mean()
    #     self.data['+DI'] = plus_di
    #     self.data['-DI'] = minus_di
    #     self.data['ADX'] = adx
    #     return self

    def add_bollinger_bands(self, n=14, trim_ratio=0.15):
        self.data['MB'] = self.data['Close'].rolling(n).apply(lambda x: trim_mean(x, proportiontocut=trim_ratio), raw=True)
        self.data['STD'] = self.data['Close'].rolling(n).std(ddof=0)
        self.data['UB'] = self.data['MB'] + 2 * self.data['STD']
        self.data['LB'] = self.data['MB'] - 2 * self.data['STD']
        self.data['WIDTH'] = self.data['UB'] - self.data['LB']
        return self

    def add_linear_regression_band(self):
        y = self.data['Close'].values
        X = pd.to_datetime(self.data.index).map(dt.datetime.toordinal).values
        intercept, slope = np.polynomial.polynomial.polyfit(X, y, deg=1)
        line = slope * X + intercept
        std = np.std(y - line)
        self.data['+1STD'] = line + std
        self.data['-1STD'] = line - std
        self.data['+2STD'] = line + 2 * std
        self.data['-2STD'] = line - 2 * std
        return self

    def add_anomaly_detection(self, window=16, threshold=2.7):
        self.data['Log_Return'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['Rolling_Mean'] = self.data['Log_Return'].rolling(window).mean()
        self.data['Rolling_STD'] = self.data['Log_Return'].rolling(window).std()
        self.data['Z_Score'] = (self.data['Log_Return'] - self.data['Rolling_Mean']) / self.data['Rolling_STD']
        self.data['Anomaly'] = (self.data['Z_Score'].abs() > threshold)
        return self

    def add_historical_volatility(self, window=14):
        if 'Log_Return' not in self.data:
            self.add_anomaly_detection()
        self.data['Volatility'] = self.data['Log_Return'].rolling(window).std() * np.sqrt(252)
        return self

    def get_support_resistance(self, order=14):
        support_idx = argrelextrema(self.data['Low'].values, np.less, order=order)[0]
        resistance_indices = argrelextrema(self.data['High'].values, np.greater, order=order)[0]
        support_levels = self.data['Low'].iloc[support_idx]
        resistance_levels = self.data['High'].iloc[resistance_indices]
        # Get the latest 4 by index (most recent) and round to 2 decimals
        latest_support = support_levels.iloc[-4:] if len(support_levels) >= 4 else support_levels
        latest_resistance = resistance_levels.iloc[-4:] if len(resistance_levels) >= 4 else resistance_levels
        return np.round(latest_support, 2).tolist(), np.round(latest_resistance, 2).tolist()

    def add_vwap(self):
        self.data['TP'] = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        self.data['cum_vol'] = self.data['Volume'].cumsum()
        self.data['cum_pv'] = (self.data['TP'] * self.data['Volume']).cumsum()
        self.data['VWAP'] = self.data['cum_pv'] / self.data['cum_vol']
        return self

    # def get_renko(self, atr_period=120):
    #     self.add_atr(atr_period)
    #     df = self.data.copy().reset_index()
    #     df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    #     df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    #     renko_obj = Renko(df)
    #     renko_obj.brick_size = max(0.5, round(self.data['ATR'].iloc[-1], 0))
    #     renko_df = renko_obj.get_ohlc_data()
    #     renko_df['bar_num'] = np.where(renko_df['uptrend'], 1, -1)
    #     for i in range(1, len(renko_df['bar_num'])):
    #         if renko_df['bar_num'][i] * renko_df['bar_num'][i - 1] > 0:
    #             renko_df['bar_num'][i] += renko_df['bar_num'][i - 1]
    #     renko_df.drop_duplicates(subset='date', keep='last', inplace=True)
    #     return renko_df

    def get_data(self):
        return self.data.copy()

    def add_half_life_signals(self, halflife=51, column='Close'):
        """Vectorized half-life signal generation"""
        m_col = f'm_half_{halflife}'
        dist_col = 'Distance_From_Mean'
        
        # Vectorized calculations
        self.data[m_col] = self.data[column].ewm(halflife=halflife).mean()
        self.data[dist_col] = self.data[column] - self.data[m_col]
        
        mean = self.data[dist_col].mean()
        std = self.data[dist_col].std()
        
        # Vectorized categorization
        conditions = [
            self.data[dist_col] >= mean + 3 * std,
            self.data[dist_col] <= mean - 3 * std,
            self.data[dist_col] >= mean + 2 * std,
            self.data[dist_col] <= mean - 2 * std,
            self.data[dist_col] >= mean + std,
            self.data[dist_col] <= mean - std
        ]
        
        choices_label = [
            'ABOVE 3 SD', 'BELOW 3 SD',
            'ABOVE 2 SD', 'BELOW 2 SD',
            'ABOVE 1 SD', 'BELOW 1 SD'
        ]
        
        choices_level = [3, -3, 2, -2, 1, -1]
        
        self.data['Label'] = np.select(conditions, choices_label, default='AT MEAN')
        self.data['SD_Level'] = np.select(conditions, choices_level, default=0)
        
        return self
    
    def cummulative_returns(self, window=30):
        """
        Calculate daily returns, log returns, cumulative returns, and rolling stats on returns.
        """
        self.data['d_rtn'] = self.data['Close'].pct_change()
        self.data['log_rtn'] = np.log(self.data['Close'] / self.data['Close'].shift(1))
        self.data['cum_rtn'] = (1 + self.data['d_rtn']).cumprod()
        # Rolling stats on daily returns (quant standard)
        self.data['r_mean'] = self.data['d_rtn'].rolling(window).mean()
        self.data['r_std'] = self.data['d_rtn'].rolling(window).std()
        # Optional: create upper/lower bands for returns
        self.data['r_upper'] = self.data['r_mean'] + 2 * self.data['r_std']
        self.data['r_lower'] = self.data['r_mean'] - 2 * self.data['r_std']
        return self
    
    def add_entry_price(
        self, halflife=51, rsi_threshold=30, vol_window=20, vol_percentile=0.7,
        ma_window=50, slope_thresh=0.05, min_volume=0
    ):
        """
        For each entry (Label is 'BELOW 2 SD' or 'BELOW 3 SD'), 
        record Entry_Price if RSI < rsi_threshold, Volume is above vol_percentile of last vol_window bars,
        and the long MA slope is flat (sideways regime).
        Also adds Stop_Loss (10% below Entry_Price) and Potential_Profit (mean value at entry).
        """
        label_col = 'Label'
        close_col = 'Close'
        rsi_col = 'RSI'
        vol_col = 'Volume'
        mean_col = f'm_half_{halflife}'

        df = self.data
        df['Entry_Price'] = np.nan
        df['Exit_Price'] = np.nan
        df['Stop_Loss'] = np.nan
        df['Potential_Profit'] = np.nan

        # Calculate rolling volume threshold
        df['Vol_Thresh'] = df[vol_col].rolling(vol_window, min_periods=1).quantile(vol_percentile)

        # Calculate long moving average and its slope
        df['MA_Long'] = df[close_col].rolling(ma_window, min_periods=1).mean()
        df['MA_Slope'] = df['MA_Long'].diff() / df['MA_Long'].shift(1)

        for i, row in df.iterrows():
            if (
                row[label_col] in ['BELOW 2 SD', 'BELOW 3 SD']
                and row[rsi_col] < rsi_threshold
                and row[vol_col] > row['Vol_Thresh']
                and abs(row['MA_Slope']) < slope_thresh
                and row[vol_col] > min_volume
            ):
                entry_price = row[close_col]
                stop_loss = entry_price * 0.90  # 10% below entry
                potential_profit = row[mean_col]  # mean value at entry
                df.at[i, 'Entry_Price'] = entry_price
                df.at[i, 'Stop_Loss'] = stop_loss
                df.at[i, 'Potential_Profit'] = potential_profit

        # Clean up temp columns if needed
        df.drop(['Vol_Thresh', 'MA_Long', 'MA_Slope'], axis=1, inplace=True, errors='ignore')
        self.data = df
        return self
    
    def add_position_sizing(self, risk_per_trade=0.02, atr_multiple=2):
        """
        Calculate position sizes based on volatility (ATR)
        risk_per_trade: fraction of capital to risk per trade (e.g., 0.02 = 2%)
        atr_multiple: multiple of ATR for stop loss
        """
        self.add_atr()  # Ensure ATR is calculated
        
        def calculate_size(row, capital=100000):
            stop_distance = row['ATR'] * atr_multiple
            if stop_distance == 0:
                return 0
            dollar_risk = capital * risk_per_trade
            return np.floor(dollar_risk / stop_distance)
        
        self.data['Position_Size'] = self.data.apply(calculate_size, axis=1)
        return self
