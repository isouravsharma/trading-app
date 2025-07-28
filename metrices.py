import numpy as np
import pandas as pd

class PerformanceMetrics:
    def __init__(self, df, periods_per_year=252, risk_free_rate=0.03):
        """
        df: pd.DataFrame with 'Entry_Price' and 'Exit_Price_Mean' columns indexed by date
        periods_per_year: number of trading periods in a year (e.g. 252 for daily)
        risk_free_rate: annualized risk free rate (e.g. 0.03 for 3%)
        """
        self.df = df.copy()
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate
        self.results = {}
        self.trades = self._extract_trades()

    def _extract_trades(self):
        trades = []
        df = self.df

        # Use Entry_Price if present, else Entry_Price
        entry_col = 'Entry_Price' if 'Entry_Price' in df.columns else 'Entry_Price'
        if entry_col not in df.columns or 'Exit_Price_Mean' not in df.columns:
            raise ValueError("DataFrame must contain 'Entry_Price' (or 'Entry_Price') and 'Exit_Price_Mean' columns.")

        entry_rows = df[df[entry_col].notnull()]
        for entry_idx in entry_rows.index:
            entry_price = df.at[entry_idx, entry_col]
            exit_idx = None
            exit_price = None
            for i in range(df.index.get_loc(entry_idx) + 1, len(df)):
                idx = df.index[i]
                if not pd.isna(df.at[idx, 'Exit_Price_Mean']):
                    exit_idx = idx
                    exit_price = df.at[idx, 'Exit_Price_Mean']
                    break
            if exit_idx is not None and exit_price is not None:
                trades.append({
                    'Entry_Date': entry_idx,
                    'Entry_Price': entry_price,
                    'Exit_Date': exit_idx,
                    'Exit_Price': exit_price,
                    'Return': (exit_price - entry_price) / entry_price
                })
        return pd.DataFrame(trades)

    def calculate_equity_curve(self):
        if self.trades.empty:
            self.equity_curve = pd.Series(dtype=float)
            return
        self.trades['Cumulative_Return'] = (1 + self.trades['Return']).cumprod()
        self.equity_curve = self.trades.set_index('Exit_Date')['Cumulative_Return']

    def calculate_CAGR(self):
        if self.trades.empty:
            self.results['CAGR'] = np.nan
            return
        start = pd.to_datetime(self.trades['Entry_Date'].iloc[0])
        end = pd.to_datetime(self.trades['Exit_Date'].iloc[-1])
        years = (end - start).days / 365.25
        ending_value = self.trades['Cumulative_Return'].iloc[-1]
        self.results['CAGR'] = ending_value ** (1 / years) - 1 if years > 0 else np.nan

    def calculate_volatility(self):
        if self.trades.empty:
            self.results['VOL'] = np.nan
            return
        trade_std = self.trades['Return'].std(ddof=1)
        self.results['VOL'] = trade_std * np.sqrt(self.periods_per_year)

    def calculate_sharpe(self):
        if self.trades.empty:
            self.results['SHARPE'] = np.nan
            return
        mean_trade = self.trades['Return'].mean()
        ann_return = mean_trade * self.periods_per_year
        vol = self.results.get('VOL', np.nan)
        if vol == 0 or np.isnan(vol):
            self.results['SHARPE'] = np.nan
        else:
            self.results['SHARPE'] = (ann_return - self.risk_free_rate) / vol

    def calculate_sortino(self):
        if self.trades.empty:
            self.results['SORTINO'] = np.nan
            return
        neg_returns = self.trades['Return'].copy()
        neg_returns[neg_returns > 0] = 0
        downside_std = np.sqrt((neg_returns ** 2).mean() * self.periods_per_year)
        cagr = self.results.get('CAGR', np.nan)
        if downside_std == 0 or np.isnan(downside_std):
            self.results['SORTINO'] = np.nan
        else:
            self.results['SORTINO'] = (cagr - self.risk_free_rate) / downside_std

    def calculate_max_drawdown(self):
        if self.trades.empty:
            self.results['Max_DD'] = np.nan
            return
        self.calculate_equity_curve()
        cum_roll_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve / cum_roll_max) - 1
        max_dd = drawdown.min()
        self.results['Max_DD'] = abs(max_dd)

    def calculate_calmar(self):
        cagr = self.results.get('CAGR', np.nan)
        max_dd = self.results.get('Max_DD', np.nan)
        if max_dd == 0 or np.isnan(max_dd):
            self.results['CALMAR'] = np.nan
        else:
            self.results['CALMAR'] = cagr / max_dd

    def run_all(self):
        self.calculate_equity_curve()
        self.calculate_CAGR()
        self.calculate_volatility()
        self.calculate_sharpe()
        self.calculate_sortino()
        self.calculate_max_drawdown()
        self.calculate_calmar()
        return self.results
