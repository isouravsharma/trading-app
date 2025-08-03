import pandas as pd
import numpy as np

def backtest_mean_reversion(
    df, mean_col='m_half_51', tolerance=0.01, stop_loss=0.10,
    slippage=0.001, max_holding=30, use_next_open=False
):
    """
    Mean reversion backtest with slippage, max holding, and risk metrics.
    Entry: 'BELOW 2 SD' or 'BELOW 3 SD'.
    Exit: within tolerance of mean, stop loss, or max holding period.
    """
    trades = []
    open_trades = []

    df = df.copy()
    df = df.reset_index()
    if 'datetime' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'datetime'})

    for i in range(len(df)):
        row = df.iloc[i]
        entry_date = row['datetime']
        # Entry: open a new trade for each signal
        if row.get('Label', None) in ['BELOW 2 SD', 'BELOW 3 SD']:
            entry_price = df.iloc[i+1]['Open'] if (use_next_open and i+1 < len(df)) else row['Close']
            open_trades.append({
                'Entry_Date': entry_date,
                'Entry_Price': entry_price,
                'Entry_Mean': row.get(mean_col, np.nan),
                'Entry_Index': i
            })
        # Check all open trades for exit
        still_open = []
        for trade in open_trades:
            holding = i - trade['Entry_Index']
            price = df.iloc[i+1]['Open'] if (use_next_open and i+1 < len(df)) else row['Close']
            exit_mean = abs(price - trade['Entry_Mean']) / trade['Entry_Mean'] <= tolerance
            exit_stop = (price - trade['Entry_Price']) / trade['Entry_Price'] <= -stop_loss
            exit_time = holding >= max_holding
            if exit_mean or exit_stop or exit_time:
                trade_return = (price - trade['Entry_Price']) / trade['Entry_Price'] - 2 * slippage
                trades.append({
                    'Entry_Date': trade['Entry_Date'],
                    'Entry_Price': trade['Entry_Price'],
                    'Exit_Date': row['datetime'],
                    'Exit_Price': price,
                    'Return': trade_return,
                    'Exit_Reason': (
                        'Mean' if exit_mean else
                        'StopLoss' if exit_stop else
                        'MaxHolding'
                    )
                })
            else:
                still_open.append(trade)
        open_trades = still_open

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['Cumulative_Return'] = (1 + trades_df['Return']).cumprod()
        win_rate = (trades_df['Return'] > 0).mean()
        avg_return = trades_df['Return'].mean()
        total_return = trades_df['Cumulative_Return'].iloc[-1] - 1
        sharpe = trades_df['Return'].mean() / trades_df['Return'].std() * np.sqrt(252) if trades_df['Return'].std() > 0 else 0
        max_dd = (trades_df['Cumulative_Return'].cummax() - trades_df['Cumulative_Return']).max()
    else:
        win_rate = avg_return = total_return = sharpe = max_dd = 0

    summary = {
        'Total Trades': len(trades_df),
        'Win Rate': win_rate,
        'Average Trade Return': avg_return,
        'Total Return': total_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    }
    return trades_df, summary


def backtest_mean_reversion_tp_sl(
    df, take_profit=0.04, stop_loss=0.07,
    slippage=0.001, max_holding=15, use_next_open=False
):
    """
    Mean reversion with fixed risk/reward, slippage, max holding, and risk metrics.
    Entry: 'BELOW 2 SD' or 'BELOW 3 SD'.
    Exit: 4% gain, 7% loss, or max holding period.
    """
    trades = []
    open_trades = []

    df = df.copy()
    df = df.reset_index()
    if 'datetime' not in df.columns:
        df = df.rename(columns={df.columns[0]: 'datetime'})

    for i in range(len(df)):
        row = df.iloc[i]
        entry_date = row['datetime']
        # Entry: open a new trade for each signal
        if row.get('Label', None) in ['BELOW 2 SD', 'BELOW 3 SD']:
            entry_price = df.iloc[i+1]['Open'] if (use_next_open and i+1 < len(df)) else row['Close']
            open_trades.append({
                'Entry_Date': entry_date,
                'Entry_Price': entry_price,
                'Entry_Index': i
            })
        # Check all open trades for exit
        still_open = []
        for trade in open_trades:
            holding = i - trade['Entry_Index']
            price = df.iloc[i+1]['Open'] if (use_next_open and i+1 < len(df)) else row['Close']
            gain = (price - trade['Entry_Price']) / trade['Entry_Price'] - 2 * slippage
            exit_tp = gain >= take_profit
            exit_sl = gain <= -stop_loss
            exit_time = holding >= max_holding
            if exit_tp or exit_sl or exit_time:
                trades.append({
                    'Entry_Date': trade['Entry_Date'],
                    'Entry_Price': trade['Entry_Price'],
                    'Exit_Date': row['datetime'],
                    'Exit_Price': price,
                    'Return': gain,
                    'Exit_Reason': (
                        'TakeProfit' if exit_tp else
                        'StopLoss' if exit_sl else
                        'MaxHolding'
                    )
                })
            else:
                still_open.append(trade)
        open_trades = still_open

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['Cumulative_Return'] = (1 + trades_df['Return']).cumprod()
        win_rate = (trades_df['Return'] > 0).mean()
        avg_return = trades_df['Return'].mean()
        total_return = trades_df['Cumulative_Return'].iloc[-1] - 1
        sharpe = trades_df['Return'].mean() / trades_df['Return'].std() * np.sqrt(252) if trades_df['Return'].std() > 0 else 0
        max_dd = (trades_df['Cumulative_Return'].cummax() - trades_df['Cumulative_Return']).max()
    else:
        win_rate = avg_return = total_return = sharpe = max_dd = 0

    summary = {
        'Total Trades': len(trades_df),
        'Win Rate': win_rate,
        'Average Trade Return': avg_return,
        'Total Return': total_return,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd
    }
    return trades_df, summary