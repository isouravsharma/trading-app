import pandas as pd

def backtest_mean_reversion(df, mean_col='m_half_51', tolerance=0.01):
    """
    Enter on every 'BELOW 2 SD' or 'BELOW 3 SD' (even if overlapping).
    Exit each trade when Close is within `tolerance` of mean at entry.
    Returns trades DataFrame and summary stats.
    """
    trades = []
    open_trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        idx = df.index[i]
        # Entry: open a new trade for each signal
        if row['Label'] in ['BELOW 2 SD', 'BELOW 3 SD']:
            open_trades.append({
                'Entry_Date': idx,
                'Entry_Price': row['Close'],
                'Entry_Mean': row[mean_col],
                'Entry_Index': i
            })
        # Check all open trades for exit
        still_open = []
        for trade in open_trades:
            # Flexible exit: close if Close is within tolerance of Entry_Mean
            if abs(row['Close'] - trade['Entry_Mean']) / trade['Entry_Mean'] <= tolerance:
                trades.append({
                    'Entry_Date': trade['Entry_Date'],
                    'Entry_Price': trade['Entry_Price'],
                    'Exit_Date': idx,
                    'Exit_Price': row['Close'],
                    'Return': (row['Close'] - trade['Entry_Price']) / trade['Entry_Price']
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
    else:
        win_rate = avg_return = total_return = 0

    summary = {
        'Total Trades': len(trades_df),
        'Win Rate': win_rate,
        'Average Trade Return': avg_return,
        'Total Return': total_return
    }
    return trades_df, summary