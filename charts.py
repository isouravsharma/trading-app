import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_stock_signals(data, support_levels, resistance_levels, stock_name):
    # Round the 'Close' column for all stocks to 2 decimals
    for key in data:
        if 'Close' in data[key].columns:
            data[key]['Close'] = data[key]['Close'].round(2)

    fig = go.Figure()

    # Add line traces for each stock in data
    for key in data:
        df = data[key]
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name=f'{key} Close'
        ))
        for col, label in zip(['MB', 'UB', 'LB', 'm_half_51'], ['MB', 'UB', 'LB', 'm_half_50']):
            if col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode='lines',
                    name=f'{key} {label}'
                ))

    # Add support lines
    for y_val in support_levels:
        if y_val is not None:
            fig.add_hline(
                y=round(y_val, 2),
                line=dict(color="green", dash="dash"),
                annotation_text=f"Support ({y_val:.2f})",
                annotation_position="bottom left"
            )

    # Add resistance lines
    for y_val in resistance_levels:
        if y_val is not None:
            fig.add_hline(
                y=round(y_val, 2),
                line=dict(color="red", dash="dash"),
                annotation_text=f"Resistance ({y_val:.2f})",
                annotation_position="top left"
            )

    # Add labeled points for the specified stock_name
    labels = ['ABOVE 2 SD', 'ABOVE 3 SD', 'BELOW 2 SD', 'BELOW 3 SD']
    marker_styles = {
        'ABOVE 2 SD': dict(color='red', size=10, symbol='triangle-down'),
        'ABOVE 3 SD': dict(color='red', size=20, symbol='triangle-down'),
        # 'BELOW 1 SD': dict(color='green', size=10, symbol='circle'),
        'BELOW 2 SD': dict(color='green', size=10, symbol='triangle-up'),
        'BELOW 3 SD': dict(color='green', size=20, symbol='triangle-up'),
    }

    df = data[stock_name]
    if 'Label' in df.columns:
        for label in labels:
            subset = df[df['Label'] == label]
            if not subset.empty:
                fig.add_trace(go.Scatter(
                    x=subset.index,
                    y=subset['Close'],
                    mode='markers',
                    name=label,
                    marker=marker_styles[label]
                ))

    # Add exit price markers for all stocks
    for key in data:
        df = data[key]
        if 'Exit_Price_Mean' in df.columns:
            exit_rows = df[df['Exit_Price_Mean'].notnull()]
            if not exit_rows.empty:
                fig.add_trace(go.Scatter(
                    x=exit_rows.index,
                    y=exit_rows['Exit_Price_Mean'],
                    mode='markers',
                    name=f'{key} -  Exit_Price_Mean',
                    marker=dict(color='blue', size=10, symbol='circle')
                ))

    for key in data:
        df = data[key]
        if 'Exit_Price_ReducedDist' in df.columns:
            exit_rows = df[df['Exit_Price_ReducedDist'].notnull()]
            if not exit_rows.empty:
                fig.add_trace(go.Scatter(
                    x=exit_rows.index,
                    y=exit_rows['Exit_Price_ReducedDist'],
                    mode='markers',
                    name=f'{key} -  Exit_Price_ReducedDist',
                    marker=dict(color='yellow', size=10, symbol='circle')
            ))

    # Customize layout
    fig.update_layout(
        title=f"{stock_name} Signal Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        autosize=False,
        width=1100,
        height=800,
        template='plotly_white',
        showlegend=True
    )
    return fig

def plot_stock_subplots(data):
    """
    Plots a 4-row subplot chart for Close, RSI, ATR, and Volume.
    """
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=("Close", "RSI", "ATR", "Volume"),
        shared_xaxes=True,
        vertical_spacing=0.03
    )

    # Close, MB, UB, LB
    for key in data:
        if 'Close' in data[key].columns:
            fig.add_trace(go.Scatter(
                x=data[key].index,
                y=data[key]['Close'],
                mode='lines',
                name=f'{key} Close'
            ), row=1, col=1)
        for col in ['MB', 'UB', 'LB']:
            if col in data[key].columns:
                fig.add_trace(go.Scatter(
                    x=data[key].index,
                    y=data[key][col],
                    mode='lines',
                    name=f'{key} {col}'
                ), row=1, col=1)

    # RSI
    for key in data:
        if 'RSI' in data[key].columns:
            fig.add_trace(go.Scatter(
                x=data[key].index,
                y=data[key]['RSI'],
                mode='lines',
                name=f'{key} RSI'
            ), row=2, col=1)

    # ATR
    for key in data:
        if 'ATR' in data[key].columns:
            fig.add_trace(go.Scatter(
                x=data[key].index,
                y=data[key]['ATR'],
                mode='lines',
                name=f'{key} ATR'
            ), row=3, col=1)

    # Volume (as bar)
    for key in data:
        if 'Volume' in data[key].columns:
            fig.add_trace(go.Bar(
                x=data[key].index,
                y=data[key]['Volume'],
                name=f'{key} Volume',
                marker=dict(color='red')
            ), row=4, col=1)

    # Set axis titles for each subplot
    fig.update_yaxes(title_text="Close Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="ATR", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    fig.update_xaxes(title_text="Date", row=4, col=1)

    fig.update_layout(
        title="Stock Analysis Subplots",
        autosize=False,
        width=1100,
        height=1400,
        template='plotly_white',
        showlegend=True
    )
    return fig