"""
infrastructure/backtester/visualizer.py
-----------------------------------------
Ported from Epsilon Fund / infrastructure / backtester / visualizer.py.
All existing functions preserved exactly (no logic changes, no import changes).
Added: plot_direction_split() at the bottom.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Ported as-is from Epsilon Fund
# ---------------------------------------------------------------------------

def plot_equity_curve(equity_curve, benchmark_equity=None, title="Portfolio Equity Curve"):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Strategy',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>Strategy</b><br>Date: %{x}<br>Equity: %{y:.4f}<extra></extra>'
    ))

    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index,
            y=benchmark_equity.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Equity: %{y:.4f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1E293B')),
        xaxis_title="Date",
        yaxis_title="Equity (Normalized to 1.0)",
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E2E8F0', borderwidth=1
        ),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#E2E8F0'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#E2E8F0'),
    )

    return fig


def plot_drawdown(equity_curve, title="Portfolio Drawdown"):

    running_max = equity_curve.cummax()
    drawdown    = (equity_curve - running_max) / running_max

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values * 100,
        mode='lines',
        name='Drawdown',
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.3)',
        line=dict(color='#ef4444', width=1),
        hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='#1E293B')),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        template='plotly_white',
        height=300,
        showlegend=False,
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='#E2E8F0'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='#E2E8F0'),
    )

    return fig


def format_metrics_annotation(metrics):

    main_text = f"""
<b>Portfolio Performance</b><br>
Total Return: <b>{metrics['total_return']*100:.2f}%</b><br>
Sharpe Ratio: <b>{metrics['sharpe_ratio']:.2f}</b><br>
Max Drawdown: <b>{metrics['max_drawdown']*100:.2f}%</b><br>
Calmar Ratio: <b>{metrics['calmar_ratio']:.2f}</b><br>
Cost: <b>{metrics['cost_percent']*100:.2f}%</b>
"""

    yearly_returns_text = "<b>Yearly Returns:</b><br>"
    for year, ret in sorted(metrics['yearly_returns'].items()):
        yearly_returns_text += f"{year}: <b>{ret*100:.2f}%</b><br>"

    yearly_sharpe_text = "<b>Yearly Sharpe Ratios:</b><br>"
    for year, sharpe in sorted(metrics['yearly_sharpe'].items()):
        yearly_sharpe_text += f"{year}: <b>{sharpe:.2f}</b><br>"

    trade_text = f"""
<b>Strategy Statistics</b><br>
Total Trades: <b>{metrics['num_trades']}</b><br>
Win Rate: <b>{metrics['win_rate']*100:.2f}%</b><br>
Profit Factor: <b>{metrics['profit_factor']:.2f}</b><br>
Avg Win/Loss: <b>{metrics['avg_win_loss_ratio']:.2f}</b>
"""

    return main_text, yearly_returns_text, yearly_sharpe_text, trade_text


def plot_results(metrics, benchmark_data=None, show=True, save_html=None):

    benchmark_equity = None
    if benchmark_data is not None:
        benchmark_returns = benchmark_data['Close'].pct_change()
        benchmark_equity  = (1 + benchmark_returns).cumprod()
        benchmark_equity.fillna(1.0, inplace=True)

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            "Portfolio Equity Curve (Net of Costs)",
            "Portfolio Drawdown (Net of Costs)",
        ),
        vertical_spacing=0.08,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    equity_curve = metrics['equity_curve']

    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode='lines', name='Strategy',
        line=dict(color='#3b82f6', width=2),
        hovertemplate='<b>Strategy</b><br>Date: %{x}<br>Equity: %{y:.4f}<extra></extra>'
    ), row=1, col=1)

    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index, y=benchmark_equity.values,
            mode='lines', name='Benchmark',
            line=dict(color='#f59e0b', width=2),
            hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Equity: %{y:.4f}<extra></extra>'
        ), row=1, col=1)

    running_max = equity_curve.cummax()
    drawdown    = (equity_curve - running_max) / running_max

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        mode='lines', name='Drawdown',
        fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)',
        line=dict(color='#ef4444', width=1),
        hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8",
                  line_width=1, row=2, col=1)

    main_text, yearly_returns_text, yearly_sharpe_text, trade_text = \
        format_metrics_annotation(metrics)

    for text, x, y in [
        (main_text,          0.01, 0.98),
        (yearly_returns_text, 0.01, 0.65),
        (yearly_sharpe_text,  0.20, 0.65),
        (trade_text,          0.01, 0.35),
    ]:
        fig.add_annotation(
            xref="x domain", yref="y domain",
            x=x, y=y, xanchor='left', yanchor='top',
            text=text, showarrow=False,
            font=dict(size=10, family="monospace"),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#cbd5e1", borderwidth=1, borderpad=8,
            row=1, col=1,
        )

    max_dd_year_text = "<b>Max Drawdown:</b><br>"
    for year, dd in sorted(metrics['yearly_max_drawdown'].items()):
        max_dd_year_text += f"{year}: <b>{dd*100:.2f}%</b><br>"

    fig.add_annotation(
        xref="x2 domain", yref="y2 domain",
        x=0.99, y=0.98, xanchor='right', yanchor='top',
        text=max_dd_year_text, showarrow=False,
        font=dict(size=9, family="monospace"),
        align="left",
        bgcolor="rgba(255, 255, 255, 0.9)",
        bordercolor="#cbd5e1", borderwidth=1, borderpad=8,
        row=2, col=1,
    )

    fig.update_layout(
        height=900, showlegend=True, hovermode='x unified',
        template='plotly_white',
        title=dict(text="<b>Backtesting Results</b>",
                   font=dict(size=22, color='#1E293B'), x=0.5, xanchor='center'),
        legend=dict(
            yanchor="top", y=0.99, xanchor="right", x=0.99,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#cbd5e1', borderwidth=1,
        ),
    )

    fig.update_xaxes(title_text="Date", showgrid=True, gridwidth=1, gridcolor='#f1f5f9', row=1, col=1)
    fig.update_xaxes(title_text="Date", showgrid=True, gridwidth=1, gridcolor='#f1f5f9', row=2, col=1)
    fig.update_yaxes(title_text="Equity", showgrid=True, gridwidth=1, gridcolor='#f1f5f9', row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", showgrid=True, gridwidth=1, gridcolor='#f1f5f9', row=2, col=1)

    if save_html:
        fig.write_html(save_html)
        print(f"OK Saved interactive chart to: {save_html}")

    if show:
        fig.show()

    return fig


def plot_trades_on_price(data, trades_df, show=True, save_html=None):

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data.index, y=data['Close'],
        mode='lines', name='Price',
        line=dict(color='#64748b', width=1.5),
        hovertemplate='<b>Price</b><br>Date: %{x}<br>Close: $%{y:.2f}<extra></extra>'
    ))

    if len(trades_df) > 0:
        long_entries = trades_df[trades_df['direction'] == 'Long']
        fig.add_trace(go.Scatter(
            x=long_entries['entry_time'], y=long_entries['entry_price'],
            mode='markers', name='Long Entry',
            marker=dict(symbol='triangle-up', size=10, color='#22c55e'),
            hovertemplate='<b>Long Entry</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))
        fig.add_trace(go.Scatter(
            x=long_entries['exit_time'], y=long_entries['exit_price'],
            mode='markers', name='Long Exit',
            marker=dict(symbol='triangle-down', size=10, color='#16a34a'),
            hovertemplate='<b>Long Exit</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        short_entries = trades_df[trades_df['direction'] == 'Short']
        if len(short_entries) > 0:
            fig.add_trace(go.Scatter(
                x=short_entries['entry_time'], y=short_entries['entry_price'],
                mode='markers', name='Short Entry',
                marker=dict(symbol='triangle-down', size=10, color='#ef4444'),
                hovertemplate='<b>Short Entry</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=short_entries['exit_time'], y=short_entries['exit_price'],
                mode='markers', name='Short Exit',
                marker=dict(symbol='triangle-up', size=10, color='#dc2626'),
                hovertemplate='<b>Short Exit</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ))

    fig.update_layout(
        title=dict(text="<b>Price Chart with Trades</b>", font=dict(size=18, color='#1E293B')),
        xaxis_title="Date", yaxis_title="Price",
        hovermode='x unified', template='plotly_white', height=600, showlegend=True,
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor='rgba(255,255,255,0.9)', bordercolor='#cbd5e1', borderwidth=1,
        ),
    )

    if save_html:
        fig.write_html(save_html)
        print(f"OK Saved trade chart to: {save_html}")

    if show:
        fig.show()

    return fig


# ---------------------------------------------------------------------------
# New: direction split chart
# ---------------------------------------------------------------------------

_BLUE   = '#3b82f6'
_GREEN  = '#22c55e'
_AMBER  = '#f59e0b'
_RED    = '#ef4444'
_GRID   = '#f1f5f9'
_BORDER = '#cbd5e1'
_BG_BOX = 'rgba(255, 255, 255, 0.9)'


def plot_direction_split(direction_split_dict, show=True, save_html=None):
    """
    Three-column bar chart comparing key metrics across Combined, Long-only,
    and Short-only directions.

    Parameters
    ----------
    direction_split_dict : dict
        The 'direction_split' dict from engine.backtest() results.
        Keys: 'combined', 'long_only', 'short_only'.
        Values are metrics dicts or None (None = no trades in that direction).
    show : bool
        Display the chart interactively. Default True.
    save_html : str, optional
        Save the chart to this HTML path.

    Returns
    -------
    plotly.graph_objects.Figure

    Example
    -------
    >>> results = backtest(ctx, strategy_fn, params)
    >>> plot_direction_split(results['direction_split'])
    """
    metrics_to_show = [
        ("total_return",      "Total Return",   True),   # (key, label, is_pct)
        ("sharpe_ratio",      "Sharpe Ratio",   False),
        ("max_drawdown",      "Max Drawdown",   True),
        ("win_rate",          "Win Rate",        True),
        ("num_trades",        "Num Trades",      False),
        ("profit_factor",     "Profit Factor",  False),
    ]

    directions = ["combined", "long_only", "short_only"]
    labels     = ["Combined", "Long Only", "Short Only"]
    colors     = [_BLUE, _GREEN, _RED]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[m[1] for m in metrics_to_show],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    for idx, (metric_key, metric_label, is_pct) in enumerate(metrics_to_show):
        row = idx // 3 + 1
        col = idx %  3 + 1

        values     = []
        bar_labels = []
        bar_colors = []

        for direction, label, color in zip(directions, labels, colors):
            m = direction_split_dict.get(direction)
            if m is None:
                values.append(0.0)
                bar_labels.append(f"{label}<br><i>no trades</i>")
            else:
                val = m.get(metric_key, 0.0)
                if val is None:
                    val = 0.0
                values.append(float(val) * (100 if is_pct else 1))
                bar_labels.append(label)
            bar_colors.append(color)

        fig.add_trace(
            go.Bar(
                x=bar_labels,
                y=values,
                marker_color=bar_colors,
                opacity=0.85,
                showlegend=False,
                hovertemplate=(
                    f"<b>{metric_label}</b><br>"
                    "%{x}<br>"
                    f"%{{y:.2f}}{'%' if is_pct else ''}"
                    "<extra></extra>"
                ),
            ),
            row=row, col=col,
        )

    fig.update_layout(
        height=600,
        template="plotly_white",
        title=dict(
            text="<b>Direction Split -- Combined vs Long-Only vs Short-Only</b>",
            font=dict(size=20, color="#1E293B"),
            x=0.5, xanchor="center",
        ),
    )

    for r in range(1, 3):
        for c in range(1, 4):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=_GRID, row=r, col=c)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=_GRID, row=r, col=c)

    if save_html:
        fig.write_html(save_html)
        print(f"OK Saved direction split chart to: {save_html}")

    if show:
        fig.show()

    return fig
