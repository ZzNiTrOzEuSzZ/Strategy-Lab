"""
infrastructure/backtester/wf_visualizer.py
--------------------------------------------
Ported from Epsilon Fund / infrastructure / walkforward / wf_visualizer.py.
Logic and style preserved exactly. Only change: import paths updated to
reference the new metrics.py location in this package.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── shared style constants (mirrors visualizer.py) ────────────────────────────
_BLUE      = '#3b82f6'
_AMBER     = '#f59e0b'
_GREEN     = '#22c55e'
_RED       = '#ef4444'
_RED_FILL  = 'rgba(239, 68, 68, 0.3)'
_GREY      = '#64748b'
_GRID      = '#f1f5f9'
_BORDER    = '#cbd5e1'
_BG_BOX    = 'rgba(255, 255, 255, 0.9)'
_FONT_MONO = 'monospace'
_TEMPLATE  = 'plotly_white'


# ──────────────────────────────────────────────────────────────────────────────
#  Fold performance bar chart
# ──────────────────────────────────────────────────────────────────────────────

def plot_fold_performance(results_df, show=True, save_html=None):
    """
    Side-by-side IS / OOS bars for return, Sharpe, and max drawdown per fold.

    Parameters
    ----------
    results_df : pd.DataFrame
        The 'results_df' key from Research.walk_forward() output.
    show : bool
    save_html : str, optional
    """
    labels = [f'Fold {r}' for r in results_df['fold']]

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)'),
        vertical_spacing=0.10,
        row_heights=[0.35, 0.35, 0.30],
    )

    for col, name, color in [
        ('train_return', 'IS Return',  _BLUE),
        ('test_return',  'OOS Return', _GREEN),
    ]:
        vals = results_df[col].fillna(0) * 100
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name,
            marker_color=color, opacity=0.85,
            hovertemplate=f'<b>{name}</b><br>%{{x}}<br>%{{y:.2f}}%<extra></extra>',
        ), row=1, col=1)

    for col, name, color in [
        ('train_sharpe', 'IS Sharpe',  _BLUE),
        ('test_sharpe',  'OOS Sharpe', _GREEN),
    ]:
        vals = results_df[col].fillna(0)
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name,
            marker_color=color, opacity=0.85, showlegend=False,
            hovertemplate=f'<b>{name}</b><br>%{{x}}<br>%{{y:.2f}}<extra></extra>',
        ), row=2, col=1)

    for col, name, color in [
        ('train_drawdown', 'IS Drawdown',  _BLUE),
        ('test_drawdown',  'OOS Drawdown', _RED),
    ]:
        vals = results_df[col].fillna(0) * 100
        fig.add_trace(go.Bar(
            x=labels, y=vals, name=name,
            marker_color=color, opacity=0.85, showlegend=False,
            hovertemplate=f'<b>{name}</b><br>%{{x}}<br>%{{y:.2f}}%<extra></extra>',
        ), row=3, col=1)

    fig.update_layout(
        height=900, barmode='group', hovermode='x unified', template=_TEMPLATE,
        title=dict(
            text='<b>Walk-Forward: IS vs OOS Performance by Fold</b>',
            font=dict(size=22, color='#1E293B'), x=0.5, xanchor='center',
        ),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.99,
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1,
        ),
    )

    for row in range(1, 4):
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=_GRID, row=row, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=_GRID, row=row, col=1)

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved fold performance chart -> {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Parameter evolution across folds
# ──────────────────────────────────────────────────────────────────────────────

def plot_parameter_evolution(results_df, param_defs, fixed_params=None,
                             show=True, save_html=None):
    """
    Line chart showing how each free parameter moved across folds.

    Parameters
    ----------
    results_df : pd.DataFrame
    param_defs : dict
        {name: ('int'|'float', lo, hi)} — the strategy's PARAM_SPACE.
    fixed_params : dict, optional
    show : bool
    save_html : str, optional
    """
    if fixed_params is None:
        fixed_params = {}

    free_params = [k for k in param_defs if k not in fixed_params]
    n = len(free_params)
    if n == 0:
        print('No free parameters to plot.')
        return None

    cols      = 2
    rows      = int(np.ceil(n / cols))
    v_spacing = min(0.08, round(0.9 / max(rows - 1, 1), 3))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=free_params,
        vertical_spacing=v_spacing,
        horizontal_spacing=0.10,
    )

    fold_nums = list(results_df['fold'])

    for idx, name in enumerate(free_params):
        row = idx // cols + 1
        col = idx %  cols + 1
        vals = results_df[f'param_{name}'].values

        med = np.median(vals)
        std = np.std(vals)

        # ±1 std band
        fig.add_trace(go.Scatter(
            x=fold_nums + fold_nums[::-1],
            y=list(np.full(len(fold_nums), med + std)) +
              list(np.full(len(fold_nums), med - std)),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.12)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=(idx == 0),
            name='±1 std',
            hoverinfo='skip',
        ), row=row, col=col)

        # Median line
        fig.add_trace(go.Scatter(
            x=fold_nums,
            y=np.full(len(fold_nums), med),
            mode='lines',
            line=dict(color=_AMBER, dash='dash', width=1),
            showlegend=(idx == 0),
            name='Median',
            hoverinfo='skip',
        ), row=row, col=col)

        # Actual value per fold
        fig.add_trace(go.Scatter(
            x=fold_nums, y=vals,
            mode='lines+markers',
            line=dict(color=_BLUE, width=2),
            marker=dict(size=7),
            showlegend=False,
            name=name,
            hovertemplate=f'<b>{name}</b><br>Fold %{{x}}<br>Value: %{{y:.4g}}<extra></extra>',
        ), row=row, col=col)

    fig.update_layout(
        height=max(400, rows * 280), template=_TEMPLATE,
        title=dict(
            text='<b>Walk-Forward: Parameter Evolution Across Folds</b>',
            font=dict(size=22, color='#1E293B'), x=0.5, xanchor='center',
        ),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.99,
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1,
        ),
    )

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved parameter evolution chart -> {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Combined OOS equity curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_oos_equity(oos_metrics, oos_combined_df, fold_boundaries=None,
                   benchmark_data=None, show=True, save_html=None,
                   show_trades=False):
    """
    Full equity + drawdown chart for the stitched OOS period.

    Parameters
    ----------
    oos_metrics : dict
        Backtest metrics dict for the combined OOS period.
    oos_combined_df : pd.DataFrame
        The stitched OOS primary DataFrame.
    fold_boundaries : list of dates, optional
        Dates at which to draw vertical fold-separation lines.
    benchmark_data : pd.DataFrame, optional
        DataFrame with 'close' column for buy-and-hold comparison.
    show : bool
    save_html : str, optional
    show_trades : bool
    """
    equity_curve = oos_metrics['equity_curve']

    benchmark_equity = None
    if benchmark_data is not None:
        col = 'close' if 'close' in benchmark_data.columns else 'Close'
        br  = benchmark_data[col].pct_change()
        benchmark_equity = (1 + br).cumprod().fillna(1.0)
        benchmark_equity = benchmark_equity.reindex(equity_curve.index, method='nearest')
        benchmark_equity = benchmark_equity / benchmark_equity.iloc[0]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            'OOS Equity Curve (Net of Costs)',
            'OOS Drawdown (Net of Costs)',
        ),
        vertical_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=equity_curve.index, y=equity_curve.values,
        mode='lines', name='OOS Strategy',
        line=dict(color=_BLUE, width=2),
        hovertemplate='<b>OOS Strategy</b><br>Date: %{x}<br>Equity: %{y:.4f}<extra></extra>',
    ), row=1, col=1)

    if benchmark_equity is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_equity.index, y=benchmark_equity.values,
            mode='lines', name='Buy and Hold',
            line=dict(color=_AMBER, width=2),
            hovertemplate='<b>Buy and Hold</b><br>Date: %{x}<br>Equity: %{y:.4f}<extra></extra>',
        ), row=1, col=1)

    running_max = equity_curve.cummax()
    drawdown    = (equity_curve - running_max) / running_max

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        mode='lines', name='Drawdown',
        fill='tozeroy', fillcolor=_RED_FILL,
        line=dict(color=_RED, width=1),
        hovertemplate='<b>Drawdown</b><br>Date: %{x}<br>%{y:.2f}%<extra></extra>',
        showlegend=False,
    ), row=2, col=1)

    if show_trades and oos_metrics.get('trades') is not None:
        trades = oos_metrics['trades']
        if len(trades) > 0:
            fig.add_trace(go.Scatter(
                x=trades['entry_time'], y=trades['entry_price'],
                mode='markers', name='Entry',
                marker=dict(symbol='triangle-up', size=9, color=_GREEN),
                hovertemplate='<b>Entry</b><br>%{x}<br>$%{y:.2f}<extra></extra>',
            ), row=1, col=1)
            fig.add_trace(go.Scatter(
                x=trades['exit_time'], y=trades['exit_price'],
                mode='markers', name='Exit',
                marker=dict(symbol='triangle-down', size=9, color=_RED),
                hovertemplate='<b>Exit</b><br>%{x}<br>$%{y:.2f}<extra></extra>',
            ), row=1, col=1)

    fig.add_hline(y=0, line_dash='dash', line_color='#94a3b8', line_width=1, row=2, col=1)

    if fold_boundaries is not None:
        for date in fold_boundaries:
            for r in [1, 2]:
                fig.add_vline(
                    x=str(date), line_dash='dot',
                    line_color='rgba(100,116,139,0.5)', line_width=1,
                    row=r, col=1,
                )

    m      = oos_metrics
    calmar = m['total_return'] / abs(m['max_drawdown']) if m['max_drawdown'] != 0 else 0
    main_text = (
        f"<b>OOS Performance</b><br>"
        f"Total Return: <b>{m['total_return']*100:.2f}%</b><br>"
        f"Sharpe Ratio: <b>{m['sharpe_ratio']:.2f}</b><br>"
        f"Max Drawdown: <b>{m['max_drawdown']*100:.2f}%</b><br>"
        f"Calmar Ratio: <b>{calmar:.2f}</b><br>"
        f"Profit Factor: <b>{m['profit_factor']:.2f}</b>"
    )
    trade_text = (
        f"<b>Trade Statistics</b><br>"
        f"Total Trades: <b>{m['num_trades']}</b><br>"
        f"Win Rate: <b>{m['win_rate']*100:.2f}%</b><br>"
        f"Avg Win/Loss: <b>{m['avg_win_loss_ratio']:.2f}</b>"
    )
    yearly_ret_text = '<b>Yearly Returns:</b><br>'
    for year, ret in sorted(m['yearly_returns'].items()):
        yearly_ret_text += f'{year}: <b>{ret*100:.2f}%</b><br>'
    yearly_sh_text = '<b>Yearly Sharpe:</b><br>'
    for year, sh in sorted(m['yearly_sharpe'].items()):
        yearly_sh_text += f'{year}: <b>{sh:.2f}</b><br>'

    for text, x, y in [
        (main_text,      0.01, 0.98),
        (yearly_ret_text, 0.01, 0.65),
        (yearly_sh_text,  0.20, 0.65),
        (trade_text,      0.01, 0.35),
    ]:
        fig.add_annotation(
            xref='x domain', yref='y domain',
            x=x, y=y, xanchor='left', yanchor='top',
            text=text, showarrow=False,
            font=dict(size=10, family=_FONT_MONO),
            align='left',
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1, borderpad=8,
            row=1, col=1,
        )

    fig.update_layout(
        height=900, showlegend=True, hovermode='x unified', template=_TEMPLATE,
        title=dict(
            text='<b>Walk-Forward: Combined OOS Results</b>',
            font=dict(size=22, color='#1E293B'), x=0.5, xanchor='center',
        ),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.99,
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1,
        ),
    )

    fig.update_xaxes(title_text='Date', showgrid=True, gridwidth=1, gridcolor=_GRID, row=1, col=1)
    fig.update_xaxes(title_text='Date', showgrid=True, gridwidth=1, gridcolor=_GRID, row=2, col=1)
    fig.update_yaxes(title_text='Equity', showgrid=True, gridwidth=1, gridcolor=_GRID, row=1, col=1)
    fig.update_yaxes(title_text='Drawdown (%)', showgrid=True, gridwidth=1, gridcolor=_GRID, row=2, col=1)

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved OOS equity chart -> {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Plateau analysis — 1-D parameter sweep curves
# ──────────────────────────────────────────────────────────────────────────────

def plot_plateau_analysis(sweep_results, consensus_params, param_defs,
                          fixed_params=None, threshold=0.20,
                          show=True, save_html=None):
    """
    Grid of 1-D sensitivity sweep curves from Research.sensitivity().

    Parameters
    ----------
    sweep_results : dict
        Output of Research.sensitivity().
    consensus_params : dict
    param_defs : dict
    fixed_params : dict, optional
    threshold : float
    show : bool
    save_html : str, optional
    """
    if fixed_params is None:
        fixed_params = {}

    free_params = [k for k in param_defs if k not in fixed_params if k in sweep_results]
    n = len(free_params)
    if n == 0:
        print('No sweep results to plot.')
        return None

    cols = 3
    rows = int(np.ceil(n / cols))

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=free_params,
        vertical_spacing=min(0.10, round(0.9 / max(rows - 1, 1), 3)),
        horizontal_spacing=0.08,
    )

    for idx, name in enumerate(free_params):
        r = idx // cols + 1
        c = idx %  cols + 1

        sdf = sweep_results[name].dropna(subset=['score'])
        if len(sdf) == 0:
            continue

        peak   = sdf['score'].max()
        cutoff = peak * (1 - threshold)

        above = sdf[sdf['score'] >= cutoff]
        if len(above) > 0:
            fig.add_trace(go.Scatter(
                x=pd.concat([above['value'], above['value'][::-1]]),
                y=pd.concat([above['score'], pd.Series([cutoff] * len(above))]),
                fill='toself',
                fillcolor='rgba(34, 197, 94, 0.15)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=(idx == 0),
                name='Plateau region',
                hoverinfo='skip',
            ), row=r, col=c)

        fig.add_trace(go.Scatter(
            x=[sdf['value'].min(), sdf['value'].max()],
            y=[cutoff, cutoff],
            mode='lines',
            line=dict(color=_GREEN, dash='dot', width=1),
            showlegend=False, hoverinfo='skip',
        ), row=r, col=c)

        fig.add_trace(go.Scatter(
            x=sdf['value'], y=sdf['score'],
            mode='lines+markers',
            line=dict(color=_BLUE, width=2),
            marker=dict(size=5),
            showlegend=False, name=name,
            hovertemplate=(
                f'<b>{name}</b><br>Value: %{{x:.4g}}<br>Score: %{{y:.4f}}<extra></extra>'
            ),
        ), row=r, col=c)

        cv = consensus_params.get(name)
        if cv is not None:
            fig.add_trace(go.Scatter(
                x=[cv, cv],
                y=[sdf['score'].min() * 0.95, sdf['score'].max() * 1.02],
                mode='lines',
                line=dict(color=_RED, dash='dash', width=1.5),
                showlegend=(idx == 0),
                name='Consensus',
                hoverinfo='skip',
            ), row=r, col=c)

    fig.update_layout(
        height=max(400, rows * 300), template=_TEMPLATE,
        title=dict(
            text='<b>Plateau Analysis — 1-D Parameter Sweeps</b>',
            font=dict(size=22, color='#1E293B'), x=0.5, xanchor='center',
        ),
        legend=dict(
            yanchor='top', y=0.99, xanchor='right', x=0.99,
            bgcolor=_BG_BOX, bordercolor=_BORDER, borderwidth=1,
        ),
    )

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=_GRID, row=row, col=col)
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=_GRID,
                             title_text='score', row=row, col=col)

    if save_html:
        fig.write_html(save_html)
        print(f'✓ Saved plateau analysis chart -> {save_html}')
    if show:
        fig.show()
    return fig


# ──────────────────────────────────────────────────────────────────────────────
#  Convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def plot_walk_forward_results(
    results,
    param_defs,
    fixed_params    = None,
    benchmark_data  = None,
    show            = True,
    save_html_dir   = None,
    show_fold_perf  = True,
    show_param_evol = True,
    show_oos_equity = True,
    show_trades     = False,
):
    """
    Master convenience function: renders all walk-forward charts.

    Parameters
    ----------
    results : dict
        Output of Research.walk_forward().
    param_defs : dict
        Strategy PARAM_SPACE.
    fixed_params : dict, optional
    benchmark_data : pd.DataFrame, optional
    show : bool
    save_html_dir : str, optional
        If provided, all HTML files are written to this directory.
    show_fold_perf : bool
    show_param_evol : bool
    show_oos_equity : bool
    show_trades : bool
    """
    import os
    if fixed_params is None:
        fixed_params = {}

    results_df   = results['results_df']
    oos_metrics  = results['oos_metrics']
    oos_combined = results['oos_combined_df']

    def _html(name):
        if save_html_dir is None:
            return None
        os.makedirs(save_html_dir, exist_ok=True)
        return os.path.join(save_html_dir, name)

    if show_fold_perf:
        plot_fold_performance(results_df, show=show,
                              save_html=_html('wf_fold_performance.html'))

    if show_param_evol:
        plot_parameter_evolution(results_df, param_defs, fixed_params,
                                 show=show,
                                 save_html=_html('wf_parameter_evolution.html'))

    if show_oos_equity and oos_metrics is not None and oos_combined is not None:
        fold_boundaries = pd.to_datetime(results_df['test_start'].values)
        plot_oos_equity(
            oos_metrics, oos_combined,
            fold_boundaries=fold_boundaries,
            benchmark_data=benchmark_data,
            show=show,
            save_html=_html('wf_oos_equity.html'),
            show_trades=show_trades,
        )
    elif show_oos_equity:
        print('No valid combined OOS data — skipping equity chart.')
