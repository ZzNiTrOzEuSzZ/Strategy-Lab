"""
infrastructure/backtester/html_report.py
------------------------------------------
Generates a single self-contained dark-themed HTML research report.

Entry point: build_html_report()

The report takes both the structured `report` dict (from build_report) AND
the raw `research_results` dict (from Research.full_report) because the
structured report strips equity curves and trade lists while the HTML needs them.

It also accepts `strategy_fn` and `context` to re-run per-fold backtests for
Section 4 (individual fold OOS equity curves).
"""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Colour palette (dark theme)
# ---------------------------------------------------------------------------
_BG        = "#0f172a"
_BG_CARD   = "#1e293b"
_BG_TABLE  = "#162032"
_BORDER    = "#334155"
_TEXT      = "#e2e8f0"
_TEXT_MUTED= "#94a3b8"
_BLUE      = "#3b82f6"
_GREEN     = "#22c55e"
_RED       = "#ef4444"
_AMBER     = "#f59e0b"
_PURPLE    = "#a855f7"

_PLOTLY_TEMPLATE = "plotly_dark"
_PLOTLY_BG       = "rgba(15,23,42,0)"     # transparent to bleed into card
_PLOTLY_PAPER    = "rgba(30,41,59,0)"

_CHART_H_FULL = 400
_CHART_H_HALF = 320
_CHART_H_FOLD = 250


# ---------------------------------------------------------------------------
# Plotly JSON serialiser helper
# ---------------------------------------------------------------------------

def _fig_to_json(fig) -> str:
    """Return Plotly figure as JSON string (no HTML wrapper)."""
    return fig.to_json()


def _apply_dark_layout(fig, height=None):
    """Apply consistent dark theme overrides to a figure."""
    upd = dict(
        paper_bgcolor = _PLOTLY_PAPER,
        plot_bgcolor  = _PLOTLY_BG,
        font          = dict(color=_TEXT, family="Inter, system-ui, sans-serif"),
        margin        = dict(l=40, r=20, t=40, b=40),
    )
    if height:
        upd["height"] = height
    fig.update_layout(**upd)
    fig.update_xaxes(
        gridcolor="#1e293b", zerolinecolor="#334155",
        tickfont=dict(color=_TEXT_MUTED),
    )
    fig.update_yaxes(
        gridcolor="#1e293b", zerolinecolor="#334155",
        tickfont=dict(color=_TEXT_MUTED),
    )


# ---------------------------------------------------------------------------
# Section 2: Full period charts
# ---------------------------------------------------------------------------

def _build_equity_drawdown_chart(full_metrics, context) -> str | None:
    """Equity + drawdown subplot, with benchmark if context provided."""
    if full_metrics is None:
        return None

    eq = full_metrics.get("equity_curve")
    if eq is None or len(eq) == 0:
        return None

    running_max = eq.cummax()
    dd = (eq - running_max) / running_max

    benchmark_eq = None
    if context is not None:
        try:
            primary = context.primary()
            bench_ret = primary["Close"].pct_change()
            benchmark_eq = (1 + bench_ret).cumprod()
            benchmark_eq.iloc[0] = 1.0
        except Exception:
            benchmark_eq = None

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.06,
        subplot_titles=["Equity Curve (Net of Costs)", "Drawdown"],
    )

    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values,
        mode="lines", name="Strategy",
        line=dict(color=_BLUE, width=2),
        hovertemplate="<b>Strategy</b><br>%{x}<br>%{y:.4f}<extra></extra>",
    ), row=1, col=1)

    if benchmark_eq is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_eq.index, y=benchmark_eq.values,
            mode="lines", name="Benchmark (B&H)",
            line=dict(color=_AMBER, width=1.5, dash="dot"),
            hovertemplate="<b>Benchmark</b><br>%{x}<br>%{y:.4f}<extra></extra>",
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dd.index, y=dd.values * 100,
        mode="lines", name="Drawdown",
        fill="tozeroy", fillcolor="rgba(239,68,68,0.25)",
        line=dict(color=_RED, width=1),
        showlegend=False,
        hovertemplate="<b>DD</b><br>%{x}<br>%{y:.2f}%<extra></extra>",
    ), row=2, col=1)

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(30,41,59,0.8)", bordercolor=_BORDER, borderwidth=1,
        ),
    )
    _apply_dark_layout(fig, height=_CHART_H_FULL)
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Section 3: Long / Short equity comparison
# ---------------------------------------------------------------------------

def _build_long_short_chart(full_metrics) -> str | None:
    if full_metrics is None:
        return None
    ds = full_metrics.get("direction_split")
    if ds is None:
        return None

    fig = go.Figure()

    for direction, label, color in [
        ("long_only",  "Long Only",  _GREEN),
        ("short_only", "Short Only", _RED),
        ("combined",   "Combined",   _BLUE),
    ]:
        m = ds.get(direction)
        if m is None:
            continue
        eq = m.get("equity_curve")
        if eq is None or len(eq) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq.values,
            mode="lines", name=label,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{label}</b><br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
        ))

    if not fig.data:
        return None

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        hovermode="x unified",
        title=dict(text="Long / Short Equity Split", font=dict(size=14)),
        legend=dict(
            yanchor="top", y=0.99, xanchor="left", x=0.01,
            bgcolor="rgba(30,41,59,0.8)", bordercolor=_BORDER, borderwidth=1,
        ),
    )
    _apply_dark_layout(fig, height=_CHART_H_HALF)
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# Section 4: Walk-forward
# ---------------------------------------------------------------------------

def _build_is_oos_bar_chart(wf_results) -> str | None:
    results_df = wf_results.get("results_df")
    if results_df is None or len(results_df) == 0:
        return None

    valid = results_df[
        results_df["test_sharpe"].notna() & results_df["train_sharpe"].notna()
    ].copy()
    if len(valid) == 0:
        return None

    fold_labels = [f"Fold {i+1}" for i in range(len(valid))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="In-Sample Sharpe",
        x=fold_labels, y=valid["train_sharpe"].tolist(),
        marker_color=_BLUE, opacity=0.85,
        hovertemplate="<b>IS Sharpe</b><br>%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="OOS Sharpe",
        x=fold_labels, y=valid["test_sharpe"].tolist(),
        marker_color=_GREEN, opacity=0.85,
        hovertemplate="<b>OOS Sharpe</b><br>%{x}: %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=_TEXT_MUTED, line_dash="dash", line_width=1)
    fig.add_hline(y=0.5, line_color=_AMBER, line_dash="dot", line_width=1,
                  annotation_text="min target (0.5)", annotation_position="top right",
                  annotation_font=dict(color=_AMBER, size=10))

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        barmode="group",
        title=dict(text="In-Sample vs Out-of-Sample Sharpe by Fold", font=dict(size=14)),
        xaxis_title="Fold",
        yaxis_title="Sharpe Ratio",
    )
    _apply_dark_layout(fig, height=_CHART_H_HALF)
    return _fig_to_json(fig)


def _build_fold_equity_charts(wf_results, strategy_fn, context) -> list[str]:
    """
    Re-run each fold's OOS period with that fold's best params to get equity curves.
    Returns a list of JSON strings (one per fold). Empty list if not possible.
    """
    if strategy_fn is None or context is None:
        return []

    fold_records = wf_results.get("fold_records", [])
    if not fold_records:
        return []

    try:
        from .engine import backtest
    except Exception:
        return []

    charts = []
    for i, rec in enumerate(fold_records):
        try:
            # Extract fold OOS dates
            test_start = pd.Timestamp(rec["test_start"])
            test_end   = pd.Timestamp(rec["test_end"])

            # Extract best params for this fold (keys are param_*)
            params = {
                k[len("param_"):]: v
                for k, v in rec.items()
                if k.startswith("param_")
            }

            if not params:
                charts.append(None)
                continue

            # Slice context and run
            fold_ctx = context.slice(start=test_start, end=test_end)
            metrics  = backtest(fold_ctx, strategy_fn, params, cost=0.0, show_plot=False)

            if metrics is None or metrics.get("equity_curve") is None:
                charts.append(None)
                continue

            eq = metrics["equity_curve"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=eq.index, y=eq.values,
                mode="lines",
                name=f"Fold {i+1} OOS",
                line=dict(color=_BLUE, width=2),
                fill="tozeroy",
                fillcolor="rgba(59,130,246,0.12)",
                hovertemplate=f"Fold {i+1}<br>%{{x}}<br>%{{y:.4f}}<extra></extra>",
            ))
            title_text = (
                f"Fold {i+1}: {test_start.date()} - {test_end.date()}"
                f"  |  Sharpe: {rec.get('test_sharpe', 0) or 0:.2f}"
                f"  |  Return: {(rec.get('test_return', 0) or 0)*100:.1f}%"
            )
            fig.update_layout(
                template=_PLOTLY_TEMPLATE,
                title=dict(text=title_text, font=dict(size=11)),
            )
            _apply_dark_layout(fig, height=_CHART_H_FOLD)
            charts.append(_fig_to_json(fig))
        except Exception:
            charts.append(None)

    return charts


# ---------------------------------------------------------------------------
# Section 5: Parameter analysis
# ---------------------------------------------------------------------------

def _build_param_evolution_charts(wf_results) -> dict[str, str]:
    """
    For each parameter with CV > 0.20, return a line chart of value across folds.
    Returns {param_name: json_string}.
    """
    stability_df = wf_results.get("stability_df")
    fold_records = wf_results.get("fold_records", [])

    if stability_df is None or len(fold_records) == 0:
        return {}

    if isinstance(stability_df, list):
        stability_df = pd.DataFrame(stability_df)

    if len(stability_df) == 0:
        return {}

    unstable = stability_df[stability_df["cv"] > 0.20]["param"].tolist()
    charts   = {}

    fold_labels = [f"F{i+1}" for i in range(len(fold_records))]

    for param in unstable:
        try:
            key    = f"param_{param}"
            values = [rec.get(key) for rec in fold_records]
            if all(v is None for v in values):
                continue

            median_row = stability_df[stability_df["param"] == param]
            median_val = float(median_row["median"].values[0]) if len(median_row) > 0 else None

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fold_labels, y=values,
                mode="lines+markers",
                name=param,
                line=dict(color=_AMBER, width=2),
                marker=dict(size=7, color=_AMBER),
                hovertemplate=f"<b>{param}</b><br>Fold %{{x}}: %{{y:.4f}}<extra></extra>",
            ))
            if median_val is not None:
                fig.add_hline(
                    y=median_val,
                    line_color=_TEXT_MUTED, line_dash="dash", line_width=1,
                    annotation_text=f"median {median_val:.3f}",
                    annotation_position="top left",
                    annotation_font=dict(color=_TEXT_MUTED, size=9),
                )
            fig.update_layout(
                template=_PLOTLY_TEMPLATE,
                title=dict(text=f"Parameter: {param}", font=dict(size=12)),
                xaxis_title="Fold",
                yaxis_title="Value",
            )
            _apply_dark_layout(fig, height=280)
            charts[param] = _fig_to_json(fig)
        except Exception:
            continue

    return charts


# ---------------------------------------------------------------------------
# Section 6: Robustness charts
# ---------------------------------------------------------------------------

def _build_perturbation_chart(perturbation) -> str | None:
    if perturbation is None:
        return None
    if isinstance(perturbation, list):
        perturbation = pd.DataFrame(perturbation)
    if len(perturbation) == 0:
        return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=perturbation["offset_pct"] * 100,
        y=perturbation["mean_score"],
        mode="lines+markers",
        name="Mean Score",
        line=dict(color=_BLUE, width=2),
        marker=dict(size=7),
        error_y=dict(
            type="data",
            array=perturbation["std_score"].tolist(),
            visible=True, color=_BLUE, thickness=1, width=4,
        ),
        hovertemplate="<b>Offset</b>: %{x:.0f}%<br>Score: %{y:.4f}<extra></extra>",
    ))
    if "min_score" in perturbation.columns:
        fig.add_trace(go.Scatter(
            x=perturbation["offset_pct"] * 100,
            y=perturbation["min_score"],
            mode="lines",
            name="Min Score",
            line=dict(color=_RED, width=1.5, dash="dot"),
            hovertemplate="<b>Min Score</b>: %{y:.4f}<extra></extra>",
        ))

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        title=dict(text="Parameter Perturbation -- Neighbourhood Robustness", font=dict(size=14)),
        xaxis_title="Parameter Offset (%)",
        yaxis_title="Score",
    )
    _apply_dark_layout(fig, height=_CHART_H_HALF)
    return _fig_to_json(fig)


def _build_cost_stress_chart(cost_stress) -> str | None:
    if cost_stress is None:
        return None
    if isinstance(cost_stress, list):
        cost_stress = pd.DataFrame(cost_stress)
    if len(cost_stress) == 0:
        return None

    fig = go.Figure()
    metrics_to_plot = [
        ("sharpe",       "Sharpe Ratio",   _BLUE),
        ("total_return", "Total Return",   _GREEN),
        ("max_drawdown", "Max Drawdown",   _RED),
    ]
    labels = [f"{row['cost_mult']:.1f}x" for _, row in cost_stress.iterrows()]

    for col, label, color in metrics_to_plot:
        if col not in cost_stress.columns:
            continue
        values = cost_stress[col].tolist()
        if col == "total_return":
            values = [v * 100 if v is not None else None for v in values]
        if col == "max_drawdown":
            values = [v * 100 if v is not None else None for v in values]
        fig.add_trace(go.Bar(
            name=label, x=labels, y=values,
            marker_color=color, opacity=0.85,
            hovertemplate=f"<b>{label}</b><br>%{{x}}: %{{y:.3f}}<extra></extra>",
        ))

    fig.update_layout(
        template=_PLOTLY_TEMPLATE,
        barmode="group",
        title=dict(text="Transaction Cost Stress Test", font=dict(size=14)),
        xaxis_title="Cost Multiplier",
        yaxis_title="Value",
    )
    _apply_dark_layout(fig, height=_CHART_H_HALF)
    return _fig_to_json(fig)


# ---------------------------------------------------------------------------
# HTML formatting helpers
# ---------------------------------------------------------------------------

def _pct(v, dp=2, na="N/A"):
    if v is None:
        return na
    return f"{v*100:.{dp}f}%"

def _num(v, dp=2, na="N/A"):
    if v is None:
        return na
    return f"{v:.{dp}f}"

def _int_val(v, na="N/A"):
    if v is None:
        return na
    return str(int(v))

def _color_val(val, positive_is_good=True):
    """Return CSS color class based on whether positive is good."""
    if val is None:
        return "color-muted"
    v = float(val)
    if v > 0:
        return "color-green" if positive_is_good else "color-red"
    elif v < 0:
        return "color-red" if positive_is_good else "color-green"
    return "color-muted"

def _badge(text, color):
    return f'<span class="badge badge-{color}">{text}</span>'

def _metric_card(label, value_html, sub=None, color_class=""):
    sub_html = f'<div class="metric-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value {color_class}">{value_html}</div>
      {sub_html}
    </div>"""


def _make_yearly_table(full_metrics) -> str:
    """Build yearly returns/sharpe/drawdown table HTML."""
    if full_metrics is None:
        return "<p class='muted'>No data available.</p>"
    yr = full_metrics.get("yearly_returns", {})
    ys = full_metrics.get("yearly_sharpe", {})
    yd = full_metrics.get("yearly_max_drawdown", {})
    if not yr:
        return "<p class='muted'>No yearly data available.</p>"

    years = sorted(yr.keys())
    rows = ""
    for y in years:
        ret = yr.get(y)
        sharpe = ys.get(y)
        dd = yd.get(y)
        ret_class = _color_val(ret)
        sharpe_class = _color_val(sharpe)
        dd_class = _color_val(dd, positive_is_good=False)
        rows += f"""
        <tr>
          <td>{y}</td>
          <td class="{ret_class} sortable-val" data-val="{ret or 0}">{_pct(ret)}</td>
          <td class="{sharpe_class} sortable-val" data-val="{sharpe or 0}">{_num(sharpe)}</td>
          <td class="{dd_class} sortable-val" data-val="{dd or 0}">{_pct(dd)}</td>
        </tr>"""

    return f"""
    <table class="data-table sortable" id="yearly-table">
      <thead>
        <tr>
          <th data-sort="str">Year</th>
          <th data-sort="num">Return</th>
          <th data-sort="num">Sharpe</th>
          <th data-sort="num">Max DD</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _make_fold_table(fold_records) -> str:
    if not fold_records:
        return "<p class='muted'>No fold data.</p>"

    rows = ""
    for i, rec in enumerate(fold_records):
        tr = rec.get("test_return")
        ts = rec.get("test_sharpe")
        td = rec.get("test_drawdown")
        tt = rec.get("test_trades")
        ir = rec.get("train_return")
        is_ = rec.get("train_sharpe")

        overfit = None
        if ts is not None and is_ is not None and is_ != 0:
            overfit = ts / is_

        tr_class = _color_val(tr)
        ts_class = _color_val(ts)
        of_class = _color_val(overfit)

        rows += f"""
        <tr>
          <td>Fold {i+1}</td>
          <td>{rec.get('train_start','')[:10]} - {rec.get('train_end','')[:10]}</td>
          <td>{rec.get('test_start','')[:10]} - {rec.get('test_end','')[:10]}</td>
          <td class="sortable-val" data-val="{ir or 0}">{_pct(ir)}</td>
          <td class="sortable-val" data-val="{is_ or 0}">{_num(is_)}</td>
          <td class="{tr_class} sortable-val" data-val="{tr or 0}">{_pct(tr)}</td>
          <td class="{ts_class} sortable-val" data-val="{ts or 0}">{_num(ts)}</td>
          <td class="sortable-val" data-val="{td or 0}">{_pct(td)}</td>
          <td>{_int_val(tt)}</td>
          <td class="{of_class} sortable-val" data-val="{overfit or 0}">{_num(overfit)}</td>
        </tr>"""

    return f"""
    <table class="data-table sortable" id="fold-table">
      <thead>
        <tr>
          <th>Fold</th>
          <th>IS Period</th>
          <th>OOS Period</th>
          <th data-sort="num">IS Return</th>
          <th data-sort="num">IS Sharpe</th>
          <th data-sort="num">OOS Return</th>
          <th data-sort="num">OOS Sharpe</th>
          <th data-sort="num">OOS DD</th>
          <th data-sort="num">OOS Trades</th>
          <th data-sort="num">Overfit Ratio</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _make_stability_table(stability_df) -> str:
    if stability_df is None or len(stability_df) == 0:
        return "<p class='muted'>No stability data.</p>"

    if isinstance(stability_df, list):
        stability_df = pd.DataFrame(stability_df)

    rows = ""
    for _, row in stability_df.iterrows():
        cv = row.get("cv", 0) or 0
        if cv < 0.15:
            badge = _badge("Stable", "green")
        elif cv <= 0.40:
            badge = _badge("Mixed", "amber")
        else:
            badge = _badge("Unstable", "red")
        fixed_html = _badge("Fixed", "purple") if row.get("fixed") else ""
        rows += f"""
        <tr>
          <td><code>{row.get('param','')}</code></td>
          <td class="sortable-val" data-val="{row.get('median',0) or 0}">{_num(row.get('median'))}</td>
          <td class="sortable-val" data-val="{row.get('std',0) or 0}">{_num(row.get('std'))}</td>
          <td class="sortable-val" data-val="{cv}">{_num(cv)}</td>
          <td>{badge} {fixed_html}</td>
        </tr>"""

    return f"""
    <table class="data-table sortable" id="stability-table">
      <thead>
        <tr>
          <th>Parameter</th>
          <th data-sort="num">Median</th>
          <th data-sort="num">Std Dev</th>
          <th data-sort="num">CV</th>
          <th>Stability</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _make_direction_table(full_period) -> str:
    """Side-by-side table comparing Combined / Long / Short metrics."""
    rows_defs = [
        ("Total Return", "total_return", True),
        ("Sharpe Ratio", "sharpe_ratio", False),
        ("Max Drawdown", "max_drawdown", True),
        ("Win Rate",     "win_rate",     True),
        ("Num Trades",   "num_trades",   None),
        ("Profit Factor","profit_factor",False),
        ("Calmar Ratio", "calmar_ratio", False),
        ("Avg W/L",      "avg_win_loss_ratio", False),
    ]

    combined = full_period.get("combined") or {}
    long_only = full_period.get("long_only") or {}
    short_only = full_period.get("short_only") or {}

    rows_html = ""
    for label, key, is_pct in rows_defs:
        cv = combined.get(key)
        lv = long_only.get(key) if long_only else None
        sv = short_only.get(key) if short_only else None

        def fmt(v):
            if v is None:
                return "<span class='color-muted'>N/A</span>"
            if key == "num_trades":
                return str(int(v))
            if is_pct:
                cc = _color_val(v) if key != "max_drawdown" else _color_val(v, positive_is_good=False)
                return f"<span class='{cc}'>{_pct(v)}</span>"
            cc = _color_val(v)
            return f"<span class='{cc}'>{_num(v)}</span>"

        rows_html += f"<tr><td>{label}</td><td>{fmt(cv)}</td><td>{fmt(lv)}</td><td>{fmt(sv)}</td></tr>"

    return f"""
    <table class="data-table" id="direction-table">
      <thead>
        <tr>
          <th>Metric</th>
          <th>Combined</th>
          <th>Long Only</th>
          <th>Short Only</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = f"""
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

body {{
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  background: {_BG};
  color: {_TEXT};
  font-size: 14px;
  line-height: 1.5;
}}

a {{ color: {_BLUE}; text-decoration: none; }}

/* ---- Sticky header ---- */
.sticky-header {{
  position: sticky; top: 0; z-index: 100;
  background: {_BG_CARD};
  border-bottom: 1px solid {_BORDER};
  padding: 12px 24px;
  display: flex; align-items: center; justify-content: space-between;
  flex-wrap: wrap; gap: 8px;
}}
.header-left  {{ display: flex; align-items: center; gap: 16px; }}
.header-right {{ display: flex; align-items: center; gap: 12px; font-size: 12px; color: {_TEXT_MUTED}; }}
.strategy-title {{ font-size: 18px; font-weight: 700; color: {_TEXT}; }}
.header-ticker  {{ font-size: 14px; color: {_TEXT_MUTED}; }}

/* ---- Layout ---- */
.content {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}

/* ---- Section ---- */
.section {{ margin-bottom: 40px; }}
.section-title {{
  font-size: 16px; font-weight: 600; color: {_TEXT};
  padding-bottom: 8px;
  border-bottom: 1px solid {_BORDER};
  margin-bottom: 16px;
}}
.section-number {{
  display: inline-block; width: 26px; height: 26px; line-height: 26px;
  text-align: center; border-radius: 50%;
  background: {_BLUE}; color: #fff; font-size: 12px; font-weight: 700;
  margin-right: 8px;
}}

/* ---- Cards ---- */
.card {{
  background: {_BG_CARD};
  border: 1px solid {_BORDER};
  border-radius: 8px;
  padding: 16px;
}}

/* ---- Metric cards row ---- */
.metrics-row {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}}
.metric-card {{
  background: {_BG_CARD};
  border: 1px solid {_BORDER};
  border-radius: 8px;
  padding: 14px 16px;
}}
.metric-label  {{ font-size: 11px; color: {_TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }}
.metric-value  {{ font-size: 20px; font-weight: 700; }}
.metric-sub    {{ font-size: 11px; color: {_TEXT_MUTED}; margin-top: 2px; }}

/* ---- Chart container ---- */
.chart-container {{ background: {_BG_CARD}; border: 1px solid {_BORDER}; border-radius: 8px; padding: 8px; margin-bottom: 16px; }}
.chart-grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }}
@media (max-width: 900px) {{ .chart-grid-2 {{ grid-template-columns: 1fr; }} }}
.chart-placeholder {{ padding: 32px; text-align: center; color: {_TEXT_MUTED}; font-size: 13px; }}

/* ---- Tables ---- */
.table-wrapper {{ overflow-x: auto; margin-bottom: 16px; }}
.data-table {{
  width: 100%; border-collapse: collapse;
  font-size: 12.5px;
  background: {_BG_TABLE};
  border-radius: 8px; overflow: hidden;
}}
.data-table th {{
  background: #1a2942;
  color: {_TEXT_MUTED};
  font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
  padding: 10px 12px;
  text-align: left;
  white-space: nowrap;
  cursor: pointer; user-select: none;
}}
.data-table th:hover {{ color: {_TEXT}; }}
.data-table th.sort-asc::after  {{ content: " ▲"; font-size: 9px; }}
.data-table th.sort-desc::after {{ content: " ▼"; font-size: 9px; }}
.data-table td {{
  padding: 8px 12px;
  border-top: 1px solid {_BORDER};
  vertical-align: middle;
  white-space: nowrap;
}}
.data-table tr:hover {{ background: rgba(59,130,246,0.06); }}
code {{ font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 11px; color: {_AMBER}; }}

/* ---- Colour helpers ---- */
.color-green {{ color: {_GREEN}; }}
.color-red   {{ color: {_RED}; }}
.color-amber {{ color: {_AMBER}; }}
.color-blue  {{ color: {_BLUE}; }}
.color-muted {{ color: {_TEXT_MUTED}; }}

/* ---- Badges ---- */
.badge {{
  display: inline-block;
  padding: 2px 8px; border-radius: 12px;
  font-size: 10px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.06em;
}}
.badge-green  {{ background: rgba(34,197,94,0.18);  color: {_GREEN}; }}
.badge-red    {{ background: rgba(239,68,68,0.18);  color: {_RED}; }}
.badge-amber  {{ background: rgba(245,158,11,0.18); color: {_AMBER}; }}
.badge-blue   {{ background: rgba(59,130,246,0.18); color: {_BLUE}; }}
.badge-purple {{ background: rgba(168,85,247,0.18); color: {_PURPLE}; }}
.badge-lg {{
  font-size: 14px; padding: 6px 18px; border-radius: 20px;
  font-weight: 700; letter-spacing: 0.08em;
}}

/* ---- Verdict card ---- */
.verdict-card {{
  max-width: 680px; margin: 0 auto;
  background: {_BG_CARD};
  border: 1px solid {_BORDER};
  border-radius: 12px;
  padding: 32px;
  text-align: center;
}}
.verdict-status {{ font-size: 28px; font-weight: 800; margin-bottom: 16px; }}
.verdict-conclusion {{ color: {_TEXT_MUTED}; margin-bottom: 20px; font-size: 13px; line-height: 1.6; }}
.verdict-pills {{ display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-bottom: 20px; }}
.verdict-warnings {{ text-align: left; }}
.verdict-warnings h4 {{ color: {_AMBER}; font-size: 12px; text-transform: uppercase; margin-bottom: 8px; }}
.verdict-warnings ul {{ list-style: none; padding: 0; }}
.verdict-warnings li {{
  padding: 6px 12px;
  margin-bottom: 4px;
  background: rgba(245,158,11,0.08);
  border-left: 3px solid {_AMBER};
  border-radius: 0 4px 4px 0;
  font-size: 12.5px;
  color: {_TEXT};
}}

/* ---- Nav dots ---- */
.nav {{ display: flex; gap: 6px; flex-wrap: wrap; }}
.nav a {{
  padding: 3px 10px;
  border-radius: 12px;
  font-size: 11px;
  border: 1px solid {_BORDER};
  color: {_TEXT_MUTED};
  transition: all 0.15s;
}}
.nav a:hover {{ border-color: {_BLUE}; color: {_BLUE}; }}

.muted {{ color: {_TEXT_MUTED}; font-size: 12.5px; }}
hr.divider {{ border: none; border-top: 1px solid {_BORDER}; margin: 16px 0; }}
"""


# ---------------------------------------------------------------------------
# JS (table sorting + Plotly render)
# ---------------------------------------------------------------------------

_JS = """
// ---- Table sorting ----
document.querySelectorAll('table.sortable').forEach(table => {
  const ths = table.querySelectorAll('thead th');
  ths.forEach((th, colIdx) => {
    if (!th.dataset.sort) return;
    th.addEventListener('click', () => {
      const asc = !th.classList.contains('sort-asc');
      ths.forEach(h => h.classList.remove('sort-asc','sort-desc'));
      th.classList.add(asc ? 'sort-asc' : 'sort-desc');
      const tbody = table.querySelector('tbody');
      const rows  = Array.from(tbody.querySelectorAll('tr'));
      rows.sort((a, b) => {
        let av = a.cells[colIdx].dataset.val ?? a.cells[colIdx].textContent.trim();
        let bv = b.cells[colIdx].dataset.val ?? b.cells[colIdx].textContent.trim();
        if (th.dataset.sort === 'num') {
          av = parseFloat(av) || 0;
          bv = parseFloat(bv) || 0;
          return asc ? av - bv : bv - av;
        }
        return asc ? av.localeCompare(bv) : bv.localeCompare(av);
      });
      rows.forEach(r => tbody.appendChild(r));
    });
  });
});

// ---- Render all Plotly charts ----
document.querySelectorAll('.plotly-chart[data-fig]').forEach(div => {
  try {
    const fig = JSON.parse(div.dataset.fig);
    Plotly.react(div, fig.data, fig.layout, {responsive: true, displayModeBar: false});
  } catch(e) { div.innerHTML = '<p style="color:#94a3b8;padding:16px">Chart unavailable</p>'; }
});
"""


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_html_report(
    report:           dict,
    research_results: dict,
    strategy_fn:      callable | None = None,
    context           = None,
    results_dir:      str = "results",
) -> str:
    """
    Build a single self-contained dark-themed HTML research report.

    Parameters
    ----------
    report : dict
        Output of report.build_report(). Contains metadata, full_period,
        walk_forward, robustness, verdict.
    research_results : dict
        Raw output of Research.full_report(). Contains full_metrics (with
        equity_curve), wf_results, sensitivity, perturbation, cost_stress.
    strategy_fn : callable, optional
        generate_signals function. Used to re-run per-fold OOS equity curves.
    context : BacktestContext, optional
        Full research context. Used for benchmark and fold re-runs.
    results_dir : str
        Directory for output. Default "results".

    Returns
    -------
    str
        Absolute path of the saved HTML file.

    Example
    -------
    >>> html_path = build_html_report(report, report_data, strategy_fn=s.generate_signals, context=r.full_context)
    """
    # ---- Unpack ----
    meta       = report.get("metadata", {})
    full_period = report.get("full_period", {})
    wf_block   = report.get("walk_forward", {})
    robustness = report.get("robustness", {})
    verdict    = report.get("verdict", {})

    wf_results   = research_results.get("wf_results", {})
    full_metrics = research_results.get("full_metrics")
    perturbation = research_results.get("perturbation")
    cost_stress  = research_results.get("cost_stress")

    fold_records  = wf_block.get("fold_records", [])
    stability_raw = wf_block.get("stability_df", [])
    stability_df  = pd.DataFrame(stability_raw) if stability_raw else pd.DataFrame()
    consensus     = wf_block.get("consensus_params", {})

    strategy_name = meta.get("strategy_name", "Strategy")
    ticker        = meta.get("ticker", "")
    timeframe     = meta.get("timeframe", "")
    run_date      = meta.get("run_date", "")[:10]
    tradeable     = verdict.get("tradeable", False)

    # ---- Build charts ----
    try:
        equity_dd_json  = _build_equity_drawdown_chart(full_metrics, context)
    except Exception:
        equity_dd_json  = None

    try:
        ls_chart_json   = _build_long_short_chart(full_metrics)
    except Exception:
        ls_chart_json   = None

    try:
        is_oos_bar_json = _build_is_oos_bar_chart(wf_results)
    except Exception:
        is_oos_bar_json = None

    try:
        fold_equity_jsons = _build_fold_equity_charts(wf_results, strategy_fn, context)
    except Exception:
        fold_equity_jsons = []

    try:
        param_charts    = _build_param_evolution_charts(wf_results)
    except Exception:
        param_charts    = {}

    try:
        pert_chart_json = _build_perturbation_chart(perturbation)
    except Exception:
        pert_chart_json = None

    try:
        cost_chart_json = _build_cost_stress_chart(cost_stress)
    except Exception:
        cost_chart_json = None

    # ---- Build tables ----
    yearly_table_html    = _make_yearly_table(full_metrics)
    fold_table_html      = _make_fold_table(fold_records)
    stability_table_html = _make_stability_table(stability_df if len(stability_df) > 0 else stability_raw)
    direction_table_html = _make_direction_table(full_period)

    # ---- Metric cards (Section 2) ----
    combined = full_period.get("combined") or {}
    def _card(label, val_html, sub=None, color=""):
        return _metric_card(label, val_html, sub, color)

    tr_v  = combined.get("total_return")
    sh_v  = combined.get("sharpe_ratio")
    dd_v  = combined.get("max_drawdown")
    wr_v  = combined.get("win_rate")
    nt_v  = combined.get("num_trades")
    pf_v  = combined.get("profit_factor")
    cl_v  = combined.get("calmar_ratio")
    wl_v  = combined.get("avg_win_loss_ratio")

    metric_cards_html = (
        _card("Total Return",   f'<span class="{_color_val(tr_v)}">{_pct(tr_v)}</span>', f"{meta.get('data_start','')[:10]} - {meta.get('data_end','')[:10]}") +
        _card("Sharpe Ratio",   f'<span class="{_color_val(sh_v)}">{_num(sh_v)}</span>') +
        _card("Max Drawdown",   f'<span class="{_color_val(dd_v, positive_is_good=False)}">{_pct(dd_v)}</span>') +
        _card("Win Rate",       f'<span class="{_color_val(wr_v)}">{_pct(wr_v)}</span>') +
        _card("Num Trades",     f'<span class="color-blue">{_int_val(nt_v)}</span>') +
        _card("Profit Factor",  f'<span class="{_color_val(pf_v)}">{_num(pf_v)}</span>') +
        _card("Calmar Ratio",   f'<span class="{_color_val(cl_v)}">{_num(cl_v)}</span>') +
        _card("Avg W/L Ratio",  f'<span class="{_color_val(wl_v)}">{_num(wl_v)}</span>')
    )

    # ---- WF summary cards ----
    oos_sh  = wf_block.get("avg_oos_sharpe")
    oos_ret = wf_block.get("avg_oos_return")
    oos_dd  = wf_block.get("avg_oos_drawdown")
    pct_prf = wf_block.get("pct_folds_profitable")
    ovf_rat = wf_block.get("overfit_ratio")

    wf_cards_html = (
        _card("OOS Sharpe",       f'<span class="{_color_val(oos_sh)}">{_num(oos_sh)}</span>', "avg across folds") +
        _card("OOS Return",       f'<span class="{_color_val(oos_ret)}">{_pct(oos_ret)}</span>', "avg across folds") +
        _card("OOS Max DD",       f'<span class="{_color_val(oos_dd, positive_is_good=False)}">{_pct(oos_dd)}</span>', "avg across folds") +
        _card("Folds Profitable", f'<span class="{_color_val(pct_prf)}">{_pct(pct_prf, dp=0)}</span>', f"{len(fold_records)} folds total") +
        _card("Overfit Ratio",    f'<span class="{_color_val(ovf_rat)}">{_num(ovf_rat)}</span>', "OOS/IS Sharpe")
    )

    # ---- Verdict section ----
    verdict_color   = "green" if tradeable else "red"
    verdict_label   = "TRADEABLE" if tradeable else "NOT TRADEABLE"
    verdict_style   = f"color: {_GREEN}" if tradeable else f"color: {_RED}"
    conclusion_text = verdict.get("conclusion", "")
    warnings        = verdict.get("warnings", [])

    pills = [
        _badge(f"OOS Sharpe: {_num(verdict.get('oos_sharpe'))}", "blue"),
        _badge(f"Overfit: {_num(verdict.get('overfit_ratio'))}", "blue"),
        _badge(f"Params: {verdict.get('param_stability', 'unknown')}", "amber"),
        _badge(f"L/S: {verdict.get('long_short_balance', 'unknown')}", "purple"),
    ]
    pills_html = "".join(pills)

    warnings_html = ""
    if warnings:
        w_items = "".join(f"<li>{w}</li>" for w in warnings)
        warnings_html = f"""
        <div class="verdict-warnings">
          <h4>Warnings</h4>
          <ul>{w_items}</ul>
        </div>"""

    # ---- Fold equity grid ----
    fold_charts_html = ""
    if fold_equity_jsons:
        cells = ""
        for i, fig_json in enumerate(fold_equity_jsons):
            if fig_json:
                cells += f"""
                <div class="chart-container">
                  <div class="plotly-chart" data-fig="{_escape_attr(fig_json)}" style="width:100%;"></div>
                </div>"""
            else:
                cells += f"""
                <div class="chart-container">
                  <div class="chart-placeholder">Fold {i+1}: No data</div>
                </div>"""
        fold_charts_html = f'<div class="chart-grid-2">{cells}</div>'
    else:
        fold_charts_html = '<p class="muted">Fold equity curves unavailable (strategy_fn or context not provided).</p>'

    # ---- Param evolution charts ----
    param_charts_html = ""
    if param_charts:
        for pname, pjson in param_charts.items():
            param_charts_html += f"""
            <div class="chart-container">
              <div class="plotly-chart" data-fig="{_escape_attr(pjson)}" style="width:100%;"></div>
            </div>"""
    else:
        param_charts_html = '<p class="muted">No parameters with CV &gt; 0.20 (all parameters are stable).</p>'

    # ---- Consensus params ----
    consensus_html = ""
    if consensus:
        items = "  ".join(f"<code>{k}</code>: <b>{v}</b>" for k, v in consensus.items())
        consensus_html = f'<div class="card" style="margin-bottom:16px;font-size:12.5px;">Consensus params: {items}</div>'

    # ---- Robustness section content ----
    def _chart_block(fig_json, placeholder="No data available."):
        if fig_json:
            return f'<div class="chart-container"><div class="plotly-chart" data-fig="{_escape_attr(fig_json)}" style="width:100%;"></div></div>'
        return f'<div class="card"><p class="muted">{placeholder}</p></div>'

    robustness_html = ""
    if pert_chart_json or cost_chart_json:
        robustness_html = (
            _chart_block(pert_chart_json, "Perturbation data not available (skip_robustness=True).") +
            _chart_block(cost_chart_json, "Cost stress data not available (skip_robustness=True).")
        )
    else:
        robustness_html = '<div class="card"><p class="muted">Robustness analysis was skipped (--skip-robustness flag). Re-run without this flag to populate this section.</p></div>'

    # ---- Assemble HTML ----
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{strategy_name} -- {ticker} Research Report</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>{_CSS}</style>
</head>
<body>

<!-- ========================= STICKY HEADER ========================= -->
<header class="sticky-header">
  <div class="header-left">
    <div>
      <div class="strategy-title">{strategy_name}</div>
      <div class="header-ticker">{ticker} &middot; {timeframe} &middot; {run_date}</div>
    </div>
    {_badge(verdict_label, verdict_color + ' badge-lg')}
  </div>
  <div class="header-right">
    <nav class="nav">
      <a href="#s1">Overview</a>
      <a href="#s2">Full Period</a>
      <a href="#s3">L/S Split</a>
      <a href="#s4">Walk-Forward</a>
      <a href="#s5">Parameters</a>
      <a href="#s6">Robustness</a>
      <a href="#s7">Verdict</a>
    </nav>
  </div>
</header>

<div class="content">

<!-- ========================= SECTION 1: OVERVIEW ========================= -->
<section class="section" id="s1">
  <div class="section-title"><span class="section-number">1</span> Overview</div>
  <div class="card" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;font-size:12.5px;">
    <div><span class="color-muted">Strategy</span><br><b>{strategy_name}</b></div>
    <div><span class="color-muted">Ticker</span><br><b>{ticker}</b></div>
    <div><span class="color-muted">Timeframe</span><br><b>{timeframe}</b></div>
    <div><span class="color-muted">Run Date</span><br><b>{run_date}</b></div>
    <div><span class="color-muted">Data Range</span><br><b>{meta.get('data_start','')[:10]} &mdash; {meta.get('data_end','')[:10]}</b></div>
    <div><span class="color-muted">Total Bars</span><br><b>{meta.get('total_bars', 'N/A')}</b></div>
    <div><span class="color-muted">WF Folds</span><br><b>{meta.get('num_folds', len(fold_records))}</b></div>
    <div><span class="color-muted">Round-Trip Cost</span><br><b>{_pct(meta.get('cost'), dp=3)}</b></div>
  </div>
</section>

<!-- ========================= SECTION 2: FULL PERIOD ========================= -->
<section class="section" id="s2">
  <div class="section-title"><span class="section-number">2</span> Full Period Performance</div>
  <div class="metrics-row">{metric_cards_html}</div>
  {_chart_block(equity_dd_json, "Equity curve not available.")}
  <div class="section-title" style="font-size:13px;margin-top:24px;">Yearly Breakdown</div>
  <div class="table-wrapper">{yearly_table_html}</div>
</section>

<!-- ========================= SECTION 3: LONG / SHORT ========================= -->
<section class="section" id="s3">
  <div class="section-title"><span class="section-number">3</span> Long / Short Direction Split</div>
  {_chart_block(ls_chart_json, "Long/short split chart not available.")}
  <div class="table-wrapper">{direction_table_html}</div>
</section>

<!-- ========================= SECTION 4: WALK-FORWARD ========================= -->
<section class="section" id="s4">
  <div class="section-title"><span class="section-number">4</span> Walk-Forward Analysis</div>
  <div class="metrics-row">{wf_cards_html}</div>
  {consensus_html}
  {_chart_block(is_oos_bar_json, "Walk-forward fold chart not available.")}
  <div class="section-title" style="font-size:13px;margin-top:24px;">Fold Details</div>
  <div class="table-wrapper">{fold_table_html}</div>
  <div class="section-title" style="font-size:13px;margin-top:24px;">Per-Fold OOS Equity Curves</div>
  {fold_charts_html}
</section>

<!-- ========================= SECTION 5: PARAMETERS ========================= -->
<section class="section" id="s5">
  <div class="section-title"><span class="section-number">5</span> Parameter Analysis</div>
  <div class="table-wrapper">{stability_table_html}</div>
  <div class="section-title" style="font-size:13px;margin-top:24px;">Evolution of Unstable Parameters</div>
  {param_charts_html}
</section>

<!-- ========================= SECTION 6: ROBUSTNESS ========================= -->
<section class="section" id="s6">
  <div class="section-title"><span class="section-number">6</span> Robustness Analysis</div>
  {robustness_html}
</section>

<!-- ========================= SECTION 7: VERDICT ========================= -->
<section class="section" id="s7">
  <div class="section-title"><span class="section-number">7</span> Verdict</div>
  <div class="verdict-card">
    <div class="verdict-status" style="{verdict_style}">{verdict_label}</div>
    <div class="verdict-pills">{pills_html}</div>
    <div class="verdict-conclusion">{conclusion_text}</div>
    {warnings_html}
  </div>
</section>

</div><!-- /content -->

<script>{_JS}</script>
</body>
</html>"""

    # ---- Save ----
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name      = str(meta.get("strategy_name", "strategy")).replace(" ", "_")
    ticker_s  = str(meta.get("ticker", "unknown"))
    tf_s      = str(meta.get("timeframe", ""))
    ts        = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    filename  = f"{name}_{ticker_s}_{tf_s}_{ts}.html"
    filepath  = out_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"OK HTML report saved -> {filepath}")
    return str(filepath)


# ---------------------------------------------------------------------------
# Attribute escaping helper
# ---------------------------------------------------------------------------

def _escape_attr(s: str) -> str:
    """Escape a string for embedding in an HTML attribute (data-fig=...)."""
    return (
        s.replace("&", "&amp;")
         .replace('"', "&quot;")
         .replace("'", "&#39;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )
