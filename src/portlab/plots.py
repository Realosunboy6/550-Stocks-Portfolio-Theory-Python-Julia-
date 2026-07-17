"""Interactive Plotly charts shared by all notebooks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from . import metrics as M

_LAYOUT = dict(template="plotly_white", height=450,
               margin=dict(l=40, r=20, t=60, b=40))


def growth_chart(rets: pd.DataFrame | pd.Series, initial: float = 10_000,
                 log_scale: bool = False, title: str = "Portfolio Growth") -> go.Figure:
    df = rets.to_frame() if isinstance(rets, pd.Series) else rets
    fig = go.Figure()
    for col in df.columns:
        wealth = initial * (1 + df[col].fillna(0)).cumprod()
        fig.add_trace(go.Scatter(x=wealth.index, y=wealth, name=str(col), mode="lines"))
    fig.update_layout(title=title, yaxis_title="Value ($)",
                      yaxis_type="log" if log_scale else "linear", **_LAYOUT)
    return fig


def drawdown_chart(rets: pd.DataFrame | pd.Series,
                   title: str = "Drawdowns") -> go.Figure:
    df = rets.to_frame() if isinstance(rets, pd.Series) else rets
    fig = go.Figure()
    for col in df.columns:
        dd = M.drawdown_series(df[col].dropna())
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=str(col),
                                 mode="lines", fill="tozeroy"))
    fig.update_layout(title=title, yaxis_tickformat=".0%", **_LAYOUT)
    return fig


def frontier_chart(frontier_df: pd.DataFrame, mu: pd.Series | None = None,
                   cov: pd.DataFrame | None = None,
                   highlight: dict[str, tuple[float, float]] | None = None,
                   title: str = "Efficient Frontier") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_df["volatility"], y=frontier_df["return"], mode="lines+markers",
        name="Frontier", marker=dict(color=frontier_df["sharpe"],
                                     colorscale="Viridis", showscale=True,
                                     colorbar=dict(title="Sharpe"))))
    if mu is not None and cov is not None:
        sd = np.sqrt(np.diag(cov.values))
        fig.add_trace(go.Scatter(x=sd, y=mu.values, mode="markers+text",
                                 text=list(mu.index), textposition="top center",
                                 name="Assets", marker=dict(symbol="diamond", size=9)))
    for name, (vol, ret) in (highlight or {}).items():
        fig.add_trace(go.Scatter(x=[vol], y=[ret], mode="markers+text",
                                 text=[name], textposition="bottom right",
                                 name=name, marker=dict(size=14, symbol="star")))
    fig.update_layout(title=title, xaxis_title="Volatility (ann.)",
                      yaxis_title="Expected Return (ann.)",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%", **_LAYOUT)
    return fig


def corr_heatmap(corr: pd.DataFrame, title: str = "Correlation Matrix") -> go.Figure:
    fig = px.imshow(corr, zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                    aspect="auto", title=title)
    fig.update_layout(**{k: v for k, v in _LAYOUT.items() if k != "height"},
                      height=max(450, 22 * len(corr)))
    return fig


def weights_chart(weights: pd.Series | pd.DataFrame,
                  title: str = "Allocation") -> go.Figure:
    if isinstance(weights, pd.Series):
        w = weights[weights.abs() > 1e-4].sort_values(ascending=True)
        fig = go.Figure(go.Bar(x=w.values, y=[str(i) for i in w.index],
                               orientation="h"))
        fig.update_layout(title=title, xaxis_tickformat=".1%", **_LAYOUT)
        return fig
    fig = go.Figure()
    for col in weights.columns:
        fig.add_trace(go.Scatter(x=weights.index, y=weights[col], name=str(col),
                                 mode="lines", stackgroup="w"))
    fig.update_layout(title=title, yaxis_tickformat=".0%", **_LAYOUT)
    return fig


def fan_chart(percentiles: pd.DataFrame, title: str = "Monte Carlo Projection",
              log_scale: bool = True) -> go.Figure:
    """percentiles: MonteCarloResult.percentiles() output (p10..p90 columns)."""
    cols = sorted(percentiles.columns, key=lambda c: int(c[1:]))
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Scatter(
            x=percentiles.index, y=percentiles[col], name=col, mode="lines",
            line=dict(width=2 if col == "p50" else 1),
            fill="tonexty" if i > 0 else None,
            fillcolor="rgba(31,119,180,0.12)"))
    fig.update_layout(title=title, xaxis_title="Years",
                      yaxis_title="Balance ($)",
                      yaxis_type="log" if log_scale else "linear", **_LAYOUT)
    return fig


def rolling_chart(series: pd.DataFrame | pd.Series, title: str,
                  yformat: str = ".0%") -> go.Figure:
    df = series.to_frame() if isinstance(series, pd.Series) else series
    fig = go.Figure()
    for col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=str(col), mode="lines"))
    fig.update_layout(title=title, yaxis_tickformat=yformat, **_LAYOUT)
    return fig


def annual_returns_chart(rets: pd.DataFrame | pd.Series,
                         title: str = "Annual Returns") -> go.Figure:
    df = rets.to_frame() if isinstance(rets, pd.Series) else rets
    fig = go.Figure()
    for col in df.columns:
        yr = M.annual_returns(df[col].dropna())
        fig.add_trace(go.Bar(x=yr.index.astype(str), y=yr.values, name=str(col)))
    fig.update_layout(title=title, yaxis_tickformat=".0%", barmode="group", **_LAYOUT)
    return fig
