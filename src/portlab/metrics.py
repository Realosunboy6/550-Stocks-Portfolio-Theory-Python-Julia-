"""Performance and risk metrics for return series.

All functions take *simple* per-period returns (a pd.Series) unless noted.
`periods` is the number of return periods per year (252 daily, 12 monthly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize as sciopt

from .config import DEFAULT_ALPHA, DEFAULT_RF, TRADING_DAYS


# ---------------------------------------------------------------- core stats

def cagr(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    growth = float((1 + rets).prod())
    if growth <= 0:
        return -1.0
    return growth ** (periods / len(rets)) - 1


def ann_vol(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(rets.std(ddof=1) * np.sqrt(periods))


def sharpe(rets: pd.Series, rf: float = DEFAULT_RF, periods: int = TRADING_DAYS) -> float:
    excess = rets - rf / periods
    sd = excess.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(excess.mean() / sd * np.sqrt(periods))


def sortino(rets: pd.Series, rf: float = DEFAULT_RF, periods: int = TRADING_DAYS) -> float:
    excess = rets - rf / periods
    downside = excess[excess < 0]
    dd = np.sqrt((downside ** 2).sum() / len(rets))
    if dd == 0:
        return np.inf if excess.mean() > 0 else 0.0
    return float(excess.mean() / dd * np.sqrt(periods))


def omega(rets: pd.Series, threshold_annual: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Omega ratio: sum of gains above threshold / sum of losses below it."""
    thr = threshold_annual / periods
    diff = rets - thr
    losses = -diff[diff < 0].sum()
    gains = diff[diff > 0].sum()
    if losses == 0:
        return np.inf if gains > 0 else 1.0
    return float(gains / losses)


def calmar(rets: pd.Series, periods: int = TRADING_DAYS) -> float:
    mdd = max_drawdown(rets)
    if mdd == 0:
        return np.inf
    return float(cagr(rets, periods) / abs(mdd))


# ---------------------------------------------------------------- drawdowns

def drawdown_series(rets: pd.Series) -> pd.Series:
    wealth = (1 + rets).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(rets: pd.Series) -> float:
    return float(drawdown_series(rets).min())


def drawdown_table(rets: pd.Series, top: int = 10) -> pd.DataFrame:
    """Worst drawdown episodes: start, trough, recovery, depth, durations."""
    dd = drawdown_series(rets)
    episodes = []
    in_dd = False
    start = trough = None
    trough_val = 0.0
    for date, val in dd.items():
        if not in_dd and val < 0:
            in_dd, start, trough, trough_val = True, date, date, val
        elif in_dd:
            if val < trough_val:
                trough, trough_val = date, val
            if val == 0:
                episodes.append((start, trough, date, trough_val))
                in_dd = False
    if in_dd:
        episodes.append((start, trough, pd.NaT, trough_val))
    rows = [
        {
            "Start": s,
            "Trough": t,
            "Recovery": r,
            "Depth": v,
            "Length (periods)": (dd.index.get_loc(r if pd.notna(r) else dd.index[-1])
                                 - dd.index.get_loc(s) + 1),
        }
        for s, t, r, v in episodes
    ]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Depth").head(top).reset_index(drop=True)


def ulcer_index(rets: pd.Series) -> float:
    dd = drawdown_series(rets)
    return float(np.sqrt((dd ** 2).mean()))


# ---------------------------------------------------------------- tail risk

def var_historical(rets: pd.Series, alpha: float = DEFAULT_ALPHA) -> float:
    """Historical Value-at-Risk (positive number = loss)."""
    return float(-np.quantile(rets, 1 - alpha))


def cvar_historical(rets: pd.Series, alpha: float = DEFAULT_ALPHA) -> float:
    """Expected shortfall beyond VaR (positive number = loss)."""
    cutoff = np.quantile(rets, 1 - alpha)
    tail = rets[rets <= cutoff]
    if len(tail) == 0:
        return var_historical(rets, alpha)
    return float(-tail.mean())


# ---------------------------------------------------------------- benchmark-relative

def beta_alpha(rets: pd.Series, bench: pd.Series, rf: float = DEFAULT_RF,
               periods: int = TRADING_DAYS) -> tuple[float, float]:
    """CAPM beta and annualized alpha vs a benchmark."""
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    r, b = df.iloc[:, 0] - rf / periods, df.iloc[:, 1] - rf / periods
    cov = np.cov(r, b)
    beta = cov[0, 1] / cov[1, 1]
    alpha = (r.mean() - beta * b.mean()) * periods
    return float(beta), float(alpha)


def tracking_error(rets: pd.Series, bench: pd.Series, periods: int = TRADING_DAYS) -> float:
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    return float((df.iloc[:, 0] - df.iloc[:, 1]).std(ddof=1) * np.sqrt(periods))


def information_ratio(rets: pd.Series, bench: pd.Series, periods: int = TRADING_DAYS) -> float:
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    active = df.iloc[:, 0] - df.iloc[:, 1]
    sd = active.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(active.mean() / sd * np.sqrt(periods))


def capture_ratios(rets: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """(upside capture, downside capture) vs benchmark, geometric."""
    df = pd.concat([rets, bench], axis=1, join="inner").dropna()
    r, b = df.iloc[:, 0], df.iloc[:, 1]

    def _capture(mask):
        if mask.sum() == 0:
            return np.nan
        rp = (1 + r[mask]).prod() ** (1 / mask.sum()) - 1
        rb = (1 + b[mask]).prod() ** (1 / mask.sum()) - 1
        return np.nan if rb == 0 else rp / rb

    return float(_capture(b > 0)), float(_capture(b < 0))


# ---------------------------------------------------------------- cashflow-aware

def money_weighted_return(cashflows: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Annualized IRR of dated cashflows (contributions negative, final value positive).

    Index must be a DatetimeIndex. Solves NPV(rate)=0 with rate compounded annually.
    """
    t0 = cashflows.index[0]
    years = np.array([(d - t0).days / 365.25 for d in cashflows.index])
    amounts = cashflows.values.astype(float)

    def npv(rate):
        return float(np.sum(amounts / (1 + rate) ** years))

    try:
        return float(sciopt.brentq(npv, -0.9999, 10.0, maxiter=200))
    except ValueError:
        return np.nan


# ---------------------------------------------------------------- tables

def rolling_returns(rets: pd.Series, window_years: int, periods: int = TRADING_DAYS) -> pd.Series:
    """Rolling annualized return over a window of `window_years`."""
    w = window_years * periods
    return ((1 + rets).rolling(w).apply(np.prod, raw=True)) ** (1 / window_years) - 1


def annual_returns(rets: pd.Series) -> pd.Series:
    return (1 + rets).groupby(rets.index.year).prod() - 1


def monthly_return_table(rets: pd.Series) -> pd.DataFrame:
    """Year x Month table of compounded returns (PV-style)."""
    m = (1 + rets).groupby([rets.index.year, rets.index.month]).prod() - 1
    tbl = m.unstack()
    tbl.columns = [pd.Timestamp(2000, c, 1).strftime("%b") for c in tbl.columns]
    return tbl


def summary(rets: pd.Series, rf: float = DEFAULT_RF, periods: int = TRADING_DAYS,
            bench: pd.Series | None = None, name: str = "Portfolio") -> pd.Series:
    """PV-style metric block for one return series."""
    yr = annual_returns(rets)
    out = {
        "CAGR": cagr(rets, periods),
        "Annualized Volatility": ann_vol(rets, periods),
        "Sharpe Ratio": sharpe(rets, rf, periods),
        "Sortino Ratio": sortino(rets, rf, periods),
        "Calmar Ratio": calmar(rets, periods),
        "Omega Ratio": omega(rets, rf, periods),
        "Max Drawdown": max_drawdown(rets),
        "Ulcer Index": ulcer_index(rets),
        "Hist. VaR 95% (per period)": var_historical(rets),
        "Hist. CVaR 95% (per period)": cvar_historical(rets),
        "Best Year": yr.max() if len(yr) else np.nan,
        "Worst Year": yr.min() if len(yr) else np.nan,
        "Skewness": float(rets.skew()),
        "Excess Kurtosis": float(rets.kurtosis()),
    }
    if bench is not None:
        b, a = beta_alpha(rets, bench, rf, periods)
        up, down = capture_ratios(rets, bench)
        out.update({
            "Beta": b,
            "Annualized Alpha": a,
            "Tracking Error": tracking_error(rets, bench, periods),
            "Information Ratio": information_ratio(rets, bench, periods),
            "Upside Capture": up,
            "Downside Capture": down,
        })
    return pd.Series(out, name=name)
