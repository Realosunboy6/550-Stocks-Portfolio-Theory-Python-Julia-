"""Asset analytics: correlations (static + rolling), autocorrelation,
cointegration, and side-by-side performance comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from .config import DEFAULT_RF, TRADING_DAYS
from . import metrics as M


def correlation_matrix(rets: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    return rets.corr(method=method)


def rolling_correlation(a: pd.Series, b: pd.Series, window: int = 63) -> pd.Series:
    df = pd.concat([a, b], axis=1, join="inner").dropna()
    return df.iloc[:, 0].rolling(window).corr(df.iloc[:, 1])


def autocorrelation(rets: pd.Series, lags: int = 12) -> pd.Series:
    """Autocorrelation of returns at lags 1..lags (with 95% CI in attrs)."""
    vals = [rets.autocorr(lag=k) for k in range(1, lags + 1)]
    out = pd.Series(vals, index=pd.Index(range(1, lags + 1), name="lag"))
    out.attrs["ci95"] = 1.96 / np.sqrt(len(rets))
    return out


def cointegration_test(price_a: pd.Series, price_b: pd.Series) -> pd.Series:
    """Engle-Granger cointegration between two (log) price series."""
    df = pd.concat([price_a, price_b], axis=1, join="inner").dropna()
    la, lb = np.log(df.iloc[:, 0]), np.log(df.iloc[:, 1])
    t_stat, p_value, crit = coint(la, lb)
    hedge = float(np.polyfit(lb, la, 1)[0])
    spread = la - hedge * lb
    adf_p = adfuller(spread)[1]
    return pd.Series({
        "EG t-stat": float(t_stat),
        "EG p-value": float(p_value),
        "crit 5%": float(crit[1]),
        "hedge ratio": hedge,
        "spread ADF p-value": float(adf_p),
        "cointegrated (5%)": bool(p_value < 0.05),
    })


def performance_table(rets: pd.DataFrame, rf: float = DEFAULT_RF,
                      periods: int = TRADING_DAYS,
                      bench: pd.Series | None = None) -> pd.DataFrame:
    """PV-style side-by-side comparison of several assets/funds."""
    return pd.concat(
        [M.summary(rets[c].dropna(), rf=rf, periods=periods, bench=bench, name=c)
         for c in rets.columns], axis=1)
