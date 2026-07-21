"""Factor regression analysis: CAPM, Fama-French 3/5, Carhart 4, FF5+Momentum.

Factor data comes free from the Ken French data library via
`portlab.data.factors.get_ff_factors`. Regressions use statsmodels OLS with
optional Newey-West (HAC) standard errors, plus rolling factor exposures.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

MODEL_FACTORS = {
    "capm":    ["Mkt-RF"],
    "ff3":     ["Mkt-RF", "SMB", "HML"],
    "carhart": ["Mkt-RF", "SMB", "HML", "Mom"],
    "ff5":     ["Mkt-RF", "SMB", "HML", "RMW", "CMA"],
    "ff5_mom": ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"],
}


def factor_regression(
    rets: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff3",
    rf_col: str = "RF",
    newey_west: bool = True,
    periods: int = 12,
) -> pd.DataFrame:
    """Regress excess returns on a factor model.

    rets: simple returns at the same frequency as `factors` (usually monthly).
    factors: DataFrame containing the model's factor columns plus `rf_col`,
        in decimal units (get_ff_factors handles the percent conversion).
    Returns a table with loadings, t-stats, p-values, and an annualized alpha
    row, with R² in `.attrs`.
    """
    if model not in MODEL_FACTORS:
        raise ValueError(f"model must be one of {sorted(MODEL_FACTORS)}")
    cols = MODEL_FACTORS[model]
    df = pd.concat([rets.rename("_ret"), factors], axis=1, join="inner").dropna()
    if len(df) < len(cols) + 10:
        raise ValueError("not enough overlapping observations for regression")
    y = df["_ret"] - df[rf_col]
    X = sm.add_constant(df[cols])
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 6}) if newey_west \
        else sm.OLS(y, X).fit()

    out = pd.DataFrame({
        "loading": fit.params,
        "t_stat": fit.tvalues,
        "p_value": fit.pvalues,
    })
    out = out.rename(index={"const": "alpha (per period)"})
    out.loc["alpha (annualized)"] = [
        (1 + fit.params["const"]) ** periods - 1, np.nan, np.nan]
    out.attrs["r_squared"] = float(fit.rsquared)
    out.attrs["r_squared_adj"] = float(fit.rsquared_adj)
    out.attrs["n_obs"] = int(fit.nobs)
    return out


def rolling_exposures(
    rets: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff3",
    window: int = 36,
    rf_col: str = "RF",
) -> pd.DataFrame:
    """Rolling factor loadings (default 36-period window)."""
    cols = MODEL_FACTORS[model]
    df = pd.concat([rets.rename("_ret"), factors], axis=1, join="inner").dropna()
    y = df["_ret"] - df[rf_col]
    X = sm.add_constant(df[cols])
    rows = {}
    for end in range(window, len(df) + 1):
        sl = slice(end - window, end)
        fit = sm.OLS(y.iloc[sl], X.iloc[sl]).fit()
        rows[df.index[end - 1]] = fit.params
    out = pd.DataFrame(rows).T
    return out.rename(columns={"const": "alpha"})


def compare_funds(
    rets: pd.DataFrame,
    factors: pd.DataFrame,
    model: str = "ff3",
    **kwargs,
) -> pd.DataFrame:
    """One regression per column; wide table of loadings/alpha/R² per fund."""
    rows = {}
    for col in rets.columns:
        try:
            tbl = factor_regression(rets[col].dropna(), factors, model, **kwargs)
        except ValueError:
            continue
        row = tbl["loading"].to_dict()
        row["R²"] = tbl.attrs["r_squared"]
        rows[col] = row
    return pd.DataFrame(rows).T


def attribution(
    rets: pd.Series,
    factors: pd.DataFrame,
    model: str = "ff3",
    rf_col: str = "RF",
) -> pd.DataFrame:
    """Factor performance attribution: cumulative return contribution of each
    factor exposure plus alpha and residual (selection) effects."""
    cols = MODEL_FACTORS[model]
    df = pd.concat([rets.rename("_ret"), factors], axis=1, join="inner").dropna()
    y = df["_ret"] - df[rf_col]
    X = sm.add_constant(df[cols])
    fit = sm.OLS(y, X).fit()
    parts = {c: fit.params[c] * df[c] for c in cols}
    parts["alpha"] = pd.Series(fit.params["const"], index=df.index)
    parts["residual"] = fit.resid
    parts["risk-free"] = df[rf_col]
    contrib = pd.DataFrame(parts)
    contrib.attrs["total"] = contrib.sum().rename("cumulative contribution")
    contrib.attrs["loadings"] = fit.params.rename(index={"const": "alpha"})
    return contrib


def match_exposure(
    target_rets: pd.Series,
    candidate_rets: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    model: str = "ff3",
    bounds=(0.0, 1.0),
) -> dict:
    """Replicate a target fund with a portfolio of candidates: minimize
    tracking error vs the target, then compare the factor loadings of
    target and replica."""
    from .optimize import min_tracking_error
    from .metrics import tracking_error as te_metric

    w = min_tracking_error(candidate_rets, target_rets, bounds=bounds)
    replica = (candidate_rets[w.index] * w.values).sum(axis=1)
    out = {"weights": w,
           "tracking_error": te_metric(replica, target_rets),
           "replica_returns": replica}
    if factors is not None:
        both = pd.DataFrame({"Target": target_rets, "Replica": replica})
        out["loadings"] = compare_funds(both, factors, model=model)
    return out
