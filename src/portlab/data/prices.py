"""Price downloads: batched yfinance with retry/backoff and parquet caching.

`auto_adjust=True` means Close is dividend/split-adjusted — total-return
prices, matching Portfolio Visualizer's reinvested-dividends convention.
"""

from __future__ import annotations

import time

import pandas as pd

from . import cache


def download_prices(tickers: list[str], start: str, end: str | None = None,
                    interval: str = "1d", batch_size: int = 50,
                    max_retries: int = 3) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker (no cache)."""
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                data = yf.download(batch, start=start, end=end, interval=interval,
                                   auto_adjust=True, progress=False, threads=True)
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep(2 ** (attempt + 1))
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else \
            data[["Close"]].rename(columns={"Close": batch[0]})
        frames.append(close)
    out = pd.concat(frames, axis=1)
    return out.dropna(how="all")


def get_prices(tickers: list[str] | str, start: str, end: str | None = None,
               interval: str = "1d", refresh: bool = False, **kwargs) -> pd.DataFrame:
    """Cache-aware price fetch — the function notebooks should use."""
    if isinstance(tickers, str):
        tickers = [t.strip() for t in tickers.replace(";", ",").split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))
    key = cache.cache_key("prices", sorted(tickers), start, end, interval)
    if not refresh:
        hit = cache.load(key)
        if hit is not None:
            return hit
    df = download_prices(tickers, start, end, interval, **kwargs)
    cache.save(key, df)
    return df


def get_returns(tickers: list[str] | str, start: str, end: str | None = None,
                interval: str = "1d", log: bool = False, **kwargs) -> pd.DataFrame:
    """Convenience: prices -> cleaned simple (or log) returns."""
    from ..returns import clean_returns, log_returns, simple_returns
    prices = get_prices(tickers, start, end, interval, **kwargs)
    rets = log_returns(prices) if log else simple_returns(prices)
    return clean_returns(rets)
