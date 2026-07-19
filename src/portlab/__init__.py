"""portlab — free, open-source Portfolio Visualizer alternative for Colab.

High-level API:
    from portlab import data, metrics, optimize, backtest, montecarlo
    from portlab import factor, analytics, tactical, plots
"""

__version__ = "0.1.0"

from . import (analytics, backtest, covariance, factor, metrics, montecarlo,
               optimize, plots, returns, tactical)
from .backtest import backtest_portfolio, compare_portfolios
from .montecarlo import glide_weights, monte_carlo

# `data` imports yfinance/pandas_datareader lazily-adjacent deps; keep it last
# so a partial install can still use the math modules.
try:
    from . import data
except ImportError:  # pragma: no cover
    data = None

__all__ = [
    "analytics", "backtest", "covariance", "data", "factor", "metrics",
    "montecarlo", "optimize", "plots", "returns", "tactical",
    "backtest_portfolio", "compare_portfolios", "monte_carlo", "glide_weights", "__version__",
]
