# portlab

Portfolio analysis that runs entirely in Google Colab, on free data. Backtesting, optimization, retirement simulation, factor regressions, tactical models. No account, no API keys, nothing to pay for. It works fine from a phone.

This project started as university coursework, Modern Portfolio Theory built from scratch on about 550 stocks. At some point I wanted the whole workflow, not just the math: pull prices, test an allocation, stress it, plan withdrawals, see where the risk actually sits. So the coursework grew into a proper Python package with notebooks on top.

## The tools

Each notebook is a form. Tap a badge, change the tickers and dates, then Runtime → Run all. The first run installs the package (about a minute); prices cache to Google Drive so reruns are quick.

| Tool | What it does | Open in Colab |
|---|---|---|
| Backtest Portfolio | Growth, CAGR/IRR, drawdowns, rolling returns, cashflows, band rebalancing, leverage, taxes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/01_backtest_portfolio.ipynb) |
| Optimization and efficient frontier | Fourteen optimizers, from minimum variance to HRP and Black-Litterman | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/02_optimization.ipynb) |
| Monte Carlo / retirement | Five return models, seven withdrawal rules, success probability, fan charts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/03_monte_carlo.ipynb) |
| Factor analysis | CAPM through five-factor + momentum regressions on Ken French data, rolling exposures, attribution | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/04_factor_analysis.ipynb) |
| Asset analytics | Correlations, cointegration, PCA, a return-based screener | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/05_asset_analytics.ipynb) |
| Tactical models | Moving-average timing, dual momentum, relative strength, target volatility, adaptive allocation, with current signals | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/06_tactical_models.ipynb) |
| Data explorer | Browse the built-in universe (roughly 550 stocks and 300 ETFs), manage the price cache | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/00_data_explorer.ipynb) |
| Full 550-stock pipeline | Walk-forward backtest of six optimizer strategies with transaction costs and stress tests | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/07_full_550_pipeline.ipynb) |
| Asset-class history | Backtests reaching to 1926 using academic and government data, glide paths, dynamic allocation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/08_asset_class_history.ipynb) |

Prefer a website? `streamlit run app.py` gives you the same tools with a sidebar and buttons, and you can deploy it free on [Streamlit Community Cloud](https://share.streamlit.io). Docs, with a formula reference and a glossary, live at https://realosunboy6.github.io/free-portfolio-visualizer/.

## Using it as a library

```bash
pip install "portlab @ git+https://github.com/Realosunboy6/free-portfolio-visualizer.git"
```

```python
from portlab import backtest_portfolio, plots
from portlab.data import get_returns

rets = get_returns(["VTI", "TLT", "GLD"], "2010-01-01")
result = backtest_portfolio(rets, {"VTI": 0.6, "TLT": 0.3, "GLD": 0.1},
                            cashflow=500, rebalance="annually")
result.summary()                      # CAGR, Sharpe, Sortino, drawdowns, IRR
result.tear_sheet("report.html")      # a single HTML file you can send to anyone
plots.growth_chart(result.returns)
```

A few things I use constantly:

```python
from portlab import rebalance_trades
rebalance_trades({"VTI": 71_000, "BND": 22_000}, {"VTI": 0.6, "BND": 0.4}, cash_to_add=1_000)
# -> the exact buy/sell dollar amounts to get back to target

from portlab.data import LAZY_PORTFOLIOS      # 17 classic allocations, ready to backtest
from portlab.optimize import hrp              # hierarchical risk parity, no solver needed
```

## What's inside

| Module | Contents |
|---|---|
| `portlab.data` | Cached price downloads, a curated 737-ticker universe, 17 preset portfolios, long-history asset classes (equities to 1926), factor and macro series, mirror fallbacks |
| `portlab.metrics` | 45+ metrics: CAGR, Sharpe, Sortino, Calmar, Omega, Treynor, historical and parametric and Cornish-Fisher VaR, CVaR, CDaR, drawdown tables and ratios, trade stats, rolling Sharpe/vol/beta |
| `portlab.optimize` | Mean-variance with an efficient frontier, HRP, risk parity, CVaR and CDaR minimization, Kelly, Omega, drawdown minimization, tracking error, Black-Litterman, Michaud resampling, group constraints |
| `portlab.backtest` | Cashflows, five rebalancing schedules plus bands, leverage with an optional margin-call rule, dividend and capital-gains taxes, glide paths, walk-forward validation, stress scenarios |
| `portlab.montecarlo` | Bootstrap, block bootstrap, normal, Student-t and statistical models; withdrawal rules including VPW and Guyton-Klinger guardrails; time-varying glide paths |
| `portlab.factor` | Factor regressions with Newey-West errors, rolling exposures, performance attribution, exposure matching |
| `portlab.analytics` | Correlations, autocorrelation, cointegration, PCA, screening, dividend income history |
| `portlab.tactical` | Timing and momentum models, evaluated with a one-period execution lag and transaction costs |
| `portlab.report` | The one-line HTML tear sheet |

## Where the data comes from

Everything is free and none of it needs a key: [yfinance](https://github.com/ranaroussi/yfinance) for prices, the [Ken French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) for factor and equity-class series, [FRED](https://fred.stlouisfed.org) for CPI and Treasury yields, [Robert Shiller's dataset](http://www.econ.yale.edu/~shiller/data.htm) for the long S&P history and CAPE, and the [Datahub mirrors on GitHub](https://github.com/datasets) as a fallback that works even where the other hosts are blocked.

## Development

```bash
pip install -e ".[dev]"
pytest                               # 139 tests, all offline on synthetic data
python scripts/smoke_notebooks.py    # runs every notebook headlessly
ruff check .
```

## Caveats, honestly

Adjusted prices from free sources carry survivorship bias for delisted stocks and get revised over time. The long asset-class histories are stitched from academic proxies, useful for research but not investable fund records. Optimizers are only as good as their inputs, which is why shrinkage estimation and walk-forward validation are the defaults here rather than options. Tactical backtests are experiments. None of this is investment advice.

## History

portlab grew out of the matrix-algebra coursework in [550-Stocks-Portfolio-Theory-Python-Julia](https://github.com/Realosunboy6/550-Stocks-Portfolio-Theory-Python-Julia-), where the same ideas were first implemented from scratch in Python and Julia. That repo keeps the original notebooks; this one is the toolkit they turned into.

## License

MIT. Use it for coursework, research, or your own money.
