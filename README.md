# portlab — Free, Open-Source Portfolio Visualizer Alternative

**Runs entirely in Google Colab. Works on your phone. Everything is free.**

Every tool family from [portfoliovisualizer.com](https://www.portfoliovisualizer.com) rebuilt as open Python you can inspect, extend, and run anywhere — plus capabilities the website puts behind its $30–55/month paywall or doesn't have at all.

## 🚀 Open a tool (tap a badge — that's it)

| Tool | What it does | Open in Colab |
|---|---|---|
| 📈 **Backtest Portfolio** | Growth, CAGR/IRR, drawdowns, rolling returns, cashflows, band rebalancing, benchmark, real returns | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/01_backtest_portfolio.ipynb) |
| 🎯 **Optimization & Efficient Frontier** | GMV, Max Sharpe, risk parity, CVaR, Sortino, Kelly, Omega, min-drawdown, Black-Litterman, resampling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/02_optimization.ipynb) |
| 🎲 **Monte Carlo / Retirement** | 5 return models, 5 withdrawal rules, success probability, percentile fan charts | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/03_monte_carlo.ipynb) |
| 🧬 **Factor Analysis** | CAPM/FF3/Carhart/FF5+Mom regressions, Ken French data auto-download, rolling exposures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/04_factor_analysis.ipynb) |
| 🔬 **Asset Analytics** | Correlation matrices (static + rolling), autocorrelation, cointegration, fund comparison | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/05_asset_analytics.ipynb) |
| ⚡ **Tactical Models** | MA timing (incl. crossover + multi-period), dual momentum, relative strength, target vol, seasonal — with current signals | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/06_tactical_models.ipynb) |
| 🗂️ **Data Explorer** | Browse the 550-stock / 300-ETF universe, cache prices to Google Drive | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/00_data_explorer.ipynb) |
| 🏭 **Full 550-Stock Pipeline** | Walk-forward backtest of 6 optimizer strategies with transaction costs + stress tests | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/07_full_550_pipeline.ipynb) |
| 🏛️ **Asset-Class History** | Long-history backtests to 1926 (Ken French + FRED proxies), dynamic allocation schedules | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Realosunboy6/free-portfolio-visualizer/blob/main/notebooks/08_asset_class_history.ipynb) |

**Quickstart:** tap a badge → edit the form fields (they render as native inputs in the Colab phone app) → *Runtime → Run all*. First run installs `portlab` (~1 min); price data caches to Google Drive so reruns are instant.

## Why this beats the website

| | Portfolio Visualizer | portlab |
|---|---|---|
| Price | Free tier limited; $30–55/mo for exports, saving, signals | **$0, forever** |
| Assets per analysis | 15 (free) / 150 (paid) | **Unlimited** |
| Data frequency | Monthly | **Daily** (or any yfinance interval) |
| Asset coverage | US-centric fund database | **Anything on Yahoo Finance** — global stocks, ETFs, crypto |
| History depth | Asset classes from 1972 | **Equities from 1926, Treasuries from the 1950s** (Ken French + FRED proxies) |
| Covariance estimation | Sample | **Ledoit-Wolf shrinkage, EWMA** |
| Optimization validation | In-sample | **Walk-forward out-of-sample with transaction costs** |
| Tactical model signals | Paid feature | **Free** (see notebook 06's last cell) |
| Fat-tail simulation | Not offered | **Student-t + block bootstrap Monte Carlo** |
| Withdrawal rules | 5 | **7** — adds VPW (Bogleheads) and Guyton-Klinger guardrails |
| After-tax backtests | Pre-tax only | **Dividend + capital-gains tax modeling** |
| Margin-call simulation | Not simulated | **Maintenance-margin rule with forced deleveraging** |
| Preset portfolios | Manual entry | **17 famous lazy portfolios built in** |
| Reproducibility | Black box | **Every formula is open source in this repo** |
| Saving / export | Paid feature | Notebooks + Drive cache; export anything with pandas |

Free data sources: [yfinance](https://github.com/ranaroussi/yfinance) (prices), [Ken French library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) (factors), [FRED](https://fred.stlouisfed.org) (CPI, T-bills), [Shiller/Yale](http://www.econ.yale.edu/~shiller/data.htm) (CAPE), plus [GitHub dataset mirrors](https://github.com/datasets) (Shiller S&P 500 1871+, 10Y Treasury yields, gold) as a zero-dependency fallback via `portlab.data.get_mirror_returns()`.

## The `portlab` package

All the math lives in an installable package (`src/portlab/`) — the notebooks are thin forms on top.

```bash
pip install "portlab @ git+https://github.com/Realosunboy6/free-portfolio-visualizer.git"
```

```python
from portlab import backtest_portfolio, monte_carlo, optimize, plots
from portlab.data import get_returns

rets = get_returns(["VTI", "TLT", "GLD"], "2010-01-01")
result = backtest_portfolio(rets, {"VTI": 0.6, "TLT": 0.3, "GLD": 0.1},
                            cashflow=500, rebalance="annually")
result.summary()                      # CAGR, Sharpe, Sortino, drawdowns, IRR...
plots.growth_chart(result.returns)    # interactive plotly
```

| Module | Contents |
|---|---|
| `portlab.data` | Cached yfinance prices, 737-ticker curated universe, **17 lazy portfolios**, asset-class proxies, **long-history asset classes (1926+)**, Ken French factors, FRED macro, Shiller CAPE, GitHub-mirror fallbacks |
| `portlab.metrics` | CAGR, IRR, Sharpe, Sortino, Calmar, Omega, drawdown tables, VaR/CVaR, capture ratios, rolling/annual tables |
| `portlab.optimize` | Mean-variance + frontier, risk parity, min-CVaR (LP), max Sortino, Kelly, Omega, min max-drawdown (LP), tracking error, info ratio, Black-Litterman, Michaud resampling, **geometric mean frontier**, group constraints, transaction costs |
| `portlab.backtest` | Target-weight backtests with cashflows/rebalancing, **PV-style leverage + maintenance margin, after-tax results, dynamic-allocation schedules, glide paths**, walk-forward `RollingBacktest`, stress episodes & shock scenarios |
| `portlab.montecarlo` | Bootstrap/block/normal/Student-t/statistical models, **7 withdrawal rules (incl. VPW + Guyton-Klinger), time-varying glide-path simulation**, success probabilities |
| `portlab.factor` | CAPM → FF5+Mom regressions (Newey-West), rolling exposures, multi-fund comparison, **performance attribution, match factor exposure** |
| `portlab.analytics` | Correlations, autocorrelation, cointegration, performance tables, **PCA, return-based fund screener** |
| `portlab.tactical` | MA/crossover/multi-period timing, dual momentum, relative strength, target vol, seasonal, CAPE switch — plus **adaptive allocation** — all evaluated lag-1, cost-aware |

**Development:** `pip install -e ".[dev]" && pytest` (106 tests, no network needed) · `python scripts/smoke_notebooks.py` runs every notebook headlessly on synthetic data.

**Honest caveats:** yfinance adjusted prices are survivorship-biased for delisted stocks and revise over time; optimizers are only as good as their inputs (that's why Ledoit-Wolf, resampling, and walk-forward validation are defaults here); tactical backtests are experiments, not trading advice. Nothing in this repo is investment advice.

---

## History

`portlab` grew out of the matrix-algebra coursework in
[550-Stocks-Portfolio-Theory-Python-Julia](https://github.com/Realosunboy6/550-Stocks-Portfolio-Theory-Python-Julia-) —
Modern Portfolio Theory implemented from scratch in Python and Julia on ~550
stocks across 11 GICS sectors. That repo remains the home of the original
notebooks; this one is the production toolkit.

## License

MIT — free for educational, research, and personal use.
