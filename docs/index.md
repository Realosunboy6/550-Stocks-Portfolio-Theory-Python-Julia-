# portlab

Portfolio analysis that runs entirely in Google Colab, on free data. Phone-friendly.

Backtesting, optimization, retirement planning, factor analysis and tactical models,
all as open Python you can inspect, extend, and run anywhere.

## Quickstart

The fastest path is a Colab notebook — tap a badge in the
[repository README](https://github.com/Realosunboy6/free-portfolio-visualizer), edit the form,
*Runtime → Run all*.

As a library:

```bash
pip install portlab   # or: pip install "portlab @ git+https://github.com/Realosunboy6/free-portfolio-visualizer.git"
```

```python
from portlab import backtest_portfolio, plots
from portlab.data import get_returns

rets = get_returns(["VTI", "TLT", "GLD"], "2010-01-01")
result = backtest_portfolio(rets, {"VTI": 0.6, "TLT": 0.3, "GLD": 0.1})
result.summary()                       # 20+ metrics vs benchmark
result.tear_sheet("my_report.html")    # shareable one-file report
```

## Who is it for?

- **Students** — every metric has an open formula ([Formula Reference](formulas.md)) and the
  notebooks dissect each analysis step by step.
- **Individuals** — backtest your actual portfolio, plan retirement with 7 withdrawal rules,
  and get an exact [rebalancing trade list](examples.md).
- **Professionals** — Ledoit-Wolf/semicovariance estimation, HRP, CVaR/CDaR optimization,
  walk-forward validation with transaction costs, factor attribution.
