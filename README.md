# 550 Stocks Portfolio Theory

Portfolio optimization analysis using matrix algebra for 550 stocks across multiple sectors, implemented in both Python and Julia.

## Repository Structure

```
notebooks/
  python/
    part1_matrix_algebra.ipynb          # Data loading, preprocessing, portfolio metrics
    part2_matrix_algebra.ipynb          # Portfolio optimization (scipy.optimize)
    portfolio_optimization_colab.ipynb  # Full workflow — Google Colab
    portfolio_optimization_vscode.ipynb # Full workflow — VS Code
  julia/
    part1_matrix_algebra.ipynb          # Data loading and preprocessing (Julia)
    part2_matrix_algebra.ipynb          # Portfolio optimization (Optim.jl)
  utilities/
    pull_data_2026.ipynb                # Stock data retrieval for 2026
docs/
  portfolio_optimization_superprompt.docx  # Prompt engineering reference guide
README.md
```

## Overview

The project implements Modern Portfolio Theory (MPT) concepts including:
- Log returns calculation
- Correlation and covariance matrix analysis
- Portfolio risk and return calculations
- Sharpe ratio optimization
- Train/test validation of optimized portfolios

## Part 1: Data Processing and Portfolio Metrics

- Load stock data from multiple sector CSV files
- Calculate log returns: `ln(P_t / P_{t-1})`
- Build return matrix (dates × tickers)
- Compute correlation and covariance matrices
- Verify positive semi-definite properties
- Visualize cumulative returns

## Part 2: Portfolio Optimization

- Maximize Sharpe ratio: `(E[R] - R_f) / σ` where R_f = 0
- Constraints: weights sum to 1, no short selling (weights ≥ 0)
- Approaches: L-BFGS-B (box) and SLSQP / IPNewton (equality)
- Train/test split for out-of-sample validation
- Compare optimized vs equal-weighted portfolios

## Requirements

### Python
```bash
pip install pandas numpy scipy plotly jupyter
```

### Julia
```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "LinearAlgebra", "Statistics", "Dates", "PlotlyJS", "Optim", "Plots"])
```

## Usage

### Python
```bash
jupyter notebook notebooks/python/part1_matrix_algebra.ipynb
jupyter notebook notebooks/python/part2_matrix_algebra.ipynb
```

### Julia
```bash
jupyter notebook notebooks/julia/part1_matrix_algebra.ipynb
jupyter notebook notebooks/julia/part2_matrix_algebra.ipynb
```

## Data Format

CSV files should include:

| Column | Description |
|--------|-------------|
| `Date` | Trading date |
| `Ticker` | Stock ticker symbol |
| `Close` | Closing price |
| `Sector` | Market sector |

## Library Mappings: Python → Julia

| Python | Julia |
|--------|-------|
| pandas | DataFrames.jl |
| numpy | LinearAlgebra, Statistics |
| scipy.optimize | Optim.jl |
| plotly | PlotlyJS.jl |

## License

Provided for educational and research purposes.
