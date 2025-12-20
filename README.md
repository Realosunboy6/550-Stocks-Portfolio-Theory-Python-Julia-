# 550 Stocks Portfolio Theory

This repository contains portfolio optimization analysis using matrix algebra for 550 stocks across multiple sectors.

## Overview

The project implements Modern Portfolio Theory (MPT) concepts including:
- Log returns calculation
- Correlation and covariance matrix analysis
- Portfolio risk and return calculations
- Sharpe ratio optimization
- Train/test validation of optimized portfolios

## Files

### Python Implementation
- `part1_Matrix_algebra.ipynb` - Data loading, preprocessing, and basic portfolio metrics
- `part2_Matrix_algebra.ipynb` - Portfolio optimization using scipy.optimize

### Julia Implementation
- `part1_Matrix_algebra_julia.ipynb` - Data loading, preprocessing, and basic portfolio metrics (Julia version)
- `part2_Matrix_algebra_julia.ipynb` - Portfolio optimization using Optim.jl (Julia version)

## Part 1: Data Processing and Portfolio Metrics

**Key Features:**
- Load stock data from multiple sector CSV files
- Calculate log returns for each stock
- Create return matrix with dates as rows and tickers as columns
- Compute correlation and covariance matrices
- Verify positive semi-definite properties
- Calculate portfolio metrics (mean, variance, Sharpe ratio)
- Visualize cumulative returns

## Part 2: Portfolio Optimization

**Key Features:**
- Define Sharpe ratio optimization objective
- Implement multiple optimization approaches:
  - L-BFGS-B with box constraints
  - SLSQP with equality constraints (Python) / IPNewton (Julia)
- Train/test split for validation
- Select top volatile assets for optimization
- Visualize optimized portfolio performance
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
jupyter notebook part1_Matrix_algebra.ipynb
jupyter notebook part2_Matrix_algebra.ipynb
```

### Julia
```bash
jupyter notebook part1_Matrix_algebra_julia.ipynb
jupyter notebook part2_Matrix_algebra_julia.ipynb
```

## Data Format

The code expects CSV files with the following columns:
- `Date` - Trading date
- `Ticker` - Stock ticker symbol
- `Close` - Closing price
- `Sector` - Market sector

## Key Concepts

### Log Returns
Log returns are calculated as: `ln(P_t / P_{t-1})`

### Sharpe Ratio
Risk-adjusted return metric: `(E[R] - R_f) / σ` where R_f = 0

### Portfolio Optimization
Maximize Sharpe ratio subject to:
- Weights sum to 1
- No short selling (weights ≥ 0)

## Conversion Notes

The Julia implementation provides equivalent functionality to the Python version with the following key library mappings:

| Python | Julia |
|--------|-------|
| pandas | DataFrames.jl |
| numpy | LinearAlgebra, Statistics |
| scipy.optimize | Optim.jl |
| plotly | PlotlyJS.jl |

## License

This project is provided for educational and research purposes.
