# Contributing to portlab

Contributions welcome — bug reports, docs fixes, new metrics/optimizers, examples.

## Setup
```bash
git clone https://github.com/Realosunboy6/free-portfolio-visualizer.git
cd free-portfolio-visualizer
pip install -e ".[dev]"
pytest -q                              # all tests run offline on synthetic data
python scripts/smoke_notebooks.py      # headless notebook check
```

## Ground rules
- Every new function ships with tests (synthetic data, no network).
- Default parameters must never change existing outputs (backward compatibility).
- scipy-first; cvxpy stays optional. Free data sources only.
- Run `ruff check .` before pushing.
- Metrics need a formula entry in `docs/formulas.md`.
