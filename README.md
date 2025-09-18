# NEM Mean Reversion Backtest

## Overview
This workspace explores mean-reversion strategies for Australian National Electricity Market (NEM) spot prices. The core idea is to recreate a Rob Carver style process: build forecast variants, normalise and cap them, search for robust weightings, and size positions with volatility targeting before measuring risk-adjusted returns.

## Repository layout
- `backtester.py` implements the Carver-style engine. It includes helper scalers, cost handling, and the `CarverBacktest` class that wraps signal generation, Optuna-based weight searches, position sizing, and PnL aggregation.
- `timeseries_help.py` houses a grab bag of notebook-friendly time-series utilities (rolling statistics, stationarity tests, plotting helpers) used during exploratory analysis.
- `aus_meanrev.ipynb` is the main research notebook. It loads regional NEM data, runs the backtester for several parameter spans, and visualises the optimisation surface and equity curves.
- `results.md` summarises the headline performance pulled from the notebook so you can review outcomes without opening the `.ipynb`.

## Backtester highlights
- **Strategy coverage**: Supports breakout, EWMA crossover, accelerated EWMAC, and the mean-reversion variants used here (`meanrev_eq` and `meanrev_trend_vol`).
- **Optuna integration**: `CarverBacktest.run_optimization` spins up Optuna studies per ticker to solve for non-negative rule weights subject to Carver capping.
- **Risk & costs**: Volatility targeting (default 20% annualised) and optional buffers, commissions, and spreads can be supplied via maps to keep each instrument consistent.
- **Extensibility**: You can inject your own forecast functions by exposing them in a `portfolio_strategy` module or passing callables into the constructor.

## Getting started
1. Prepare a price DataFrame indexed by datetime with columns per ticker (see the notebook for shape expectations).
2. Instantiate the backtester:
   ```python
   bt = CarverBacktest(
       price_frame=prices,
       tickers=["QLD1", "NSW1", "SA1", "VIC1", "TAS1"],
       strategy_type="meanrev_eq",
       rule_variations=[[span] for span in range(20, 361, 5)],
       capital=1_000_000,
       pct_vol_target=0.20,
       commission_bps_map={"QLD1": 0.5},  # optional
       spread_bps_map=0.5                  # optional
   )
   ```
3. Run `bt.run_optimization()` to obtain per-ticker weightings, then call `bt.run()` to generate signals, PnL, and a summary DataFrame.
4. Use helpers such as `bt.get_summary()` or the plotting utilities in the notebook to inspect results.

## Notebooks and results
The notebook is intentionally verbose, saving Plotly figures and Optuna diagnostics inline. For a quick narrative of the latest run (daily vs hourly, with and without fees) see `results.md`.

## Requirements
Python >= 3.10 with `pandas`, `numpy`, `plotly`, `optuna`, and supporting scientific stack libraries listed at the top of `timeseries_help.py`. Install them via your preferred environment manager before running scripts.

## Next steps
- Swap in alternative spans or add new rule families to challenge the optimiser.
- Extend the cost model (e.g. tick-size aware slippage) before taking signals live.
- Automate data refresh and run scheduling so the research notebook can double as a nightly report.

