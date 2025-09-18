# Backtest Results Snapshot

Figures below are copied from `aus_meanrev.ipynb` after running the Carver-style optimiser across Australian NEM regions. Each study swept equilibrium spans from 5 to 360 (step 5) and defaulted to the 20% volatility target in `CarverBacktest`.

## Daily mean reversion (no fees)
- Optuna pushed every region to the single-span rule labelled `100`, implying the 100-day equilibrium performed best when costs were ignored.
- Sharpe ratios per ticker:
  - QLD1: 1.2367
  - NSW1: 0.9575
  - SA1: 0.3837
  - VIC1: 0.5514
  - TAS1: 0.6031

## Daily mean reversion (with ASX fees)
- Commission/spread settings from the notebook trim returns but keep the span preference unchanged (`100`).
- Sharpe ratios per ticker:
  - QLD1: 1.0115
  - NSW1: 0.5496
  - SA1: 0.3789
  - VIC1: 0.5233
  - TAS1: 0.5830

## Hourly mean reversion
- Re-running the study on hourly bars leads to far weaker profiles even before costs; Sharpe values fall below 0.20 for every region.
- Sharpe ratios per ticker:
  - QLD1: 0.1789
  - NSW1: 0.1271
  - SA1: 0.0934
  - VIC1: 0.0588
  - TAS1: 0.0754

## Takeaways
- The 100-day equilibrium span dominated the search in every scenario, suggesting other spans may need additional constraints or blended weighting to diversify.
- Costs matter: introducing realistic ASX fees compresses Sharpe by 30-45% depending on the region.
- Hourly sampling does not justify the extra trading activity under the current configuration; investigate alternative signals or stricter risk controls before considering intraday deployment.
