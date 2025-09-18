"""
carver_optimizers_two_stage_cap.py
----------------------------------
Self-contained Carver-style optimizer that preserves YOUR original forecast functions,
changes only scaling & capping, and performs correct position sizing / PnL / Sharpe.

Method (per Carver):
  1) For each rule variant (sub-forecast): scale so long-run |forecast| ~ 10, then **cap to +/-20**.
  2) Combine sub-forecasts with non-negative weights (sum=1), then **cap combined forecast to +/-20**.
  3) Size positions with volatility targeting:
        position (blocks) = volatility_scalar * (combined_forecast / 10)
     where volatility_scalar = daily_cash_vol_target / instrument_value_daily_vol.
     instrument_value_daily_vol = std(price_diff) * value_per_point * FX * sqrt(bars/day).
     All volatility inputs are shifted by one bar to avoid look-ahead.
  4) PnL in account CCY = positions.shift(1) * price.diff() * value_per_point * FX.
     Returns = PnL / capital. Sharpe annualised with sqrt(256).

Exposed functions:
  - optimize_breakout_weights_carver_method(...)
  - optimize_ewma_cross_weights_carver_method(...)

This module will automatically call your own forecast functions if available:
  - calc_breakout_forecast(price_series: pd.Series, horizon: int) -> pd.Series
  - calc_ewma_cross_forecast(price_series: pd.Series, fast: int, slow: int) -> pd.Series
It looks in a module named 'portfolio_strategy' first, then in __main__.

You can also pass them explicitly via the `external_*_forecast_fn` parameters.
"""

from __future__ import annotations
import glob
import importlib
from typing import Dict, Any, List, Tuple, Optional, Callable, Tuple as Tup
import numpy as np
import pandas as pd
import pickle
import os
import json
import re
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import importlib
import inspect
from contextlib import contextmanager
import optuna

BUSDAYS = 365


# ---------- Try to resolve user-supplied forecast functions ----------
def _try_resolve(name: str) -> Optional[Callable]:
    # 1) portfolio_strategy module (your uploaded file)
    try:
        mod = importlib.import_module("portfolio_strategy")
        if hasattr(mod, name):
            return getattr(mod, name)
    except Exception:
        pass
    # 2) __main__
    try:
        import __main__

        if hasattr(__main__, name):
            return getattr(__main__, name)
    except Exception:
        pass
    return None


# ---------- Scaling & Capping ----------
def _median_abs(x: pd.Series) -> float:
    xa = pd.Series(x).abs().dropna()
    if len(xa) == 0:
        return 1.0
    med = float(xa.median())
    return 1.0 if med == 0.0 else med


def scale_and_cap_subforecast_abs10(raw: pd.Series, cap: float = 20.0) -> pd.Series:
    s = pd.Series(raw, index=raw.index, dtype=float)
    k = 10.0 / _median_abs(s)
    return (s * k).clip(-cap, cap)


# --- NO-LOOKAHEAD SCALING OPTIONS ---


def scale_and_cap_abs10_expanding(
    raw: pd.Series, cap: float = 20.0, min_periods: int = 252
) -> pd.Series:
    """
    Expanding, past-only calibration:
    k_t = 10 / median(|raw| up to t-1), then cap to +/-cap.
    """
    s = pd.Series(raw, dtype=float)
    med = s.abs().expanding(min_periods=min_periods).median().shift(1)
    med = med.replace(0.0, np.nan).ffill()
    k = (10.0 / med).clip(lower=0.0)
    scaled = (s * k).clip(-cap, cap)
    return scaled.fillna(0.0)


def scale_and_cap_abs10_fixed_window(
    raw: pd.Series,
    cap: float = 20.0,
    calibration_end=None,
    lookback_days: int | None = None,
) -> pd.Series:
    """
    Fixed pre-sample calibration (no look-ahead).
    Choose either a date `calibration_end` or a first N-day window.
    """
    s = pd.Series(raw, dtype=float)
    if calibration_end is None and lookback_days is None:
        # default: first 25% of the sample
        cut = int(len(s) * 0.25)
        med = s.iloc[:cut].abs().median()
    else:
        if calibration_end is not None:
            med = s.loc[:calibration_end].abs().median()
        else:
            med = s.iloc[:lookback_days].abs().median()
    if not np.isfinite(med) or med == 0.0:
        med = 1.0
    k = 10.0 / float(med)
    return (s * k).clip(-cap, cap)


# Optional: Carver fixed scalars for breakout horizons (no look-ahead by design)
CARVER_BREAKOUT_SCALARS = {10: 0.60, 20: 0.67, 40: 0.70, 80: 0.73, 160: 0.74, 320: 0.74}


def scale_breakout_subforecast(
    raw: pd.Series,
    horizon: int,
    use_carver_scalars: bool = True,
    scalars_dict: dict[int, float] | None = None,
    cap: float = 20.0,
) -> pd.Series:
    s = pd.Series(raw, dtype=float)
    if use_carver_scalars:
        table = CARVER_BREAKOUT_SCALARS if scalars_dict is None else scalars_dict
        scalar = float(table.get(int(horizon), list(table.values())[-1]))
        scaled = s * scalar
    else:
        # fall back to expanding, past-only calibration
        scaled = scale_and_cap_abs10_expanding(s, cap=cap)
        # early exit because that helper already caps:
        return scaled
    return scaled.clip(-cap, cap)


# === Carver scalar for FAST mean reversion (Strategy 26) ===
MEANREV_EQ_SCALAR = 9.3  # book value for the fixed scalar


def scale_meanrev_eq_subforecast(
    raw: pd.Series, cap: float = 20.0, scalar: float = MEANREV_EQ_SCALAR
) -> pd.Series:
    """
    Carver-style scaling for fast mean reversion:
    scaled = raw * scalar, then cap to +/-20. No dynamic normalisation.
    """
    s = pd.Series(raw, dtype=float)
    return (s * float(scalar)).clip(-cap, cap)


def expanding_abs10_scale(
    raw: pd.Series, cap: float = 20.0, min_history: int = 252
) -> pd.Series:
    """
    If you want a *no-lookahead* normaliser: scale so that expanding median(|forecast|)=10.
    This is optional and NOT used for Strategy 26, but safe for experimentation.
    """
    s = pd.Series(raw, dtype=float)
    med = s.abs().expanding(min_periods=min_history).median().replace(0.0, np.nan)
    k = 10.0 / med
    return (s * k).clip(-cap, cap)


def scale_meanrev_trend_vol_carver(
    raw_subforecast: pd.Series,
    *,
    scalar: float = 20.0,
    cap: float = 20.0,
) -> pd.Series:
    """
    Carver-style scalar for the mean-reversion + trend overlay + vol multiplier rule.
    Scaled forecast = Modified risk-adjusted forecast x 20; then cap to +/-20.
    """
    s = pd.Series(raw_subforecast, copy=True).astype(float)
    s = s * float(scalar)
    return s.clip(-cap, cap)


def scale_meanrev_trend_vol_expand(
    raw_subforecast: pd.Series,
    *,
    target_rms: float = 10.0,
    min_history: int = 252,
    cap: float = 20.0,
    robust: bool = False,
) -> pd.Series:
    """
    Leak-free expanding scaler:
    - computes an expanding volatility estimate of the RAW sub-forecast
      (std or robust MAD-based) using ONLY past data;
    - rescales so RMS ~ `target_rms`;
    - caps to +/-`cap`.

    Notes:
      - Use this when NOT using Carver's constant scalar and you want
        instrument-agnostic scaling learned from history without lookahead.
      - `min_history` delays activation until enough samples are seen.
    """
    x = pd.Series(raw_subforecast, copy=True).astype(float)

    if robust:
        # robust "std" via MAD * 1.4826 (expanding)
        med = x.expanding(min_history).median()
        mad = (x - med).abs().expanding(min_history).median()
        vol = mad * 1.4826
    else:
        vol = x.expanding(min_history).std()

    # Avoid divide-by-zero & lookahead
    k = target_rms / vol.replace(0.0, np.nan)
    k = k.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    scaled = (x * k).clip(-cap, cap)
    # Drop early periods before there's enough history (optional)
    scaled[vol.isna()] = 0.0
    return scaled


def cap_final_forecast(series: pd.Series, cap: float = 20.0) -> pd.Series:
    return pd.Series(series, index=series.index, dtype=float).clip(-cap, cap)


# ---------- Fallback internal forecasts (only if user functions not found) ----------
def _fallback_breakout(price: pd.Series, horizon: int) -> pd.Series:
    # Donchian-style with previous window centre/range (smoothed); safe default
    hi = price.rolling(horizon, min_periods=horizon).max().shift(1)
    lo = price.rolling(horizon, min_periods=horizon).min().shift(1)
    rng = hi - lo
    rng[rng < 1e-8] = np.nan
    centre = (hi + lo) / 2.0
    raw = 40.0 * (price - centre) / rng
    span = max(1, int(horizon / 4))
    return raw.ewm(span=span, adjust=False).mean()


def _fallback_ewma_cross(price: pd.Series, fast: int, slow: int) -> pd.Series:
    f = price.ewm(span=fast, adjust=False).mean()
    s = price.ewm(span=slow, adjust=False).mean()
    spread = f - s
    volp = price.diff().rolling(max(20, slow)).std().shift(1)  # <- shift to t-1
    volp = volp.replace(0.0, np.nan).bfill()
    return (spread / volp).replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ======================================================================================
# NEW: Additional Strategy Forecast Functions
# ======================================================================================


def _sigma_price_points(
    price: pd.Series, vol_span: int = 60, annualize_factor: float | None = None
) -> pd.Series:
    """Helper for new strategies: sigma_p,t in *price points*."""
    p = pd.Series(price, dtype=float)
    ret = p.pct_change()
    var = (
        ret.pow(2)
        .ewm(span=vol_span, adjust=False, min_periods=max(10, vol_span // 5))
        .mean()
    )
    sig_pct = var.pow(0.5)
    s = p * sig_pct
    if annualize_factor is not None:
        s = s * float(annualize_factor)
    return s.replace(0, np.nan)


def calc_ewmac_accel_forecast(
    price: pd.Series, N: int, vol_span: int = 60
) -> pd.Series:
    """Raw EWMAC_N,t = (EWMA(N) - EWMA(4N)) / sigma_p,t ;  accel = EWMAC_t - EWMAC_{t-N}."""
    p = pd.Series(price, dtype=float)
    ewN = p.ewm(span=N, adjust=False).mean()
    ew4N = p.ewm(span=4 * N, adjust=False).mean()
    sigp = _sigma_price_points(p, vol_span=vol_span, annualize_factor=None)
    ewmac_raw = (ewN - ew4N) / sigp
    return ewmac_raw - ewmac_raw.shift(N)


# def calc_meanrev_equilibrium_forecast(price: pd.Series, eq_span: int = 5, vol_span: int = 60) -> pd.Series:
#     """Risk-adjusted mean reversion: (Equilibrium_t - p_t) / sigma_p,t."""
#     p = pd.Series(price, dtype=float)
#     equilibrium = p.ewm(span=eq_span, adjust=False).mean()
#     raw = equilibrium - p
#     sigp = _sigma_price_points(p, vol_span=vol_span, annualize_factor=16.0)
#     return raw / sigp


def calc_meanrev_equilibrium_forecast(
    price: pd.Series, eq_span: int = 5, vol_window: int = 25
) -> pd.Series:
    """
    Carver Strategy 26 (fast mean reversion to equilibrium).
    equilibrium = EWMA_span=eq_span(price)
    raw        = equilibrium - price
    risk_adj   = raw / (price * daily_pct_stdev * 16)
    Return the risk-adjusted forecast (unscaled). Scale after with 9.3 and then cap +/-20.
    """
    p = price.dropna().astype(float)
    # Equilibrium
    eq = p.ewm(span=eq_span, adjust=False).mean()
    raw = eq - p

    # IMPORTANT: daily % stdev (NOT annualized)
    ret = p.pct_change()
    sigma_pct_daily = ret.rolling(vol_window, min_periods=vol_window).std()

    # Carver table: multiply by 16 (not sqrt) to get monthly-ish price scale
    sigma_p = p * (sigma_pct_daily * 16.0)
    risk_adj = raw / sigma_p
    return risk_adj.reindex(price.index)  # align back to original index


def _expanding_percentile_rank(s: pd.Series) -> pd.Series:
    """Helper for vol multiplier: percentile rank of s_t within s_{1..t}."""

    def last_percentile(x: pd.Series) -> float:
        r = x.rank(pct=True)
        return float(r.iloc[-1])

    return pd.Series(s, dtype=float).expanding().apply(last_percentile, raw=False)


def calc_meanrev_trend_volmult_forecast(
    price: pd.Series, rule_params: dict
) -> pd.Series:
    """Mean reversion with trend gating and a volatility multiplier."""
    p = pd.Series(price, dtype=float)
    # Default parameters can be overridden by rule_params dict
    params = {
        "eq_span": 5,
        "vol_span": 60,
        "trend_N": 16,
        "vol_rel_long_days": 2520,
        "vol_mult_ewma_span": 10,
    }
    params.update(rule_params)

    # 1) Base mean-reversion
    equilibrium = p.ewm(span=params["eq_span"], adjust=False).mean()
    raw_mr = equilibrium - p
    sigp_mr = _sigma_price_points(p, vol_span=params["vol_span"], annualize_factor=16.0)
    mr = raw_mr / sigp_mr

    # 2) Trend overlay
    ewN = p.ewm(span=params["trend_N"], adjust=False).mean()
    ew4N = p.ewm(span=4 * params["trend_N"], adjust=False).mean()
    sigp_tr = _sigma_price_points(p, vol_span=params["vol_span"], annualize_factor=None)
    tr_raw = (ewN - ew4N) / sigp_tr
    sign_match = np.sign(mr) * np.sign(tr_raw)
    mr_gated = mr.where((sign_match >= 0) | (~np.isfinite(sign_match)), 0.0)

    # 3) Volatility multiplier
    o_t = p.pct_change().ewm(span=params["vol_span"]).std()
    long_vol = o_t.rolling(window=params["vol_rel_long_days"], min_periods=60).mean()
    V_t = (o_t / long_vol).replace([np.inf, -np.inf], np.nan)
    Q_t = _expanding_percentile_rank(V_t).clip(0.0, 1.0)
    M_t = (2.0 - 1.5 * Q_t).ewm(span=params["vol_mult_ewma_span"], adjust=False).mean()

    return mr_gated * M_t


# ---------- Volatility targeting (self-contained sizer) ----------
def infer_periods_per_day(index: pd.DatetimeIndex) -> int:
    if not isinstance(index, pd.DatetimeIndex) or len(index) == 0:
        return 1
    per_day = pd.Series(1, index=index).groupby(index.normalize()).sum()
    return int(max(1, per_day.median()))


def _prepare_fx(index, fx_series: Optional[pd.Series]) -> pd.Series:
    if fx_series is None:
        return pd.Series(1.0, index=index)
    fx = pd.Series(fx_series).reindex(index).ffill().bfill()
    return fx


def compute_value_volatility(
    price: pd.Series,
    value_per_point: float = 1.0,
    fx_series: Optional[pd.Series] = None,
    lookback: int = 500,
    periods_per_day: Optional[int] = None,
) -> pd.Series:
    price = pd.Series(price).astype(float)
    if periods_per_day is None:
        periods_per_day = infer_periods_per_day(price.index)
    point_vol = (
        price.diff()
        .rolling(lookback, min_periods=max(2, lookback // 2))
        .std()
        .shift(1)
        .bfill()
    )
    fx = _prepare_fx(price.index, fx_series)
    value_vol_per_bar = point_vol * float(value_per_point) * fx
    value_vol_daily = value_vol_per_bar * np.sqrt(max(1, periods_per_day))
    return value_vol_daily


# def compute_volatility_scalar(price: pd.Series,
#                               ann_cash_vol_target: float,
#                               value_per_point: float = 1.0,
#                               fx_series: Optional[pd.Series] = None,
#                               lookback: int = 500,
#                               periods_per_day: Optional[int] = None) -> pd.Series:
#     if periods_per_day is None:
#         periods_per_day = infer_periods_per_day(pd.Series(price).index)
#     value_vol_daily = compute_value_volatility(price, value_per_point=value_per_point,
#                                               fx_series=fx_series, lookback=lookback,
#                                               periods_per_day=periods_per_day)
#     daily_cash_vol_target = float(ann_cash_vol_target) / 16.0  # ~ sqrt(256 business days)
#     vol_scalar = daily_cash_vol_target / value_vol_daily.replace(0.0, np.nan)
#     return pd.Series(vol_scalar, index=pd.Series(price).index)


def _infer_periods_per_day(index) -> float:
    """Crude detector for business-day data; keep at 1 for daily."""
    return 1.0


# ------------------ volatility pieces (past only) ------------------
def _sigma_pct_daily(price: pd.Series, span: int) -> pd.Series:
    """EWMA daily percent vol, shifted 1 day to avoid look-ahead."""
    r = price.pct_change()
    sig = r.ewm(span=span, adjust=False).std()
    return sig.shift(1)  # strictly past


def _sigma_points_for_meanrev(price: pd.Series, vol_span: int) -> pd.Series:
    """
    Carver mean-reversion denominator:
      sigma_p,t = price_t x sigma%_daily,t x 16
    where sigma%_daily,t is an EWMA daily % volatility (past-only).
    """
    sig_pct_daily = _sigma_pct_daily(price, span=vol_span)
    return price * sig_pct_daily * 16.0


# def compute_volatility_scalar(price: pd.Series,
#                               capital: float,
#                               pct_vol_target: float | None,
#                               lookback: int,
#                               value_per_point: float = 1.0,
#                               fx_series: pd.Series | None = None,
#                               periods_per_day: Optional[int] = None,
#                               period_per_day: Optional[int] = None,
#                               **legacy_kwargs) -> pd.Series:
#     """
#     Scalar that converts 1 risk block into contracts:
#       vol_scalar_t = (Capital x target_daily_RU) / sigma_dollar,t
#       sigma_dollar,t   = price_t x FX_t x sigma%_daily,t
#       target_daily_RU = pct_vol_target / sqrt(252)
#     If pct_vol_target is None: fall back to any legacy ann_cash input or return 1.0.

#     `periods_per_day` / `period_per_day` and any legacy kwargs are accepted so older
#     callers do not trigger TypeErrors.
#     """
#     price_series = pd.Series(price).astype(float)

#     periods_hint = periods_per_day if periods_per_day is not None else period_per_day

#     value_vol_daily = compute_value_volatility(
#         price_series,
#         value_per_point=value_per_point,
#         fx_series=fx_series,
#         lookback=lookback,
#         periods_per_day=periods_hint,
#     )
#     denom = value_vol_daily.replace(0.0, np.nan)

#     numer = None
#     if pct_vol_target is not None:
#         target_daily_ru = float(pct_vol_target) / np.sqrt(BUSDAYS)
#         numer = float(capital) * target_daily_ru
#     else:
#         ann_cash = legacy_kwargs.get('ann_cash_vol_target')
#         if ann_cash is not None:
#             numer = float(ann_cash) / np.sqrt(BUSDAYS)
#         else:
#             legacy_capital = legacy_kwargs.get('capital')
#             legacy_pct = legacy_kwargs.get('pct_vol_target')
#             if legacy_capital is not None and legacy_pct is not None:
#                 numer = float(legacy_capital) * float(legacy_pct) / np.sqrt(BUSDAYS)
#     if numer is None:
#         return pd.Series(1.0, index=price_series.index)

#     vol_scalar = (numer / denom).clip(lower=0)
#     return vol_scalar.reindex(price_series.index).fillna(0.0)


def compute_volatility_scalar(
    price: pd.Series,
    ann_cash_vol_target: float | None = None,  # old style
    capital: float | None = None,  # new style
    pct_vol_target: float | None = None,  # new style
    lookback: int = 500,
    value_per_point: float = 1.0,
    fx_series: pd.Series | None = None,
    periods_per_day: int | None = None,  # new kwarg (from caller)
    period_per_day: int | None = None,  # tolerate alias
    **_legacy_kwargs,  # swallow any stragglers
) -> pd.Series:
    price = pd.Series(price).astype(float)

    # Allow either hint for bars per day
    periods_hint = periods_per_day if periods_per_day is not None else period_per_day

    # Dollar vol per day (past-only)
    value_vol_daily = compute_value_volatility(
        price,
        value_per_point=value_per_point,
        fx_series=fx_series,
        lookback=lookback,
        periods_per_day=periods_hint,
    )
    denom = value_vol_daily.replace(0.0, np.nan)

    # Determine numerator (target daily cash vol)
    numer = None
    if ann_cash_vol_target is not None:
        # old API: annual cash vol -> daily using ~sqrt(256) ~ 16
        numer = float(ann_cash_vol_target) / 16.0
    elif capital is not None and pct_vol_target is not None:
        # new API: capital x (daily risk units)
        numer = float(capital) * (float(pct_vol_target) / np.sqrt(BUSDAYS))
    else:
        # tolerate old kwarg names if passed in **_legacy_kwargs
        legacy_ann = _legacy_kwargs.get("ann_cash_vol_target")
        legacy_cap = _legacy_kwargs.get("capital")
        legacy_pct = _legacy_kwargs.get("pct_vol_target")
        if legacy_ann is not None:
            numer = float(legacy_ann) / 16.0
        elif legacy_cap is not None and legacy_pct is not None:
            numer = float(legacy_cap) * (float(legacy_pct) / np.sqrt(BUSDAYS))

    if numer is None:
        return pd.Series(1.0, index=price.index)  # harmless fallback

    vol_scalar = (numer / denom).clip(lower=0)
    return vol_scalar.reindex(price.index).fillna(0.0)


# def apply_carver_position_sizing(signal: pd.Series | np.ndarray,
#                                  price: pd.Series,
#                                  capital: float,
#                                  pct_vol_target: float | None,
#                                  value_per_point: float = 1.0,
#                                  fx_series: Optional[pd.Series] = None,
#                                  lookback: int = 500,
#                                  periods_per_day: Optional[int] = None,
#                                  period_per_day: Optional[int] = None,
#                                  **_legacy_kwargs) -> pd.Series:
#     price = pd.Series(price).astype(float)
#     sig = pd.Series(signal, index=price.index).astype(float)  # assume already capped/combined
#     vol_scalar = compute_volatility_scalar(
#         price=price,
#         capital=float(capital),
#         pct_vol_target=None if pct_vol_target is None else float(pct_vol_target),
#         lookback=lookback,
#         value_per_point=value_per_point,
#         fx_series=fx_series,
#         periods_per_day=periods_per_day,
#         period_per_day=period_per_day,
#         **_legacy_kwargs,
#     )
#     pos = vol_scalar * (sig / 10.0)
#     return pos.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
def apply_carver_position_sizing(
    signal: pd.Series | np.ndarray,
    price: pd.Series,
    ann_cash_vol_target: float | None = None,  # old style supported
    capital: float | None = None,  # new style supported
    pct_vol_target: float | None = None,  # new style supported
    value_per_point: float = 1.0,
    fx_series: pd.Series | None = None,
    lookback: int = 500,
    periods_per_day: int | None = None,
    period_per_day: int | None = None,
    **_legacy_kwargs,
) -> pd.Series:
    price = pd.Series(price).astype(float)
    sig = pd.Series(signal, index=price.index).astype(float)

    vol_scalar = compute_volatility_scalar(
        price=price,
        ann_cash_vol_target=ann_cash_vol_target,
        capital=capital,
        pct_vol_target=pct_vol_target,
        lookback=lookback,
        value_per_point=value_per_point,
        fx_series=fx_series,
        periods_per_day=periods_per_day,
        period_per_day=period_per_day,
        **_legacy_kwargs,
    )
    pos = vol_scalar * (sig / 10.0)
    return pos.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


# ---------- PnL & Sharpe ----------
# def _pnl_sharpe_from_positions(
#     price: pd.Series,
#     positions_blocks: pd.Series,
#     value_per_point: float,
#     fx_series: Optional[pd.Series],
#     capital: float,
# ) -> Tup[pd.Series, float]:
#     idx = price.index
#     fx = (pd.Series(1.0, index=idx) if fx_series is None
#           else pd.Series(fx_series).reindex(idx).ffill().bfill())
#     price_diff = price.diff()
#     pnl_ccy = positions_blocks.shift(1) * price_diff * float(value_per_point) * fx
#     ret = pnl_ccy / float(capital)
#     ret = ret.replace([np.inf, -np.inf], np.nan).dropna()
#     if ret.std(ddof=0) == 0 or ret.empty:
#         return ret, 0.0
#     sharpe = (ret.mean() / ret.std(ddof=0)) * np.sqrt(BUSDAYS)
#     return ret, float(sharpe)


def _pnl_sharpe_from_positions(
    price: pd.Series,
    positions_blocks: pd.Series,
    value_per_point: float,
    fx_series: Optional[pd.Series],
    capital: float,
    *,
    cost_per_contract_point: float = 0.0,  # e.g. 0.5 ticks -> 0.5 if 1 point=1 tick
    cost_bps_notional: float = 0.0,  # alternative: bps of traded notional
    deadband_blocks: float = 0.0,  # e.g. 0.1 -> ignore trades <0.1 blocks
) -> Tup[pd.Series, float]:
    idx = price.index
    fx = (
        pd.Series(1.0, index=idx)
        if fx_series is None
        else pd.Series(fx_series).reindex(idx).ffill().bfill()
    )

    price_diff = price.diff()
    # Base PnL (value units)
    pnl_ccy = positions_blocks.shift(1) * price_diff * float(value_per_point) * fx

    # --- trading costs ---
    dpos = positions_blocks.diff().abs().fillna(0.0)
    if deadband_blocks > 0:
        dpos = dpos.where(dpos >= float(deadband_blocks), 0.0)

    # cost in "points x value_per_point"
    cost_points = dpos * float(cost_per_contract_point)
    cost_val = cost_points * float(value_per_point) * fx

    # optional: bps of traded notional per bar
    notional_traded = dpos * price * float(value_per_point) * fx
    cost_val += notional_traded * (float(cost_bps_notional) / 10_000.0)

    pnl_net = pnl_ccy - cost_val

    ret = pnl_net / float(capital)
    ret = ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if ret.std(ddof=0) == 0 or ret.empty:
        return ret, 0.0
    sharpe = (ret.mean() / ret.std(ddof=0)) * np.sqrt(BUSDAYS)
    return ret, float(sharpe)


# ---------- Price alignment ----------
def _align_prices(price_frame: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    pf = price_frame.copy()
    if not isinstance(pf.index, pd.DatetimeIndex):
        pf.index = pd.to_datetime(pf.index)
    pf = pf.sort_index()
    cols = [t for t in tickers if t in pf.columns]
    return pf[cols].astype(float).ffill().bfill()


# ---------- Combine forecasts ----------
def _combine_weighted_forecasts(
    sub_forecasts: List[pd.Series], weights: np.ndarray
) -> pd.Series:
    w = np.array(weights, dtype=float)
    w = np.maximum(w, 0.0)
    if w.sum() == 0:
        w = np.ones_like(w)
    w = w / w.sum()
    base_index = sub_forecasts[0].index
    combo = sum(
        w[i] * sub_forecasts[i].reindex(base_index).fillna(0.0) for i in range(len(w))
    )
    return pd.Series(combo, index=base_index, dtype=float)


# ---------- Backtest per ticker (Carver sizing) ----------


## With trading frictions
def _backtest_ticker_carver(
    price: pd.Series,
    sub_forecasts: list[pd.Series],
    weights: np.ndarray,
    *,
    capital: float,
    pct_vol_target: float | None,
    vol_lookback_for_position_sizing: int,
    value_per_point: float = 1.0,
    fx_series: pd.Series | None = None,
    # NEW: trading frictions
    buffer_risk_units: float = 0.0,  # Carver deadband in blocks
    commission_bps: float = 0.0,  # per side (or roundtrip if you prefer)
    spread_bps: float = 0.0,  # half-spread * 2 ~ roundtrip bps
    cost_per_contract_point: float = 0.0,  # if you model a fixed tick cost
) -> tuple[pd.Series, float]:
    """
    Builds combined forecast -> positions -> net PnL & Sharpe.
    """
    # ---- combine & two-stage cap ----
    # (assume sub_forecasts already scaled & capped +/-20 individually)
    aligned = [sf.reindex(price.index).fillna(0.0) for sf in sub_forecasts]
    w = np.array(weights, dtype=float)
    w = w / (w.sum() if w.sum() != 0 else 1.0)
    combined = np.tensordot(np.vstack(aligned).T, w, axes=(1, 0))
    # final cap +/-20
    combined = pd.Series(combined, index=price.index).clip(-20.0, 20.0)

    # ---- position sizing in blocks (vol-targeted) ----
    # sigma% is past-only; sizing uses t info, trades at t+1 (in PnL fn)
    ret = price.pct_change()
    sig_pct = (
        ret.ewm(span=vol_lookback_for_position_sizing, adjust=False).std().shift(1)
    )
    sig_ann = sig_pct * np.sqrt(BUSDAYS)
    # sigma_dollar = price * (fx_series.ffill() if fx_series is not None else 1.0) * sig_ann
    sigma_dollar = (
        price
        * sig_ann
        * float(value_per_point)
        * (fx_series.ffill() if fx_series is not None else 1.0)
    )

    # blocks that would deliver target vol if forecast=10
    if pct_vol_target is None:
        vol_scalar = pd.Series(1.0, index=price.index)
    else:
        target_ru = float(pct_vol_target) / np.sqrt(BUSDAYS)  # daily risk units
        numer = capital * target_ru
        denom = sigma_dollar.replace(0.0, np.nan)
        vol_scalar = (numer / denom).clip(lower=0).fillna(0.0)

    # map forecast (+/-20) to blocks via /10 rule
    positions_blocks = vol_scalar * (combined / 10.0)

    # ---- NET PnL (commissions, spread, deadband) ----
    returns, sharpe = _pnl_sharpe_from_positions(
        price=price,
        positions_blocks=positions_blocks,
        value_per_point=value_per_point,
        fx_series=fx_series,
        capital=float(capital),
        # combine bps costs; you can separate if your PnL fn has two knobs
        cost_bps_notional=float(commission_bps + spread_bps),
        cost_per_contract_point=float(cost_per_contract_point),
        deadband_blocks=float(buffer_risk_units),
    )
    return returns, sharpe


# ---------- Public: Breakout optimizer (two-stage cap) ----------
# =========================
# OPTUNA optimizers (maximize Sharpe)
# =========================

# =========================
# Quiet Optuna helpers
# =========================
from contextlib import contextmanager
import inspect


@contextmanager
def _optuna_quiet(silent: bool = True):
    import optuna

    if not silent:
        yield
        return
    prev = optuna.logging.get_verbosity()
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        yield
    finally:
        optuna.logging.set_verbosity(prev)


def _study_optimize_quiet(
    study, objective, n_trials: int, n_jobs: int | None, show_progress_bar: bool
):
    """Call Study.optimize without printing progress bars, but stay compatible with older Optuna."""
    sig = inspect.signature(study.optimize)
    kwargs = {"n_trials": int(n_trials)}
    if n_jobs is not None and "n_jobs" in sig.parameters:
        kwargs["n_jobs"] = int(n_jobs)
    if "show_progress_bar" in sig.parameters:
        kwargs["show_progress_bar"] = bool(show_progress_bar)
    study.optimize(objective, **kwargs)


from collections.abc import Mapping
import numpy as np
import pandas as pd


def _is_num(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating))


def _get(source, key: str, default=None):
    """
    Return a per-ticker parameter from `source`.

    `source` may be:
      - scalar number (applied to all tickers)
      - dict-like mapping {ticker -> value}
      - pandas Series indexed by ticker
      - pandas DataFrame with one column and index=ticker
      - callable(key) -> value
      - None

    Numeric-like values are returned as float; otherwise returned as-is.
    """
    if source is None:
        return default

    # scalar applied to all tickers
    if _is_num(source):
        return float(source)

    # callable
    if callable(source):
        try:
            val = source(key)
            return float(val) if _is_num(val) else val
        except Exception:
            return default

    # mapping / dict
    if isinstance(source, Mapping):
        val = source.get(key, default)
        return float(val) if _is_num(val) else val

    # pandas Series
    if isinstance(source, pd.Series):
        val = source.get(key, default)
        return float(val) if _is_num(val) else val

    # single-column DataFrame indexed by ticker
    if isinstance(source, pd.DataFrame):
        if key in source.index:
            val = source.loc[key]
            # squeeze single cell/row
            try:
                val = val.squeeze()
            except Exception:
                pass
            return float(val) if _is_num(val) else val
        return default

    # fallback
    return default


def _get_fx_series(fx_series_map, key: str):
    """
    Return an FX series (pd.Series) for ticker `key` if provided.
    Accepts dict-like {ticker -> Series} or None. Otherwise returns None.
    """
    if fx_series_map is None:
        return None
    if isinstance(fx_series_map, Mapping):
        return fx_series_map.get(key, None)
    # Add other shapes if you later support them
    return None


# =========================
# OPTUNA optimizer: Breakout (quiet)
# =========================
def optimize_breakout_weights_carver(
    price_frame: pd.DataFrame,
    tickers: list[str],
    breakout_horizons: list[int],
    vol_lookback_for_position_sizing: int = 500,
    n_trials_per_ticker: int = 500,
    capital: float = 1_000_000.0,
    pct_vol_target: float = 0.20,
    value_per_point_map: dict[str, float] | None = None,
    fx_map: dict[str, pd.Series] | None = None,
    random_seed: int = 42,
    external_breakout_forecast_fn=None,
    # fixed-scalar option
    use_carver_scalars: bool = False,
    carver_breakout_scalars: dict[int, float] | None = None,
    # NEW: quiet controls
    silent: bool = True,
    n_jobs: int | None = None,  # parallelize trials within this process
    show_progress_bar: bool = False,  # keep off to avoid terminal spam
) -> dict[str, dict]:
    """
    Optuna optimizer for breakout sub-forecast weights (per ticker), maximizing Sharpe.
    Quiet by default (no per-trial prints).
    Returns: {ticker: {"best_weights": [...], "sharpe": float, "horizons": [...]}}.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except Exception as e:
        raise ImportError("optuna is required: pip install optuna") from e

    pf = _align_prices(price_frame, tickers)
    results: dict[str, dict] = {}
    sampler = TPESampler(seed=random_seed)

    user_breakout_fn = external_breakout_forecast_fn or _try_resolve(
        "calc_breakout_forecast"
    )

    for t in tickers:
        px = pf[t]

        # Build raw -> scaled sub-forecasts
        subs_raw = []
        for h in breakout_horizons:
            sf = (
                user_breakout_fn(px, int(h))
                if user_breakout_fn is not None
                else _fallback_breakout(px, int(h))
            )
            subs_raw.append(pd.Series(sf, index=px.index, dtype=float))

        if use_carver_scalars and "scale_breakout_subforecast" in globals():
            subs = [
                scale_breakout_subforecast(
                    sf,
                    int(h),
                    use_carver_scalars=True,
                    scalars_dict=carver_breakout_scalars,
                    cap=20.0,
                )
                for sf, h in zip(subs_raw, breakout_horizons)
            ]
        else:
            subs = [scale_and_cap_subforecast_abs10(sf, cap=20.0) for sf in subs_raw]

        vpp = 1.0 if not value_per_point_map else float(value_per_point_map.get(t, 1.0))
        fx_series = None if not fx_map else fx_map.get(t, None)

        def objective(trial: "optuna.trial.Trial") -> float:
            w = np.array(
                [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(subs))],
                dtype=float,
            )
            if w.sum() == 0.0:
                w[:] = 1.0
            _, sr = _backtest_ticker_carver(
                price=px,
                sub_forecasts=subs,
                weights=w,
                capital=capital,
                pct_vol_target=pct_vol_target,
                vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
                value_per_point=vpp,
                fx_series=fx_series,
            )
            if not np.isfinite(sr):
                return -1e12
            return float(sr)

        study = optuna.create_study(direction="maximize", sampler=sampler)
        with _optuna_quiet(silent):
            _study_optimize_quiet(
                study,
                objective,
                n_trials=n_trials_per_ticker,
                n_jobs=n_jobs,
                show_progress_bar=show_progress_bar,
            )

        # Normalize and store best weights
        best_params = study.best_trial.params
        raw = np.array(
            [best_params.get(f"w_{i}", 0.0) for i in range(len(subs))], dtype=float
        )
        if raw.sum() == 0.0:
            raw[:] = 1.0
        best_w = (raw / raw.sum()).tolist()

        results[t] = {
            "best_weights": best_w,
            "sharpe": float(study.best_value),
            "horizons": list(map(int, breakout_horizons)),
        }

    return results


def optimize_breakout_weights_optuna_fullsample(
    price_frame: pd.DataFrame,
    tickers: List[str],
    breakout_horizons: List[int],
    *,
    vol_lookback_for_position_sizing: int,
    capital: float = 1_000_000,
    pct_vol_target: float = 0.20,
    use_carver_scalars: bool = True,  # default: Carver scalars (no scaling leak)
    scalar_min_history: int = 252,  # used if use_carver_scalars=False
    commission_bps_map: Dict[str, float] | float = 0.0,
    spread_bps_map: Dict[str, float] | float = 0.0,
    buffer_risk_units_map: Dict[str, float] | float = 0.0,
    value_per_point_map: Dict[str, float] | float = 1.0,
    fx_series_map: Dict[str, pd.Series] | None = None,
    n_trials_per_ticker: int = 200,
    n_jobs: int | None = None,
    silent: bool = True,
    show_progress_bar: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """
    Optimizes convex weights across breakout horizons on the full sample, using leak-free scaling.
    Returns {ticker: {"best_weights": [...], "sharpe": float, "horizons": [...]}}.
    """
    if silent:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    results: Dict[str, Dict[str, Any]] = {}
    H = [int(h) for h in breakout_horizons]

    for t in tickers:
        if t not in price_frame.columns:
            continue
        px = price_frame[t].dropna()
        if len(px) < max(H) + 30:
            continue

        def _objective(trial: optuna.Trial) -> float:
            w = np.array(
                [trial.suggest_float(f"w_{h}", 0.0, 1.0) for h in H], dtype=float
            )
            if w.sum() == 0:
                w[0] = 1.0
            w = w / w.sum()

            subs = []
            for h in H:
                raw = calc_breakout_forecast(px, h)  # must use .shift(1) windows inside
                if use_carver_scalars:
                    sf = _scale_carver_breakout(raw, h)
                else:
                    sf = scale_and_cap_subforecast_abs10(
                        raw, cap=20.0, min_periods=scalar_min_history
                    )
                subs.append(sf)

            ret, sr = _backtest_ticker_carver(
                price=px,
                sub_forecasts=subs,
                weights=w,
                capital=capital,
                pct_vol_target=pct_vol_target,
                vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
                value_per_point=_get(value_per_point_map, t, 1.0),
                fx_series=None if fx_series_map is None else fx_series_map.get(t),
                buffer_risk_units=_get(buffer_risk_units_map, t, 0.0),
                commission_bps=_get(commission_bps_map, t, 0.0),
                spread_bps=_get(spread_bps_map, t, 0.0),
            )
            return float(sr if np.isfinite(sr) else -1e9)

        study = optuna.create_study(direction="maximize")
        study.optimize(
            _objective,
            n_trials=n_trials_per_ticker,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
        )
        best_w = np.array(
            [study.best_params.get(f"w_{h}", 1.0) for h in H], dtype=float
        )
        best_w = best_w / (best_w.sum() if best_w.sum() != 0 else 1.0)

        # final score with best weights
        subs_best = []
        for h in H:
            raw = calc_breakout_forecast(px, h)
            sf = (
                _scale_carver_breakout(raw, h)
                if use_carver_scalars
                else scale_and_cap_subforecast_abs10(raw, 20.0, scalar_min_history)
            )
            subs_best.append(sf)
        _, best_sr = _backtest_ticker_carver(
            price=px,
            sub_forecasts=subs_best,
            weights=best_w,
            capital=capital,
            pct_vol_target=pct_vol_target,
            vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
            value_per_point=_get(value_per_point_map, t, 1.0),
            fx_series=None if fx_series_map is None else fx_series_map.get(t),
            buffer_risk_units=_get(buffer_risk_units_map, t, 0.0),
            commission_bps=_get(commission_bps_map, t, 0.0),
            spread_bps=_get(spread_bps_map, t, 0.0),
        )

        results[t] = {
            "best_weights": best_w.tolist(),
            "sharpe": float(best_sr),
            "horizons": H,
        }
    return results


# =========================
# OPTUNA optimizer: EWMA cross (quiet)
# =========================
def optimize_ewma_cross_weights_carver(
    price_frame: pd.DataFrame,
    tickers: list[str],
    ewma_pairs: list[tuple[int, int]],
    vol_lookback_for_position_sizing: int = 500,
    n_trials_per_ticker: int = 500,
    capital: float = 1_000_000.0,
    pct_vol_target: float = 0.20,
    value_per_point_map: dict[str, float] | None = None,
    fx_map: dict[str, pd.Series] | None = None,
    random_seed: int = 42,
    external_ewma_forecast_fn=None,
    # NEW: quiet controls
    silent: bool = True,
    n_jobs: int | None = None,
    show_progress_bar: bool = False,
) -> dict[str, dict]:
    """
    Optuna optimizer for EWMA-cross sub-forecast weights (per ticker), maximizing Sharpe.
    Quiet by default (no per-trial prints).
    Returns: {ticker: {"best_weights": [...], "sharpe": float, "pairs": [[fast,slow], ...]}}.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except Exception as e:
        raise ImportError("optuna is required: pip install optuna") from e

    pf = _align_prices(price_frame, tickers)
    results: dict[str, dict] = {}
    sampler = TPESampler(seed=random_seed)

    user_ewma_fn = external_ewma_forecast_fn or _try_resolve("calc_ewma_cross_forecast")

    for t in tickers:
        px = pf[t]

        subs_raw = []
        for f, s in ewma_pairs:
            sf = (
                user_ewma_fn(px, int(f), int(s))
                if user_ewma_fn is not None
                else _fallback_ewma_cross(px, int(f), int(s))
            )
            subs_raw.append(pd.Series(sf, index=px.index, dtype=float))

        subs = [scale_and_cap_subforecast_abs10(sf, cap=20.0) for sf in subs_raw]

        vpp = 1.0 if not value_per_point_map else float(value_per_point_map.get(t, 1.0))
        fx_series = None if not fx_map else fx_map.get(t, None)

        def objective(trial: "optuna.trial.Trial") -> float:
            w = np.array(
                [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(subs))],
                dtype=float,
            )
            if w.sum() == 0.0:
                w[:] = 1.0
            _, sr = _backtest_ticker_carver(
                price=px,
                sub_forecasts=subs,
                weights=w,
                capital=capital,
                pct_vol_target=pct_vol_target,
                vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
                value_per_point=vpp,
                fx_series=fx_series,
            )
            if not np.isfinite(sr):
                return -1e12
            return float(sr)

        study = optuna.create_study(direction="maximize", sampler=sampler)
        with _optuna_quiet(silent):
            _study_optimize_quiet(
                study,
                objective,
                n_trials=n_trials_per_ticker,
                n_jobs=n_jobs,
                show_progress_bar=show_progress_bar,
            )

        best_params = study.best_trial.params
        raw = np.array(
            [best_params.get(f"w_{i}", 0.0) for i in range(len(subs))], dtype=float
        )
        if raw.sum() == 0.0:
            raw[:] = 1.0
        best_w = (raw / raw.sum()).tolist()

        results[t] = {
            "best_weights": best_w,
            "sharpe": float(study.best_value),
            "pairs": [list(map(int, p)) for p in ewma_pairs],
        }

    return results


# =========================
# NEW: OPTUNA optimizers for additional strategies
# =========================
def _create_generic_optimizer(
    strategy_forecast_func, rule_arg_name, result_key_name, strat_lower="vanilla"
):
    """A factory to create optimizer functions for different strategies."""

    def generic_optimizer(
        price_frame: pd.DataFrame,
        tickers: list[str],
        rule_variations: list,
        vol_lookback_for_position_sizing: int = 500,
        n_trials_per_ticker: int = 250,
        capital: float = 1_000_000.0,
        pct_vol_target: float = 0.20,
        value_per_point_map: dict[str, float] | None = None,
        fx_map: dict[str, pd.Series] | None = None,
        random_seed: int = 42,
        silent: bool = True,
        n_jobs: int | None = None,
        show_progress_bar: bool = False,
        **strategy_kwargs,
    ) -> dict[str, dict]:
        try:
            import optuna
            from optuna.samplers import TPESampler
        except ImportError as e:
            raise ImportError("optuna is required: pip install optuna") from e

        pf = _align_prices(price_frame, tickers)
        results: dict[str, dict] = {}
        sampler = TPESampler(seed=random_seed)

        for t in tickers:
            px = pf[t]
            subs_raw = [
                strategy_forecast_func(px, **{rule_arg_name: rv, **strategy_kwargs})
                for rv in rule_variations
            ]
            # subs = [scale_and_cap_subforecast_abs10(sf, cap=20.0) for sf in subs_raw]
            if strat_lower == "meanrev_eq":
                # >>> FIX: use Carver fixed scalar 9.3, not dynamic abs-10
                subs = [scale_meanrev_eq_subforecast(sf, cap=20.0) for sf in subs_raw]
            else:
                # keep generic abs-10 scaler for other strategies, if you wish
                subs = [
                    scale_and_cap_subforecast_abs10(sf, cap=20.0) for sf in subs_raw
                ]

            vpp = value_per_point_map.get(t, 1.0) if value_per_point_map else 1.0
            fx_series = fx_map.get(t) if fx_map else None

            def objective(trial: "optuna.trial.Trial") -> float:
                w = np.array(
                    [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(subs))]
                )
                if w.sum() == 0.0:
                    w[:] = 1.0
                _, sr = _backtest_ticker_carver(
                    price=px,
                    sub_forecasts=subs,
                    weights=w,
                    capital=capital,
                    pct_vol_target=pct_vol_target,
                    vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
                    value_per_point=vpp,
                    fx_series=fx_series,
                )
                return float(sr) if np.isfinite(sr) else -1e12

            study = optuna.create_study(direction="maximize", sampler=sampler)
            with _optuna_quiet(silent):
                _study_optimize_quiet(
                    study, objective, n_trials_per_ticker, n_jobs, show_progress_bar
                )

            best_params = study.best_trial.params
            raw = np.array([best_params.get(f"w_{i}", 0.0) for i in range(len(subs))])
            if raw.sum() == 0.0:
                raw[:] = 1.0
            best_w = (raw / raw.sum()).tolist()

            results[t] = {
                "best_weights": best_w,
                "sharpe": float(study.best_value),
                result_key_name: rule_variations,
            }
        return results

    return generic_optimizer


def optimize_meanrev_eq_optuna_fs(
    price_frame: pd.DataFrame,
    tickers: List[str],
    mr_eq_spans: List[int] = (5,),
    *,
    vol_lookback_for_position_sizing: int,
    capital: float = 1_000_000,
    pct_vol_target: float = 0.20,
    use_carver_scalars: bool = True,  # True -> scalar 9.3 (book); False -> expanding scaler
    scalar_min_history: int = 252,
    commission_bps_map: Dict[str, float] | float = 0.0,
    spread_bps_map: Dict[str, float] | float = 0.0,
    buffer_risk_units_map: Dict[str, float] | float = 0.0,
    value_per_point_map: Dict[str, float] | float = 1.0,
    fx_map: Dict[str, pd.Series] | None = None,
    n_trials_per_ticker: int = 80,  # if >1 spans, we learn weights
    n_jobs: int | None = None,
    silent: bool = True,
    show_progress_bar: bool = False,
) -> Dict[str, Dict[str, Any]]:
    if silent:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    S = [int(s) for s in mr_eq_spans]
    results: Dict[str, Dict[str, Any]] = {}

    for t in tickers:
        if t not in price_frame.columns:
            continue
        px = price_frame[t].dropna()
        if len(px) < max(S) + 60:
            continue

        def _objective(trial: optuna.Trial) -> float:
            if len(S) == 1:
                w = np.array([1.0])
            else:
                w = np.array(
                    [trial.suggest_float(f"w_{s}", 0.0, 1.0) for s in S], dtype=float
                )
                if w.sum() == 0:
                    w[0] = 1.0
                w = w / w.sum()

            subs = []
            for sspan in S:
                raw = calc_meanrev_equilibrium_forecast(
                    px, eq_span=sspan
                )  # risk-adjusted raw (book)
                sf = (
                    scale_meanrev_eq_subforecast(raw)
                    if use_carver_scalars
                    else scale_and_cap_subforecast_abs10(raw, 20.0, scalar_min_history)
                )
                subs.append(sf)

            _, sr = _backtest_ticker_carver(
                price=px,
                sub_forecasts=subs,
                weights=w,
                capital=capital,
                pct_vol_target=pct_vol_target,
                vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
                value_per_point=_get(value_per_point_map, t, 1.0),
                fx_series=None if fx_map is None else fx_map.get(t),
                buffer_risk_units=_get(buffer_risk_units_map, t, 0.0),
                commission_bps=_get(commission_bps_map, t, 0.0),
                spread_bps=_get(spread_bps_map, t, 0.0),
            )
            return float(sr if np.isfinite(sr) else -1e9)

        study = optuna.create_study(direction="maximize")
        total_trials = n_trials_per_ticker if len(S) > 1 else 1
        _study_optimize_quiet(
            study, _objective, total_trials, n_jobs, show_progress_bar
        )

        if len(S) == 1:
            best_w = np.array([1.0])
        else:
            best_w = np.array(
                [study.best_params.get(f"w_{s}", 1.0) for s in S], dtype=float
            )
            best_w = best_w / (best_w.sum() if best_w.sum() != 0 else 1.0)

        subs_best = []
        for sspan in S:
            raw = calc_meanrev_equilibrium_forecast(px, eq_span=sspan)
            sf = (
                scale_meanrev_eq_subforecast(raw)
                if use_carver_scalars
                else scale_and_cap_subforecast_abs10(raw, 20.0, scalar_min_history)
            )
            subs_best.append(sf)
        _, best_sr = _backtest_ticker_carver(
            price=px,
            sub_forecasts=subs_best,
            weights=best_w,
            capital=capital,
            pct_vol_target=pct_vol_target,
            vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
            value_per_point=_get(value_per_point_map, t, 1.0),
            fx_series=None if fx_map is None else fx_map.get(t),
            buffer_risk_units=_get(buffer_risk_units_map, t, 0.0),
            commission_bps=_get(commission_bps_map, t, 0.0),
            spread_bps=_get(spread_bps_map, t, 0.0),
        )
        results[t] = {
            "best_weights": best_w.tolist(),
            "sharpe": float(best_sr),
            "spans": S,
        }
    return results


def optimize_meanrev_trend_vol_fs(
    price_frame: pd.DataFrame,
    tickers: List[str],
    rule_param_list: List[
        Dict[str, Any]
    ],  # e.g. [{"fast":16,"slow":64,"vol_window":10}, ...]
    *,
    vol_lookback_for_position_sizing: int,
    capital: float = 1_000_000,
    pct_vol_target: float = 0.20,
    use_carver_scalars: bool = True,  # True->scalar 20; False->expanding scaler
    scalar_min_history: int = 252,
    commission_bps_map: Dict[str, float] | float = 0.0,
    spread_bps_map: Dict[str, float] | float = 0.0,
    buffer_risk_units_map: Dict[str, float] | float = 0.0,
    value_per_point_map: Dict[str, float] | float = 1.0,
    fx_series_map: Dict[str, pd.Series] | None = None,
    n_trials_per_ticker: int = 80,
    n_jobs: int | None = None,
    silent: bool = True,
    show_progress_bar: bool = False,
) -> Dict[str, Dict[str, Any]]:
    if silent:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    R = (
        rule_param_list
        if rule_param_list
        else [{"fast": 16, "slow": 64, "vol_window": 10}]
    )
    results: Dict[str, Dict[str, Any]] = {}

    for t in tickers:
        if t not in price_frame.columns:
            continue
        px = price_frame[t].dropna()
        if len(px) < 200:
            continue

        def _objective(trial: optuna.Trial) -> float:
            if len(R) == 1:
                w = np.array([1.0])
            else:
                w = np.array(
                    [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(R))],
                    dtype=float,
                )
                if w.sum() == 0:
                    w[0] = 1.0
                w = w / w.sum()

            subs = []
            for params in R:
                raw = calc_meanrev_trend_volmult_forecast(
                    px, rule_params=params
                )  # your implementation
                sf = (
                    scale_meanrev_trend_vol_carver(raw)
                    if use_carver_scalars
                    else scale_and_cap_subforecast_abs10(raw, 20.0, scalar_min_history)
                )
                subs.append(sf)

            _, sr = _backtest_ticker_carver(
                price=px,
                sub_forecasts=subs,
                weights=w,
                capital=capital,
                pct_vol_target=pct_vol_target,
                vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
                value_per_point=_get(value_per_point_map, t, 1.0),
                fx_series=None if fx_series_map is None else fx_series_map.get(t),
                buffer_risk_units=_get(buffer_risk_units_map, t, 0.0),
                commission_bps=_get(commission_bps_map, t, 0.0),
                spread_bps=_get(spread_bps_map, t, 0.0),
            )
            return float(sr if np.isfinite(sr) else -1e9)

        study = optuna.create_study(direction="maximize")
        total_trials = n_trials_per_ticker if len(R) > 1 else 1
        _study_optimize_quiet(
            study, _objective, total_trials, n_jobs, show_progress_bar
        )

        best_w = (
            np.array([1.0])
            if len(R) == 1
            else np.array(
                [study.best_params.get(f"w_{i}", 1.0) for i in range(len(R))],
                dtype=float,
            )
        )
        best_w = best_w / (best_w.sum() if best_w.sum() != 0 else 1.0)

        subs_best = []
        for params in R:
            raw = calc_meanrev_trend_volmult_forecast(px, rule_params=params)
            sf = (
                scale_meanrev_trend_vol_carver(raw)
                if use_carver_scalars
                else scale_and_cap_subforecast_abs10(raw, 20.0, scalar_min_history)
            )
            subs_best.append(sf)
        _, best_sr = _backtest_ticker_carver(
            price=px,
            sub_forecasts=subs_best,
            weights=best_w,
            capital=capital,
            pct_vol_target=pct_vol_target,
            vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
            value_per_point=_get(value_per_point_map, t, 1.0),
            fx_series=None if fx_series_map is None else fx_series_map.get(t),
            buffer_risk_units=_get(buffer_risk_units_map, t, 0.0),
            commission_bps=_get(commission_bps_map, t, 0.0),
            spread_bps=_get(spread_bps_map, t, 0.0),
        )
        results[t] = {
            "best_weights": best_w.tolist(),
            "sharpe": float(best_sr),
            "rules": R,
        }
    return results


# Note: The trend/vol strategy is more complex and might need a custom optimizer if more params are varied.
# This generic one assumes 'rule_variations' maps to a single changing parameter.
# optimize_meanrev_trend_vol_weights_carver = _create_generic_optimizer(
#     strategy_forecast_func=lambda price, rule_params: calc_meanrev_trend_volmult_forecast(price, rule_params),
#     rule_arg_name='rule_params',
#     result_key_name='rule_params_list'
# )


# =========================
# Unified signal -> sizing -> PnL generator (with email_df)
# =========================
class DummyEmailer:
    def __init__(self):
        self.lines = []

    def add_line(self, text: str):
        self.lines.append(str(text))

    def summary(self) -> str:
        return "\n".join(self.lines)


CARVER_BREAKOUT_SCALARS = {
    10: 0.60,
    20: 0.67,
    40: 0.70,
    80: 0.73,
    160: 0.74,
    320: 0.74,
}


def _interp_breakout_scalar(horizon: int, table: dict[int, float]) -> float:
    import math

    h = int(horizon)
    keys = sorted(int(k) for k in table.keys())
    if h <= keys[0]:
        return float(table[keys[0]])
    if h >= keys[-1]:
        return float(table[keys[-1]])
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if a <= h <= b:
            wa = (math.log(h) - math.log(a)) / (math.log(b) - math.log(a))
            return float(table[a] * (1.0 - wa) + table[b] * wa)
    return 1.0


def scale_breakout_subforecast(
    sf, horizon, use_carver_scalars=False, scalars_dict=None, cap: float = 20.0
):
    s = pd.Series(sf, index=sf.index, dtype=float)
    if use_carver_scalars:
        table = scalars_dict or CARVER_BREAKOUT_SCALARS
        mult = _interp_breakout_scalar(int(horizon), table)
        return (s * float(mult)).clip(-cap, cap)
    else:
        return scale_and_cap_subforecast_abs10(s, cap=cap)


def carver_gen_signal_unified(
    price_frame,
    tickers,
    ticker_dict=None,
    optimized_inputs_dict=None,
    rule_variations=None,
    strategy_type="breakout",  # 'breakout' or 'ewma'
    vol_lookback_for_position_sizing=500,
    capital=1_000_0.0,
    pct_vol_target=0.20,
    value_per_point_map=None,
    fx_map=None,
    external_breakout_forecast_fn=None,
    external_ewma_forecast_fn=None,
    show_progress=True,
    pnl_dict_gen=True,
    buffer_risk_units_map=None,
    commission_bps_map=None,
    spread_bps_map=None,
    # Optional: use fixed Carver scalars for breakout instead of dynamic abs-10
    use_carver_scalars=False,
    carver_breakout_scalars=None,
):
    """
    Uses YOUR forecast functions, applies Carver two-stage capping, sizes via vol targeting,
    computes PnL/returns/Sharpe, and returns exactly THREE variables:
        (pnl_dict, signal_dict, email_df)

    email_df: one row per ticker with Sharpe, realized vol, returns, cum PnL, params, etc.
    """
    import numpy as np
    import pandas as pd
    import json

    # Use a simple progress iterator if tqdm is not available
    try:
        from tqdm.auto import trange, tqdm

        _prog = tqdm
    except ImportError:
        _prog = lambda it, **k: it

    pf = _align_prices(price_frame, tickers)
    signal_dict = {}
    pnl_dict = {}
    email_rows = []

    # Resolve external forecast functions if provided/available
    user_breakout_fn = external_breakout_forecast_fn or _try_resolve(
        "calc_breakout_forecast"
    )
    user_ewma_fn = external_ewma_forecast_fn or _try_resolve("calc_ewma_cross_forecast")

    # helpers for per-ticker maps
    def _safe_float(val, default=0.0):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def _vpp(t):
        return _safe_float(_get(value_per_point_map, t, 1.0), 1.0)

    def _fxs(t):
        if fx_map is None:
            return None
        return fx_map.get(t, None)

    def _buffer_ru(t):
        return _safe_float(_get(buffer_risk_units_map, t, 0.0), 0.0)

    def _commission_bps(t):
        return _safe_float(_get(commission_bps_map, t, 0.0), 0.0)

    def _spread_bps(t):
        return _safe_float(_get(spread_bps_map, t, 0.0), 0.0)

    for t in _prog(tickers, desc="Unified run: tickers", disable=not show_progress):
        px = pf[t]
        out_key = (ticker_dict or {}).get(t, t)

        # weights & rule variations from optimized dict if present
        w = None
        subspec = None
        if optimized_inputs_dict and t in optimized_inputs_dict:
            rec = optimized_inputs_dict[t]
            w = (
                np.array(rec.get("best_weights", []), dtype=float)
                if "best_weights" in rec
                else None
            )
            # Find the key for rule variations (horizons, pairs, etc.)
            key_options = ["horizons", "pairs", "eq_spans", "rule_params_list"]
            for k in key_options:
                if k in rec:
                    subspec = rec[k]
                    break
        if subspec is None and rule_variations is not None:
            subspec = rule_variations
        if subspec is None or len(subspec) == 0:
            raise ValueError(f"No rule_variations/pairs found for ticker {t}.")

        # Build sub-forecasts
        subs_raw = []
        strat_lower = strategy_type.lower()
        if strat_lower == "breakout":
            for h in subspec:
                sf = (
                    user_breakout_fn(px, int(h))
                    if user_breakout_fn is not None
                    else _fallback_breakout(px, int(h))
                )
                subs_raw.append(pd.Series(sf, index=px.index, dtype=float))
        elif strat_lower == "ewma":
            for f, s in subspec:
                sf = (
                    user_ewma_fn(px, int(f), int(s))
                    if user_ewma_fn is not None
                    else _fallback_ewma_cross(px, int(f), int(s))
                )
                subs_raw.append(pd.Series(sf, index=px.index, dtype=float))
        elif strat_lower == "ewmac_accel":
            for n in subspec:
                subs_raw.append(calc_ewmac_accel_forecast(px, N=int(n)))
        elif strat_lower == "meanrev_eq":
            for span in subspec:
                subs_raw.append(
                    calc_meanrev_equilibrium_forecast(px, eq_span=int(span))
                )
        elif strat_lower == "meanrev_trend_vol":
            for params in subspec:
                subs_raw.append(
                    calc_meanrev_trend_volmult_forecast(px, rule_params=params)
                )
        else:
            raise ValueError(f"Unknown strategy_type: {strategy_type}")

        # Stage 1: scale to abs-10 and cap to +/-20
        if strat_lower == "breakout" and use_carver_scalars:
            subs = [
                scale_breakout_subforecast(
                    sf,
                    int(h),
                    use_carver_scalars=True,
                    scalars_dict=carver_breakout_scalars,
                    cap=20.0,
                )
                for sf, h in zip(subs_raw, subspec)
            ]
        elif strat_lower == "meanrev_eq":
            # >>> FIX: use Carver fixed scalar 9.3, not dynamic abs-10
            subs = [scale_meanrev_eq_subforecast(sf, cap=20.0) for sf in subs_raw]
        else:
            # keep generic abs-10 scaler for other strategies, if you wish
            subs = [scale_and_cap_subforecast_abs10(sf, cap=20.0) for sf in subs_raw]

        # Weights
        if w is None or w.size != len(subs):
            w = np.ones(len(subs), dtype=float) / float(len(subs))

        # Stage 2: combine & final cap
        combined_uncapped = _combine_weighted_forecasts(subs, w)
        combined = cap_final_forecast(combined_uncapped, 20.0)
        signal_dict[out_key] = combined

        # Size positions and compute PnL/returns/Sharpe
        vpp = _vpp(t)
        fx_input = _fxs(t)
        buffer_val = _buffer_ru(t)
        comm_bps = _commission_bps(t)
        sprd_bps = _spread_bps(t)
        total_cost_bps = comm_bps + sprd_bps

        positions = apply_carver_position_sizing(
            signal=combined,
            price=px,
            capital=capital,
            pct_vol_target=pct_vol_target,
            value_per_point=vpp,
            fx_series=fx_input,
            lookback=int(vol_lookback_for_position_sizing),
        )
        ret_series, sharpe = _pnl_sharpe_from_positions(
            price=px,
            positions_blocks=positions,
            value_per_point=vpp,
            fx_series=fx_input,
            capital=capital,
            cost_bps_notional=total_cost_bps,
            deadband_blocks=buffer_val,
        )

        fx_series = (
            pd.Series(1.0, index=px.index)
            if fx_input is None
            else pd.Series(fx_input).reindex(px.index).ffill().bfill()
        )
        gross_pnl = (positions.shift(1) * px.diff() * vpp * fx_series).fillna(0.0)
        ret_series = pd.Series(ret_series).reindex(gross_pnl.index).fillna(0.0)
        pnl_series = (ret_series * float(capital)).fillna(0.0)
        costs_series = (gross_pnl - pnl_series).fillna(0.0)

        # Metrics
        realized_ann_vol = (
            (ret_series.std(ddof=0) * np.sqrt(BUSDAYS)) if len(ret_series) else np.nan
        )
        ann_return = (ret_series.mean() * BUSDAYS) if len(ret_series) else np.nan
        avg_daily_return = ret_series.mean() if len(ret_series) else np.nan
        std_daily_return = ret_series.std(ddof=0) if len(ret_series) else np.nan
        cum_pnl_ccy = pnl_series.sum() if len(pnl_series) else 0.0
        start_dt = px.index.min()
        end_dt = px.index.max()
        n_bars = int(px.size)

        if pnl_dict_gen:
            pnl_dict[out_key] = {
                "positions": positions,
                "pnl_ccy": pnl_series,
                "returns": ret_series,
                "costs_ccy": costs_series,
                "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
                "realized_ann_vol": (
                    float(realized_ann_vol) if np.isfinite(realized_ann_vol) else np.nan
                ),
                "weights": w.tolist(),
                "rule_variations": subspec,
            }

        # row for the reporting dataframe
        email_rows.append(
            {
                "ticker": out_key,
                "raw_ticker": t,
                "strategy": strategy_type.lower(),
                "start": start_dt,
                "end": end_dt,
                "n_bars": n_bars,
                "capital": float(capital),
                "pct_vol_target": float(pct_vol_target),
                "vol_lookback": int(vol_lookback_for_position_sizing),
                "value_per_point": float(_vpp(t)),
                "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
                "realized_ann_vol": (
                    float(realized_ann_vol) if np.isfinite(realized_ann_vol) else np.nan
                ),
                "ann_return": float(ann_return) if np.isfinite(ann_return) else np.nan,
                "cum_pnl_ccy": float(cum_pnl_ccy),
                "avg_daily_return": (
                    float(avg_daily_return) if np.isfinite(avg_daily_return) else np.nan
                ),
                "std_daily_return": (
                    float(std_daily_return) if np.isfinite(std_daily_return) else np.nan
                ),
                "weights": json.dumps([round(x, 6) for x in w.tolist()]),
                "rule_variations": json.dumps(subspec),
            }
        )

    email_df = pd.DataFrame(
        email_rows,
        columns=[
            "ticker",
            "raw_ticker",
            "strategy",
            "start",
            "end",
            "n_bars",
            "capital",
            "pct_vol_target",
            "vol_lookback",
            "value_per_point",
            "sharpe",
            "realized_ann_vol",
            "ann_return",
            "cum_pnl_ccy",
            "avg_daily_return",
            "std_daily_return",
            "weights",
            "rule_variations",
        ],
    )

    return pnl_dict if pnl_dict_gen else {}, signal_dict, email_df


### COst sensitive optimization
import numpy as np
import pandas as pd


def _apply_buffered_trading_series(target_ru: pd.Series, buffer_ru: float) -> pd.Series:
    """
    Carver-style buffering in RISK UNITS:
      - If |target - prev| <= buffer -> no trade (stay at prev)
      - Else move to the edge of the buffer: target - sign(delta)*buffer
    Returns the executed risk-units series.
    """
    if buffer_ru is None or buffer_ru <= 0:
        return target_ru.astype(float)

    vals = target_ru.astype(float).values
    out = np.empty_like(vals, dtype=float)
    prev = 0.0
    for i, x in enumerate(vals):
        delta = x - prev
        if abs(delta) <= buffer_ru:
            new = prev
        else:
            new = x - np.sign(delta) * buffer_ru
        out[i] = new
        prev = new
    return pd.Series(out, index=target_ru.index, name="executed_ru")


def _transaction_costs_bps(
    price: pd.Series,
    delta_units: pd.Series,
    value_per_point: float = 1.0,
    fx_series: pd.Series | None = None,
    commission_bps: float | None = 0.0,
    spread_bps: float | None = 0.0,
) -> pd.Series:
    """
    Cost per step in account currency:
      traded_notional_t = |Deltaunits_t| * price_t * value_per_point * (fx_t or 1)
      bps_paid_t = commission_bps + (spread_bps / 2)
      cost_t = traded_notional_t * bps_paid_t / 10_000
    """
    px = price.astype(float)
    fx = (
        (fx_series if fx_series is not None else pd.Series(1.0, index=px.index))
        .astype(float)
        .reindex(px.index)
        .fillna(method="ffill")
        .fillna(1.0)
    )
    traded_notional = delta_units.abs().astype(float) * px * float(value_per_point) * fx
    bps_paid = float(commission_bps or 0.0) + float(spread_bps or 0.0) / 2.0
    return traded_notional * (bps_paid / 10_000.0)


def _backtest_ticker_carver_costs(
    price: pd.Series,
    sub_forecasts: list[pd.Series],
    weights: np.ndarray,
    capital: float,
    pct_vol_target: float,
    vol_lookback_for_position_sizing: int,
    value_per_point: float = 1.0,
    fx_series: pd.Series | None = None,
    # NEW: Carver buffer + costs
    buffer_risk_units: float = 0.0,  # e.g., 0.25 risk units
    commission_bps: float = 0.0,  # per trade (one-way)
    spread_bps: float = 0.0,  # average bid-ask width in bps
) -> tuple[pd.DataFrame, float]:
    """
    Returns (diagnostics_df, net_sharpe_after_costs).
    - Forecasts are assumed already scaled to +/- 20 and combined using 'weights'.
    - Risk targeting: risk_units = combined_forecast / 10
    - Costs applied on Deltaunits_t using commission + half-spread (bps).
    - Buffering applied in risk-units before converting to units.
    """
    # align everything
    idx = price.index
    subs = [sf.reindex(idx).astype(float) for sf in sub_forecasts]
    w = np.array(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(subs), dtype=float)
    w = w / w.sum()

    # combine (already scaled/capped sub-forecasts)
    combined = sum(w[i] * subs[i] for i in range(len(subs)))
    combined = cap_final_forecast(combined, 20.0).rename("combined_forecast")

    # map forecast -> target risk units (Carver): +/-20 -> +/-2 risk units
    target_ru = (combined / 10.0).clip(-2.0, 2.0).rename("target_ru")

    # apply Carver no-trade buffer in risk units
    executed_ru = _apply_buffered_trading_series(target_ru, float(buffer_risk_units))

    # Volatility scalar for position sizing
    vol_scalar = compute_volatility_scalar(
        price=price,
        capital=float(capital),
        pct_vol_target=float(pct_vol_target),
        lookback=vol_lookback_for_position_sizing,
        value_per_point=float(value_per_point),
        fx_series=fx_series,
    )

    # convert risk-units to actual units of the instrument
    # position = vol_scalar * (forecast / 10) = vol_scalar * risk_units
    units = (
        (vol_scalar * executed_ru)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0.0)
        .rename("position_units")
    )

    # PnL gross (account ccy)
    fx = (
        (fx_series if fx_series is not None else pd.Series(1.0, index=idx))
        .astype(float)
        .reindex(idx)
        .fillna(method="ffill")
        .fillna(1.0)
    )
    dP = price.diff()
    pnl_gross = (units.shift(1).fillna(0.0) * dP * float(value_per_point) * fx).rename(
        "pnl_gross"
    )

    # trading costs on Deltaunits
    delta_units = units.diff().fillna(units)  # include initial entry
    costs = _transaction_costs_bps(
        price=price,
        delta_units=delta_units,
        value_per_point=float(value_per_point),
        fx_series=fx,
        commission_bps=float(commission_bps),
        spread_bps=float(spread_bps),
    ).rename("costs")

    pnl_net = (pnl_gross - costs).rename("pnl_net")

    # Sharpe on returns (pnl / capital)
    ret = (pnl_net / float(capital)).fillna(0.0)
    if ret.std(ddof=0) == 0 or ret.empty:
        sharpe = 0.0
    else:
        sharpe = (ret.mean() / ret.std(ddof=0)) * np.sqrt(BUSDAYS)

    diag = pd.concat(
        [
            price.rename("price"),
            combined,
            target_ru,
            executed_ru,
            units,
            pnl_gross,
            costs,
            pnl_net,
        ],
        axis=1,
    )

    return diag, sharpe


def optimize_breakout_weights_carver_cost(
    price_frame: pd.DataFrame,
    tickers: list[str],
    breakout_horizons: list[int],
    vol_lookback_for_position_sizing: int = 500,
    n_trials_per_ticker: int = 500,
    capital: float = 1_000_000.0,
    pct_vol_target: float = 0.20,
    value_per_point_map: dict[str, float] | None = None,
    fx_map: dict[str, pd.Series] | None = None,
    random_seed: int = 42,
    external_breakout_forecast_fn=None,
    use_carver_scalars: bool = False,
    carver_breakout_scalars: dict[int, float] | None = None,
    # quiet controls
    silent: bool = True,
    n_jobs: int | None = None,
    show_progress_bar: bool = False,
    # NEW: costs + buffer
    buffer_risk_units: float = 0.0,
    commission_bps_map: dict[str, float] | float | None = None,
    spread_bps_map: dict[str, float] | float | None = None,
) -> dict[str, dict]:
    import optuna
    from optuna.samplers import TPESampler

    pf = price_frame.loc[:, tickers].copy()
    results: dict[str, dict] = {}
    sampler = TPESampler(seed=random_seed)

    user_breakout_fn = external_breakout_forecast_fn or _try_resolve(
        "calc_breakout_forecast"
    )

    # Resolve cost lookups (scalar or per-ticker map)
    def _get(map_or_scalar, t, default=0.0):
        if map_or_scalar is None:
            return default
        if isinstance(map_or_scalar, dict):
            return float(map_or_scalar.get(t, default))
        return float(map_or_scalar)

    for t in tickers:
        px = pf[t].dropna()

        # build and scale sub-forecasts
        subs_raw = []
        for h in breakout_horizons:
            sf = (
                user_breakout_fn(px, int(h))
                if user_breakout_fn is not None
                else _fallback_breakout(px, int(h))
            )
            subs_raw.append(pd.Series(sf, index=px.index, dtype=float))

        if use_carver_scalars and "scale_breakout_subforecast" in globals():
            subs = [
                scale_breakout_subforecast(
                    sf,
                    int(h),
                    use_carver_scalars=True,
                    scalars_dict=carver_breakout_scalars,
                    cap=20.0,
                )
                for sf, h in zip(subs_raw, breakout_horizons)
            ]
        else:
            subs = [scale_and_cap_subforecast_abs10(sf, cap=20.0) for sf in subs_raw]

        vpp = 1.0 if not value_per_point_map else float(value_per_point_map.get(t, 1.0))
        fx_series = None if not fx_map else fx_map.get(t, None)
        comm_bps = _get(commission_bps_map, t, default=0.0)
        sprd_bps = _get(spread_bps_map, t, default=0.0)

        def objective(trial: "optuna.trial.Trial") -> float:
            w = np.array(
                [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(subs))],
                dtype=float,
            )
            if w.sum() == 0.0:
                w[:] = 1.0
            _, sr = _backtest_ticker_carver_costs(
                price=px,
                sub_forecasts=subs,
                weights=w,
                capital=capital,
                pct_vol_target=pct_vol_target,
                vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
                value_per_point=vpp,
                fx_series=fx_series,
                # costs + buffer -> NET Sharpe
                buffer_risk_units=buffer_risk_units,
                commission_bps=comm_bps,
                spread_bps=sprd_bps,
            )
            if not np.isfinite(sr):
                return -1e12
            return float(sr)

        study = optuna.create_study(direction="maximize", sampler=sampler)
        with _optuna_quiet(silent):
            _study_optimize_quiet(
                study,
                objective,
                n_trials=n_trials_per_ticker,
                n_jobs=n_jobs,
                show_progress_bar=show_progress_bar,
            )

        best_params = study.best_trial.params
        raw = np.array(
            [best_params.get(f"w_{i}", 0.0) for i in range(len(subs))], dtype=float
        )
        if raw.sum() == 0.0:
            raw[:] = 1.0
        best_w = (raw / raw.sum()).tolist()

        results[t] = {
            "best_weights": best_w,
            "sharpe": float(study.best_value),  # NET after costs
            "horizons": list(map(int, breakout_horizons)),
            "buffer_risk_units": float(buffer_risk_units),
            "commission_bps": float(comm_bps),
            "spread_bps": float(sprd_bps),
        }

    return results


# =========================
# NEW: OPTUNA optimizer: EWMA cross with costs (quiet)
# =========================
def optimize_ewma_cross_weights_carver_cost(
    price_frame: pd.DataFrame,
    tickers: list[str],
    ewma_pairs: list[tuple[int, int]],
    vol_lookback_for_position_sizing: int = 500,
    n_trials_per_ticker: int = 500,
    capital: float = 1_000_000.0,
    pct_vol_target: float = 0.20,
    value_per_point_map: dict[str, float] | None = None,
    fx_map: dict[str, pd.Series] | None = None,
    random_seed: int = 42,
    external_ewma_forecast_fn=None,
    # quiet controls
    silent: bool = True,
    n_jobs: int | None = None,
    show_progress_bar: bool = False,
    # NEW: costs + buffer
    buffer_risk_units: float = 0.0,
    commission_bps_map: dict[str, float] | float | None = None,
    spread_bps_map: dict[str, float] | float | None = None,
) -> dict[str, dict]:
    """
    Optuna optimizer for EWMA-cross weights, maximizing NET Sharpe after costs.
    """
    import optuna
    from optuna.samplers import TPESampler

    pf = price_frame.loc[:, tickers].copy()
    results: dict[str, dict] = {}
    sampler = TPESampler(seed=random_seed)

    user_ewma_fn = external_ewma_forecast_fn or _try_resolve("calc_ewma_cross_forecast")

    def _get(map_or_scalar, t, default=0.0):
        if map_or_scalar is None:
            return default
        if isinstance(map_or_scalar, dict):
            return float(map_or_scalar.get(t, default))
        return float(map_or_scalar)

    for t in tickers:
        px = pf[t].dropna()

        subs_raw = []
        for f, s in ewma_pairs:
            sf = (
                user_ewma_fn(px, int(f), int(s))
                if user_ewma_fn is not None
                else _fallback_ewma_cross(px, int(f), int(s))
            )
            subs_raw.append(pd.Series(sf, index=px.index, dtype=float))

        subs = [scale_and_cap_subforecast_abs10(sf, cap=20.0) for sf in subs_raw]

        vpp = 1.0 if not value_per_point_map else float(value_per_point_map.get(t, 1.0))
        fx_series = None if not fx_map else fx_map.get(t, None)
        comm_bps = _get(commission_bps_map, t, default=0.0)
        sprd_bps = _get(spread_bps_map, t, default=0.0)

        def objective(trial: "optuna.trial.Trial") -> float:
            w = np.array(
                [trial.suggest_float(f"w_{i}", 0.0, 1.0) for i in range(len(subs))],
                dtype=float,
            )
            if w.sum() == 0.0:
                w[:] = 1.0

            _, sr = _backtest_ticker_carver_costs(
                price=px,
                sub_forecasts=subs,
                weights=w,
                capital=capital,
                pct_vol_target=pct_vol_target,
                vol_lookback_for_position_sizing=vol_lookback_for_position_sizing,
                value_per_point=vpp,
                fx_series=fx_series,
                buffer_risk_units=buffer_risk_units,
                commission_bps=comm_bps,
                spread_bps=sprd_bps,
            )
            return float(sr) if np.isfinite(sr) else -1e12

        study = optuna.create_study(direction="maximize", sampler=sampler)
        with _optuna_quiet(silent):
            _study_optimize_quiet(
                study, objective, n_trials_per_ticker, n_jobs, show_progress_bar
            )

        best_params = study.best_trial.params
        raw = np.array(
            [best_params.get(f"w_{i}", 0.0) for i in range(len(subs))], dtype=float
        )
        if raw.sum() == 0.0:
            raw[:] = 1.0
        best_w = (raw / raw.sum()).tolist()

        results[t] = {
            "best_weights": best_w,
            "sharpe": float(study.best_value),
            "pairs": [list(map(int, p)) for p in ewma_pairs],
            "buffer_risk_units": float(buffer_risk_units),
            "commission_bps": float(comm_bps),
            "spread_bps": float(sprd_bps),
        }
    return results


# =========================
# Vol-target sweep & heatmap
# =========================

# === Sensitivity utilities for CarverBacktest ===
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(x, **k):
        return x


# ---------------- Plot helper (yours) ----------------
def plot_sensitivity_with_plotly(
    results_df: pd.DataFrame, parameter_name: str, metric_name: str
):
    """
    Creates an interactive Plotly chart to visualize sensitivity analysis results.

    Args:
        results_df (pd.DataFrame): DataFrame containing the analysis results.
        parameter_name (str): The name of the column with the parameter values (e.g., 'vol_lookback').
        metric_name (str): The name of the column with the metric values (e.g., 'average_sharpe').
    """
    fig = go.Figure()

    # Add the line and markers trace
    fig.add_trace(
        go.Scatter(
            x=results_df[parameter_name],
            y=results_df[metric_name],
            mode="lines+markers+text",
            name=metric_name.replace("_", " ").title(),
            text=results_df[metric_name].apply(
                lambda x: f"{x:.2f}" if pd.notnull(x) else ""
            ),
            textposition="top center",
            marker=dict(size=8),
            line=dict(width=2),
        )
    )

    # Customize layout
    fig.update_layout(
        title=dict(
            text=f'Strategy Sensitivity to {parameter_name.replace("_", " ").title()}',
            x=0.5,
        ),
        xaxis_title=parameter_name.replace("_", " ").title(),
        yaxis_title=metric_name.replace("_", " ").title(),
        width=1000,
        height=600,
        hovermode="x unified",
        template="plotly_white",
    )

    # Ensure all tested parameter values are shown on the x-axis
    fig.update_xaxes(tickmode="array", tickvals=results_df[parameter_name])

    print("Displaying interactive sensitivity analysis plot...")
    fig.show()


# ---------------- Internals ----------------
def _to_frac_vol(x) -> float:
    """
    Normalize a vol target to an annualized fraction:
    - 0.2, '0.2'   -> 0.2
    - 20, '20', '20%' or 'vol20' -> 0.20
    """
    if isinstance(x, str):
        s = x.lower().replace("vol", "").replace("%", "").strip()
        v = float(s)
    else:
        v = float(x)
    return v / 100.0 if v > 1.5 else v


def _get_email_df_from_run_backtest_output(out):
    """
    Works with both (pnl_dict, signal_dict, email_df)
    and (pnl_dict, signal_dict, position_dict, email_df).
    Returns email_df or None.
    """
    if not isinstance(out, tuple):
        return None
    for obj in reversed(out):
        if isinstance(obj, pd.DataFrame):
            return obj
    return None


# ---------------- Main runners ----------------
def run_volatility_lookback_sensitivity_bt(
    price_frame: pd.DataFrame,
    tickers: list[str],
    strategy_type: str,
    rule_variations,
    *,
    vol_lookbacks_to_test: list[int] | None = None,
    pct_vol_target: float | int | str = 0.20,
    capital: float = 1_000_000.0,
    value_per_point_map: dict[str, float] | None = None,
    fx_map: dict[str, pd.Series] | None = None,
    use_carver_scalars: bool = True,
    carver_breakout_scalars: dict[int, float] | None = None,
    optimized_params: dict | None = None,  # pass pre-optimized weights if you have them
    n_trials_if_optimize: int = 80,  # used only if optimized_params is None
    show_plot: bool = True,
    show_progress: bool = True,
    BacktestClass=None,  # pass bt.CarverBacktest if you didn't import as bt
) -> pd.DataFrame:
    """
    Sweep vol_lookback_for_position_sizing and compute average/median Sharpe across tickers
    using the CarverBacktest class. Returns a DataFrame with:
    ['vol_lookback','average_sharpe','median_sharpe'].
    """
    # Resolve CarverBacktest class if not provided
    if BacktestClass is None:
        try:
            import backtester as bt

            BacktestClass = bt.CarverBacktest
        except Exception:
            BacktestClass = (
                globals().get("CarverBacktest") or globals()["bt"].CarverBacktest
            )

    if vol_lookbacks_to_test is None:
        vol_lookbacks_to_test = list(range(30, 360, 10))

    pct_vol_target = _to_frac_vol(pct_vol_target)
    results = []

    iterator = tqdm(
        vol_lookbacks_to_test, desc="Vol lookback sweep", disable=not show_progress
    )
    for lookback in iterator:
        runner = BacktestClass(
            price_frame=price_frame,
            tickers=tickers,
            strategy_type=strategy_type,
            rule_variations=rule_variations,
            capital=capital,
            pct_vol_target=pct_vol_target,
            vol_lookback_for_position_sizing=int(lookback),
            value_per_point_map=value_per_point_map,
            fx_map=fx_map,
            use_carver_scalars=use_carver_scalars,
            carver_breakout_scalars=carver_breakout_scalars,
        )

        if optimized_params is None:
            # quick/quiet optimize just to get reasonable weights
            try:
                runner.run_optimization(
                    n_trials=n_trials_if_optimize, show_progress_bar=False
                )
            except Exception:
                pass
        else:
            runner.optimized_params = optimized_params

        out = runner.run_backtest(use_optimized_weights=True)
        email_df = _get_email_df_from_run_backtest_output(out)

        if email_df is not None and not email_df.empty and "sharpe" in email_df:
            avg_sr = float(email_df["sharpe"].dropna().mean())
            med_sr = float(email_df["sharpe"].dropna().median())
        else:
            avg_sr = np.nan
            med_sr = np.nan

        results.append(
            {
                "vol_lookback": int(lookback),
                "average_sharpe": avg_sr,
                "median_sharpe": med_sr,
            }
        )

    results_df = (
        pd.DataFrame(results).sort_values("vol_lookback").reset_index(drop=True)
    )

    if show_plot and not results_df.empty:
        # You can call once for median and once for average
        plot_sensitivity_with_plotly(results_df, "vol_lookback", "median_sharpe")
        plot_sensitivity_with_plotly(results_df, "vol_lookback", "average_sharpe")

    return results_df


def run_vol_target_sensitivity_bt(
    price_frame: pd.DataFrame,
    tickers: list[str],
    strategy_type: str,
    rule_variations,
    *,
    vol_targets_to_test: (
        list[float] | None
    ) = None,  # e.g. [10,15,20] or [0.10,0.15,0.20]
    vol_lookback_for_position_sizing: int = 500,
    capital: float = 1_000_000.0,
    value_per_point_map: dict[str, float] | None = None,
    fx_map: dict[str, pd.Series] | None = None,
    use_carver_scalars: bool = True,
    carver_breakout_scalars: dict[int, float] | None = None,
    optimized_params: dict | None = None,
    n_trials_if_optimize: int = 80,
    show_plot: bool = True,
    show_progress: bool = True,
    BacktestClass=None,
) -> pd.DataFrame:
    """
    Sweep the portfolio volatility target (annualized) while holding the sizing lookback fixed.
    Returns a DataFrame with: ['pct_vol_target','pct_vol_target_pct','average_sharpe','median_sharpe'].
    """
    if BacktestClass is None:
        try:
            import backtester as bt

            BacktestClass = bt.CarverBacktest
        except Exception:
            BacktestClass = (
                globals().get("CarverBacktest") or globals()["bt"].CarverBacktest
            )

    if vol_targets_to_test is None:
        vol_targets_to_test = [10, 12, 15, 18, 20, 22, 25, 30]  # % p.a.

    vol_targets_frac = [_to_frac_vol(v) for v in vol_targets_to_test]
    results = []

    iterator = tqdm(
        vol_targets_frac, desc="Vol target sweep", disable=not show_progress
    )
    for vt in iterator:
        runner = BacktestClass(
            price_frame=price_frame,
            tickers=tickers,
            strategy_type=strategy_type,
            rule_variations=rule_variations,
            capital=capital,
            pct_vol_target=float(vt),
            vol_lookback_for_position_sizing=int(vol_lookback_for_position_sizing),
            value_per_point_map=value_per_point_map,
            fx_map=fx_map,
            use_carver_scalars=use_carver_scalars,
            carver_breakout_scalars=carver_breakout_scalars,
        )

        if optimized_params is None:
            try:
                runner.run_optimization(
                    n_trials=n_trials_if_optimize, show_progress_bar=False
                )
            except Exception:
                pass
        else:
            runner.optimized_params = optimized_params

        out = runner.run_backtest(use_optimized_weights=True)
        email_df = _get_email_df_from_run_backtest_output(out)

        if email_df is not None and not email_df.empty and "sharpe" in email_df:
            avg_sr = float(email_df["sharpe"].dropna().mean())
            med_sr = float(email_df["sharpe"].dropna().median())
        else:
            avg_sr = np.nan
            med_sr = np.nan

        results.append(
            {
                "pct_vol_target": float(vt),
                "pct_vol_target_pct": round(vt * 100.0, 2),
                "average_sharpe": avg_sr,
                "median_sharpe": med_sr,
            }
        )

    results_df = (
        pd.DataFrame(results).sort_values("pct_vol_target").reset_index(drop=True)
    )

    if show_plot and not results_df.empty:
        # Plot against the % label column so the x-axis is intuitive
        plot_sensitivity_with_plotly(results_df, "pct_vol_target_pct", "median_sharpe")
        plot_sensitivity_with_plotly(results_df, "pct_vol_target_pct", "average_sharpe")

    return results_df


# ---------- helpers ----------
import os
import glob
import re
import pickle
from typing import List, Dict, Any, Tuple


def _horizons_tag(horizons: List[int]) -> str:
    """Turn [10,20,40,80,160] -> '10_20_40_80_160'."""
    return "_".join(str(int(h)) for h in horizons)


def _vol_token(pct_vol_target) -> str:
    """
    Always return 'volNN' (NN=whole percent).
    Accepts: 0.80, 80, '80', '0.8', 'vol80', 'vol0.8', '80%'.
    """
    if isinstance(pct_vol_target, str):
        s = pct_vol_target.lower().replace("vol", "").replace("%", "").strip()
        try:
            v = float(s)
        except Exception:
            v = 0.0
    else:
        v = float(pct_vol_target)

    v_percent = v * 100.0 if v <= 2.0 else v
    vol_int = int(round(v_percent))
    return f"vol{vol_int}"  # e.g., 'vol80'


def _build_filepattern(
    directory: str,
    file_prefix: str,
    horizons: List[int],
    pct_vol_target,
) -> str:
    """
    Prefer: <dir>/<file_prefix><horizons>_<volNN>_*.pkl (underscore)
            <dir>/<file_prefix><horizons>_<volNN>-*.pkl (dash)
    Fallback: decimal style (vol0.8_* / vol0.8-*)
    Returns the first pattern that has has matches; otherwise the preferred pattern.
    """
    dir_abs = os.path.abspath(directory)
    horizons_str = _horizons_tag(horizons)
    vol_tok = _vol_token(pct_vol_target)

    pat1 = os.path.join(dir_abs, f"{file_prefix}{horizons_str}_{vol_tok}_*.pkl")
    pat1b = os.path.join(dir_abs, f"{file_prefix}{horizons_str}_{vol_tok}-*.pkl")

    # Fallback to decimal style if you ever saved that way
    vol_dec = f"vol{(float(pct_vol_target) if not isinstance(pct_vol_target, str) else pct_vol_target)}"
    pat2 = os.path.join(dir_abs, f"{file_prefix}{horizons_str}_{vol_dec}_*.pkl")
    pat2b = os.path.join(dir_abs, f"{file_prefix}{horizons_str}_{vol_dec}-*.pkl")

    for pat in (pat1, pat1b, pat2, pat2b):
        if glob.glob(pat):
            return pat
    return pat1


def _load_pickle_safe(path: str):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None


# ---------- analysis ----------
def analyze_lookback_sharpe_heatmap(
    directory: str,
    tickers: List[str],
    horizons: List[int],
    pct_vol_target: float | int | str,
    *,
    file_prefix: str = "optimized_breakout_params_",
    show_heatmap: bool = True,
    also_plot_median_line: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Any, Any]:
    """
    Load all optimization pickles for these horizons & vol target.
    Build a Ticker x Lookback Sharpe grid and plot results.
    """
    pattern = _build_filepattern(directory, file_prefix, horizons, pct_vol_target)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[INFO] No files matched pattern: {pattern}")
        return pd.DataFrame(), pd.DataFrame(), None, None

    lookback_to_path: Dict[int, str] = {}
    for p in files:
        base = os.path.splitext(os.path.basename(p))[0]
        m = re.search(r"(?:_|-)(\d+)$", base)
        if m:
            lookback_to_path[int(m.group(1))] = p

    if not lookback_to_path:
        print(f"[INFO] Files matched but no lookbacks parsed under: {pattern}")
        return pd.DataFrame(), pd.DataFrame(), None, None

    cols = sorted(lookback_to_path.keys())
    tickers = [str(t) for t in tickers]
    sharpe_grid = pd.DataFrame(index=tickers, columns=cols, dtype=float)

    for lb, path in lookback_to_path.items():
        res = _load_pickle_safe(path)
        if not isinstance(res, dict):
            continue
        for t in tickers:
            if t in res and isinstance(res[t], dict):
                sr = res[t].get("sharpe", np.nan)
                sharpe_grid.loc[t, lb] = (
                    float(sr) if pd.notnull(sr) and np.isfinite(sr) else np.nan
                )

    summary_df = (
        pd.DataFrame(
            {
                "lookback": sharpe_grid.columns.values,
                "median_sharpe": sharpe_grid.median(axis=0, skipna=True).values,
                "average_sharpe": sharpe_grid.mean(axis=0, skipna=True).values,
            }
        )
        .sort_values("lookback")
        .reset_index(drop=True)
    )

    heatmap_fig, median_fig = None, None
    if show_heatmap and not sharpe_grid.empty:
        heatmap_fig = go.Figure(
            data=go.Heatmap(
                z=sharpe_grid.values,
                x=sharpe_grid.columns.astype(int).tolist(),
                y=sharpe_grid.index.tolist(),
                colorbar=dict(title="Sharpe"),
                zauto=True,
            )
        )
        heatmap_fig.update_layout(
            title=f"Sharpe by Ticker vs Lookback ({_vol_token(pct_vol_target)}, horizons={_horizons_tag(horizons)})",
            xaxis_title="vol_lookback_for_position_sizing",
            yaxis_title="ticker",
            template="plotly_white",
            width=1100,
            height=600 + 8 * len(tickers),
        )
        heatmap_fig.show()

    if also_plot_median_line and not summary_df.empty:
        median_fig = go.Figure()
        median_fig.add_trace(
            go.Scatter(
                x=summary_df["lookback"],
                y=summary_df["median_sharpe"],
                mode="lines+markers+text",
                text=summary_df["median_sharpe"].apply(
                    lambda v: f"{v:.2f}" if pd.notnull(v) else ""
                ),
                textposition="top center",
                name="Median Sharpe",
                line=dict(width=2),
            )
        )
        median_fig.add_trace(
            go.Scatter(
                x=summary_df["lookback"],
                y=summary_df["average_sharpe"],
                mode="lines+markers",
                name="Average Sharpe",
                line=dict(width=2, dash="dash"),
            )
        )
        median_fig.update_layout(
            title="Sharpe vs Lookback (Median & Average)",
            xaxis_title="vol_lookback_for_position_sizing",
            yaxis_title="Sharpe",
            template="plotly_white",
            width=1000,
            height=500,
            hovermode="x unified",
        )
        median_fig.show()

    return sharpe_grid, summary_df, heatmap_fig, median_fig


def select_best_params_across_lookbacks(
    directory: str,
    tickers: List[str],
    horizons: List[int],
    pct_vol_target: float | int | str,
    *,
    file_prefix: str = "optimized_breakout_params_",
) -> Dict[str, Dict[str, Any]]:
    """
    Scan all lookback pickles and return the entry with the highest Sharpe for each ticker.
    """
    pattern = _build_filepattern(directory, file_prefix, horizons, pct_vol_target)
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[INFO] No files matched pattern: {pattern}")
        return {}

    tickers = [str(t) for t in tickers]
    best: Dict[str, Dict[str, Any]] = {}

    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r"(?:_|-)(\d+)$", base)
        if not m:
            continue
        lb = int(m.group(1))

        res = _load_pickle_safe(path)
        if not isinstance(res, dict):
            continue

        for t in tickers:
            if t not in res or not isinstance(res[t], dict):
                continue
            sr = res[t].get("sharpe", None)
            if sr is None or not np.isfinite(sr):
                continue
            if (t not in best) or (sr > best[t]["sharpe"]):
                best[t] = {
                    "best_weights": res[t].get("best_weights", []),
                    "sharpe": float(sr),
                    "horizons": res[t].get("horizons", horizons),
                    "lookback": int(lb),
                    "source_file": path,
                }

    for t, v in best.items():
        w = np.array(v.get("best_weights", []), dtype=float)
        if w.size and w.sum() > 0:
            v["best_weights"] = (w / w.sum()).tolist()

    print(f"[INFO] Selected best parameters for {len(best)} / {len(tickers)} tickers.")
    return best


# ======================================================================================
# Backtest Class for structured execution and optimization
# ======================================================================================


class CarverBacktest:
    """
    A class to structure and execute a Carver-style backtest and optimization.

    This class holds all parameters for a backtest, allowing for cleaner execution
    of signal generation, optimization, and PnL calculation.

    Example Usage:
    --------------
    # 1. Initialize with data and parameters
    bt = CarverBacktest(
        price_frame=my_price_df,
        tickers=['SPY', 'GLD'],
        strategy_type='breakout',
        rule_variations=[20, 40, 80, 160],
        capital=1_000_000,
        pct_vol_target=0.20,
        commission_bps_map={'SPY': 0.5, 'GLD': 1.0}, # Optional costs
        spread_bps_map=0.5,                         # Optional costs
        buffer_risk_units=0.1                       # Optional buffer
    )

    # 2. Run optimization to find the best weights
    optimized_params = bt.run_optimization(n_trials=500)

    # 3. Run the final backtest using the optimized parameters
    pnl_results, signal_dict, summary_df = bt.run()

    # 4. Access results
    print(bt.get_summary())
    spy_pnl = bt.pnl_dict['SPY']['pnl_ccy']
    """

    def __init__(
        self,
        price_frame: pd.DataFrame,
        tickers: List[str],
        strategy_type: str,
        rule_variations: List,
        **kwargs,
    ):
        """
        Initializes the backtest environment.

        Args:
            price_frame (pd.DataFrame): DataFrame with datetime index and ticker prices in columns.
            tickers (List[str]): List of tickers to include in the backtest.
            strategy_type (str): 'breakout', 'ewma', 'ewmac_accel', 'meanrev_eq', 'meanrev_trend_vol'.
            rule_variations (List): List of horizons, pairs, or parameter dicts for the strategy.
            **kwargs: Other optional backtest parameters.
        """
        self.price_frame = _align_prices(price_frame, tickers)
        self.tickers = tickers
        self.strategy_type = strategy_type.lower()
        self.rule_variations = rule_variations

        # Core backtest params
        self.capital = float(kwargs.get("capital", 1_000_000.0))
        self.pct_vol_target = float(kwargs.get("pct_vol_target", 0.20))
        self.vol_lookback = int(kwargs.get("vol_lookback_for_position_sizing", 365))

        # Data maps
        self.value_per_point_map = kwargs.get("value_per_point_map")
        self.fx_map = kwargs.get("fx_map")
        self.ticker_dict = kwargs.get("ticker_dict")

        # Cost params
        self.buffer_ru = kwargs.get("buffer_risk_units", 0.0)
        self.commission_bps = kwargs.get("commission_bps_map")
        self.spread_bps = kwargs.get("spread_bps_map")

        # External functions & breakout-specific scalars
        self.breakout_fn = kwargs.get("external_breakout_forecast_fn")
        self.ewma_fn = kwargs.get("external_ewma_forecast_fn")
        self.use_carver_scalars = bool(kwargs.get("use_carver_scalars", False))
        self.carver_breakout_scalars = kwargs.get("carver_breakout_scalars")

        # Results attributes
        self.optimized_params: Optional[Dict] = None
        self.pnl_dict: Optional[Dict] = None
        self.signal_dict: Optional[Dict] = None
        self.summary_df: Optional[pd.DataFrame] = None
        self.position_dict: Optional[Dict[str, pd.Series]] = None

    def run_optimization(
        self,
        n_trials: int = 250,
        random_seed: int = 42,
        n_jobs: Optional[int] = -1,
        show_progress_bar: bool = False,
    ) -> Dict:
        """
        Runs Optuna optimization to find the best weights for the specified strategy.

        It automatically detects whether to run a cost-aware optimization based on
        whether cost parameters were provided during initialization.

        Args:
            n_trials (int): Number of Optuna trials per ticker.
            random_seed (int): Seed for reproducibility.
            n_jobs (int, optional): Number of parallel jobs for Optuna. Defaults to -1 (all CPUs).
            show_progress_bar (bool): Whether to show Optuna's progress bar.

        Returns:
            Dict: A dictionary of optimized parameters for each ticker.
        """

        def _has_buffer_cost(val):
            if val is None:
                return False
            if _is_num(val):
                return float(val) > 0
            if isinstance(val, Mapping):
                return any(_has_buffer_cost(v) for v in val.values())
            if isinstance(val, pd.Series):
                return bool(val.fillna(0.0).abs().sum() > 0)
            if isinstance(val, pd.DataFrame):
                return bool(val.fillna(0.0).abs().values.sum() > 0)
            return True

        has_costs = (
            _has_buffer_cost(self.buffer_ru)
            or self.commission_bps is not None
            or self.spread_bps is not None
        )

        common_args = {
            "price_frame": self.price_frame,
            "tickers": self.tickers,
            "vol_lookback_for_position_sizing": self.vol_lookback,
            "n_trials_per_ticker": n_trials,
            "capital": self.capital,
            "pct_vol_target": self.pct_vol_target,
            "value_per_point_map": self.value_per_point_map,
            "fx_map": self.fx_map,
            # "random_seed": random_seed,
            "silent": not show_progress_bar,
            "n_jobs": n_jobs,
            "show_progress_bar": show_progress_bar,
        }

        cost_args = {
            "buffer_risk_units": self.buffer_ru,
            "commission_bps_map": self.commission_bps,
            "spread_bps_map": self.spread_bps,
        }

        strat_lower = self.strategy_type.lower()

        if strat_lower in ["meanrev_eq", "meanrev_trend_vol"] and has_costs:
            # remap to the *_map* key that these optimizers expect
            common_args.update(
                {
                    "buffer_risk_units_map": self.buffer_ru,
                    "commission_bps_map": self.commission_bps,
                    "spread_bps_map": self.spread_bps,
                }
            )

        if strat_lower == "breakout":
            optimizer = (
                optimize_breakout_weights_carver_cost
                if has_costs
                else optimize_breakout_weights_carver
            )
            if has_costs:
                common_args.update(cost_args)
            self.optimized_params = optimizer(
                breakout_horizons=self.rule_variations,
                external_breakout_forecast_fn=self.breakout_fn,
                use_carver_scalars=self.use_carver_scalars,
                carver_breakout_scalars=self.carver_breakout_scalars,
                **common_args,
            )
        elif strat_lower == "ewma":
            optimizer = (
                optimize_ewma_cross_weights_carver_cost
                if has_costs
                else optimize_ewma_cross_weights_carver
            )
            if has_costs:
                common_args.update(cost_args)
            self.optimized_params = optimizer(
                ewma_pairs=self.rule_variations,
                external_ewma_forecast_fn=self.ewma_fn,
                **common_args,
            )
        elif strat_lower in ["ewmac_accel", "meanrev_eq", "meanrev_trend_vol"]:
            # For now, new strategies use the non-cost-sensitive generic optimizer
            # A cost-sensitive generic optimizer could be added if needed
            if strat_lower == "ewmac_accel":
                if has_costs:
                    common_args.update(cost_args)
                self.optimized_params = optimize_ewmac_accel_weights_carver(
                    rule_variations=self.rule_variations, **common_args
                )

            elif strat_lower == "meanrev_eq":
                self.optimized_params = optimize_meanrev_eq_optuna_fs(
                    mr_eq_spans=self.rule_variations, **common_args
                )
            elif strat_lower == "meanrev_trend_vol":
                if has_costs:
                    common_args.update(cost_args)
                self.optimized_params = optimize_meanrev_trend_vol_fs(
                    rule_variations=self.rule_variations, **common_args
                )
        else:
            raise ValueError(
                f"Unknown strategy_type for optimization: {self.strategy_type}"
            )

        return self.optimized_params

    def run(
        self, optimized_params: Optional[Dict] = None, show_progress: bool = True
    ) -> Tuple[Dict, Dict, pd.DataFrame]:
        """
        Executes the backtest using the stored configuration.

        Uses `self.optimized_params` if available, or falls back to the provided
        `optimized_params` argument or equal weights.

        Args:
            optimized_params (Dict, optional): Pre-computed optimization results. If None,
                                               uses results from `run_optimization`.
            show_progress (bool): Whether to display a progress bar over tickers.

        Returns:
            Tuple[Dict, Dict, pd.DataFrame]: A tuple containing:
                - pnl_dict: Detailed PnL and position data per ticker.
                - signal_dict: The final combined forecast series per ticker.
                - summary_df: A DataFrame with key performance metrics per ticker.
        """
        # Use class's optimized params if available, otherwise use provided arg
        params_to_use = (
            self.optimized_params
            if self.optimized_params is not None
            else optimized_params
        )

        pnl_dict, signal_dict, summary_df = carver_gen_signal_unified(
            price_frame=self.price_frame,
            tickers=self.tickers,
            ticker_dict=self.ticker_dict,
            optimized_inputs_dict=params_to_use,
            rule_variations=self.rule_variations,
            strategy_type=self.strategy_type,
            vol_lookback_for_position_sizing=self.vol_lookback,
            capital=self.capital,
            pct_vol_target=self.pct_vol_target,
            value_per_point_map=self.value_per_point_map,
            fx_map=self.fx_map,
            external_breakout_forecast_fn=self.breakout_fn,
            external_ewma_forecast_fn=self.ewma_fn,
            show_progress=show_progress,
            pnl_dict_gen=True,
            buffer_risk_units_map=self.buffer_ru,
            commission_bps_map=self.commission_bps,
            spread_bps_map=self.spread_bps,
            use_carver_scalars=self.use_carver_scalars,
            carver_breakout_scalars=self.carver_breakout_scalars,
        )

        # Store results
        self.pnl_dict = pnl_dict
        self.signal_dict = signal_dict
        self.summary_df = summary_df
        self.position_dict = {
            t: data.get("positions")
            for t, data in (pnl_dict or {}).items()
            if isinstance(data, dict) and data.get("positions") is not None
        }

        return self.pnl_dict, self.signal_dict, self.summary_df

    def get_summary(self) -> Optional[pd.DataFrame]:
        """Returns the summary DataFrame from the last run."""
        if self.summary_df is None:
            print(
                "No summary available. Please run the backtest first using the .run() method."
            )
        return self.summary_df


def plot_final_results(
    backtest_runner: CarverBacktest, optimized_params: dict, ticker: str
):
    """
    Runs the backtest with optimal parameters and plots the PnL curve, signal,
    and price for a single specified ticker.

    Args:
        backtest_runner (CarverBacktest): The configured backtest runner instance.
        optimized_params (dict): The dictionary of optimized parameters from .run_optimization().
        ticker (str): The specific ticker to plot.
    """
    print("\n" + "=" * 50)
    print(f"### Generating Final Results Plot for Ticker: {ticker} ###")
    print("=" * 50)

    if ticker not in backtest_runner.tickers:
        print(f"Error: Ticker '{ticker}' not found in the backtest instance.")
        return

    if ticker not in optimized_params:
        print(f"Error: Optimized parameters for ticker '{ticker}' not found.")
        return

    # 1. Run the backtest to generate PnL and signal data with optimal weights
    pnl_dict, signal_dict, _ = backtest_runner.run(
        optimized_params=optimized_params, show_progress=False
    )

    # 2. Extract the relevant data for the chosen ticker
    ticker_pnl = pnl_dict.get(ticker, {}).get("pnl_ccy")
    ticker_signal = signal_dict.get(ticker)
    ticker_price = backtest_runner.price_frame[ticker]

    if ticker_pnl is None or ticker_signal is None:
        print(f"Error: Could not retrieve PnL or signal data for '{ticker}'.")
        return

    # 3. Print the optimal parameters found for this ticker
    optimal_info = optimized_params[ticker]
    print(f"Optimal Parameters for {ticker}:")
    sharpe_val = optimal_info.get("sharpe")
    if isinstance(sharpe_val, (int, float, np.floating)) and np.isfinite(sharpe_val):
        sharpe_str = f"{sharpe_val:.4f}"
    else:
        sharpe_str = "N/A"
    print(f"  - Sharpe Ratio: {sharpe_str}")
    weights = list(optimal_info.get("best_weights", []))
    rules = (
        optimal_info.get("eq_spans")
        or optimal_info.get("spans")
        or optimal_info.get("pairs")
        or optimal_info.get("horizons")
        or optimal_info.get("rule_variations")
    )
    if rules and weights:
        print("  - Optimal Weights:")
        for rule, weight in zip(rules, weights):
            if weight > 0.01:  # Only show significant weights
                print(f"    - Rule '{rule}': {weight:.2%}")
    elif rules:
        print("  - Optimal Weights: unavailable in results")

    # 4. Create the plot
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f"Cumulative PnL for {ticker}", "Forecast Signal vs. Price"),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    # Plot 1: Cumulative PnL
    cumulative_pnl = ticker_pnl.cumsum()
    fig.add_trace(
        go.Scatter(
            x=cumulative_pnl.index,
            y=cumulative_pnl,
            name="Cumulative PnL",
            line=dict(color="purple"),
        ),
        row=1,
        col=1,
    )

    # Plot 2: Signal and Price
    fig.add_trace(
        go.Scatter(
            x=ticker_signal.index,
            y=ticker_signal,
            name="Forecast Signal",
            line=dict(color="orange"),
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=ticker_price.index,
            y=ticker_price,
            name="Price",
            line=dict(color="blue", width=1),
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    # Style the plot
    fig.update_layout(
        title_text=f"Backtest Results for {ticker} ({backtest_runner.strategy_type})",
        height=700,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Cumulative PnL ($)", row=1, col=1)
    fig.update_yaxes(title_text="Forecast Strength", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Price", row=2, col=1, secondary_y=True)

    fig.show()


# ============================================================
# 1) Build trade labels from signal + (optionally) positions
# ============================================================
def prepare_trade_actions_from_signal(
    signal: pd.Series,
    positions: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (action_open, action_close, action_single).

    action_open:  1 = open long (signal crosses up through 0)
                  -1 = open short (signal crosses down through 0)
    action_close: -2 = close long,  +2 = close short
                  (if positions provided, closes come from position -> flat;
                   else from signal crossing back through 0)
    action_single: a single compatibility column combining both
                   (close codes take precedence if both happen same bar)
    """
    s = pd.Series(signal, copy=True).astype(float)
    prev_s = s.shift(1).fillna(0.0)

    # Opens come from zero-crossings of the signal
    open_long = (s > 0) & (prev_s <= 0)
    open_short = (s < 0) & (prev_s >= 0)

    action_open = pd.Series(0, index=s.index, dtype=int)
    action_open.loc[open_long] = 1
    action_open.loc[open_short] = -1

    # Closes: prefer actual positions if you have them (reflects buffering, deadband, etc.)
    action_close = pd.Series(0, index=s.index, dtype=int)
    if positions is not None:
        pos_now = np.sign(pd.Series(positions, copy=False).fillna(0.0))
        pos_prev = pos_now.shift(1).fillna(0.0)

        close_long = (pos_prev > 0) & (pos_now <= 0)
        close_short = (pos_prev < 0) & (pos_now >= 0)

        action_close.loc[close_long] = -2
        action_close.loc[close_short] = 2
    else:
        # Fallback: define closes from signal crossing back through 0
        close_long = (prev_s > 0) & (s <= 0)
        close_short = (prev_s < 0) & (s >= 0)
        action_close.loc[close_long] = -2
        action_close.loc[close_short] = 2

    # Single compatibility column (close takes precedence if both occur on same bar)
    action_single = pd.Series(0, index=s.index, dtype=int)
    mask_close = action_close != 0
    mask_open = (action_open != 0) & ~mask_close
    action_single.loc[mask_close] = action_close.loc[mask_close]
    action_single.loc[mask_open] = action_open.loc[mask_open]

    return action_open, action_close, action_single


# ============================================================
# 2) Plot signals & trades (opens and closes shown separately)
# ============================================================
def plot_signals_and_trades_v2(
    data: pd.DataFrame,
    price_col: str,
    signal_col: str,
    open_action_col: str = "action_open",
    close_action_col: str = "action_close",
    title_suffix: str = "",
):
    """Price + signal with explicit Buy/Sell (open) and Close markers."""
    fig = go.Figure()

    # Price (left y) and signal (right y)
    fig.add_trace(
        go.Scatter(x=data.index, y=data[price_col], mode="lines", name=price_col)
    )
    fig.add_trace(
        go.Scatter(
            x=data.index, y=data[signal_col], mode="lines", name=signal_col, yaxis="y2"
        )
    )

    # Open markers from signal zero-crossings
    open_long = data[data[open_action_col] == 1]
    open_short = data[data[open_action_col] == -1]
    fig.add_trace(
        go.Scatter(
            x=open_long.index,
            y=open_long[price_col],
            mode="markers",
            marker=dict(color="green", size=10, symbol="triangle-up"),
            name="Buy (open long)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=open_short.index,
            y=open_short[price_col],
            mode="markers",
            marker=dict(color="red", size=10, symbol="triangle-down"),
            name="Sell (open short)",
        )
    )

    # Close markers (from positions -> flat if provided)
    close_long = data[data[close_action_col] == -2]
    close_short = data[data[close_action_col] == 2]
    fig.add_trace(
        go.Scatter(
            x=close_long.index,
            y=close_long[price_col],
            mode="markers",
            marker=dict(color="black", size=8, symbol="x"),
            name="Close Long",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=close_short.index,
            y=close_short[price_col],
            mode="markers",
            marker=dict(color="purple", size=8, symbol="x"),
            name="Close Short",
        )
    )

    fig.update_layout(
        title=f"Trade Signals vs. Price {title_suffix}",
        xaxis_title="Date",
        yaxis=dict(title=price_col),
        yaxis2=dict(title=signal_col, overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        width=1300,
        height=600,
        template="plotly_white",
    )
    fig.show()


# ============================================================
# 3) Performance plots (equity / drawdown / daily PnL)
# ============================================================
def plot_trading_performance(performance_df: pd.DataFrame, abs_dd: bool = True):
    """Equity curve, drawdown, daily PnL, and PnL distribution."""
    perf = performance_df.copy()
    if abs_dd:
        perf["drawdown"] = perf["equity_curve"] - perf["equity_curve"].cummax()
    else:
        perf["drawdown"] = (
            perf["equity_curve"] - perf["equity_curve"].cummax()
        ) / perf["equity_curve"].cummax()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=("Equity Curve", "Drawdown", "Daily PnL", "PnL Distribution"),
    )
    fig.add_trace(
        go.Scatter(
            x=perf.index, y=perf["equity_curve"], mode="lines", name="Equity Curve"
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=perf.index, y=perf["drawdown"], name="Drawdown"), row=2, col=1
    )
    fig.add_trace(
        go.Bar(x=perf.index, y=perf["daily_pnl"], name="Daily PnL"), row=3, col=1
    )
    fig.add_trace(
        go.Histogram(x=perf["daily_pnl"], name="PnL Distribution", nbinsx=100),
        row=4,
        col=1,
    )

    fig.update_layout(
        height=1400,
        width=1300,
        title_text="Trading Performance",
        showlegend=False,
        template="plotly_white",
    )
    fig.show()


# ============================================================
# 4) End-to-end: prepare DF and plot both figures
# ============================================================
def plot_detailed_backtest_analytics(backtest_runner, ticker: str):
    """
    Generate detailed signal, position, and PnL plots for a single ticker.

    Call `backtest_runner.run(...)` first so that pnl_dict/signal_dict are populated.
    Positions are pulled from `backtest_runner.position_dict` when available or from
    `backtest_runner.pnl_dict[ticker]['positions']` as a fallback.
    """
    print("\n" + "=" * 50)
    print(f"### Generating Detailed Analytics for Ticker: {ticker} ###")
    print("=" * 50)

    pnl_store = getattr(backtest_runner, "pnl_dict", None) or {}
    signal_store = getattr(backtest_runner, "signal_dict", None) or {}
    position_store = getattr(backtest_runner, "position_dict", None) or {}

    if ticker not in signal_store or ticker not in pnl_store:
        print("Error: missing signal or PnL for this ticker. Run the backtest first.")
        return

    pnl_entry = pnl_store.get(ticker) or {}
    position_series = position_store.get(ticker) or pnl_entry.get("positions")
    if position_series is None:
        print("Error: no position series available for this ticker.")
        return

    pnl_series = pnl_entry.get("pnl_ccy")
    if pnl_series is None:
        print("Error: no PnL series available for this ticker.")
        return

    price = pd.Series(backtest_runner.price_frame[ticker], copy=False).astype(float)
    idx = price.index
    signal = (
        pd.Series(signal_store[ticker], copy=False)
        .reindex(idx)
        .astype(float)
        .fillna(0.0)
    )
    positions = (
        pd.Series(position_series, copy=False).reindex(idx).astype(float).fillna(0.0)
    )
    pnl = pd.Series(pnl_series, copy=False).reindex(idx).astype(float).fillna(0.0)

    df = pd.DataFrame(
        {
            "price": price,
            "signal": signal,
            "position": positions,
            "daily_pnl": pnl,
        }
    ).dropna(subset=["price"])

    if df.empty:
        print("Error: no overlapping data to plot for this ticker.")
        return

    action_open, action_close, action_single = prepare_trade_actions_from_signal(
        signal=df["signal"], positions=df["position"]
    )
    df["action_open"] = action_open
    df["action_close"] = action_close
    df["action"] = action_single  # optional single column

    df["equity_curve"] = backtest_runner.capital + df["daily_pnl"].cumsum()

    print("Displaying signals and trades plot...")
    plot_signals_and_trades_v2(
        df,
        price_col="price",
        signal_col="signal",
        open_action_col="action_open",
        close_action_col="action_close",
        title_suffix=f"({ticker})",
    )

    print("\nDisplaying trading performance plot...")
    plot_trading_performance(df, abs_dd=True)


def estimate_periods_per_year(index: pd.Index) -> float:
    """Estimate the number of observations per year from a datetime index."""
    if not isinstance(index, pd.DatetimeIndex) or index.size < 2:
        return float(BUSDAYS)

    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return float(BUSDAYS)

    avg_days = diffs.dt.total_seconds().mean() / 86_400.0
    if avg_days <= 0:
        return float(BUSDAYS)

    return float(365.25 / avg_days)


def carver_backtest_stats(
    pnl_ccy: pd.Series,
    *,
    capital: float,
    position: pd.Series | None = None,
    costs_ccy: pd.Series | None = None,
    periods_per_year: int | None = None,
) -> dict[str, float]:
    """
    Returns a dict with:
      mean_ann_return, ann_costs, avg_drawdown, ann_std, sharpe,
      turnover_pa, skew, lower_tail, upper_tail
    """
    pnl = pd.Series(pnl_ccy).astype(float).dropna()
    if pnl.empty:
        return {
            k: np.nan
            for k in [
                "mean_ann_return",
                "ann_costs",
                "avg_drawdown",
                "ann_std",
                "sharpe",
                "turnover_pa",
                "skew",
                "lower_tail",
                "upper_tail",
            ]
        }

    rets = pnl / float(capital)
    ppy = periods_per_year or estimate_periods_per_year(pnl.index)

    mean_ann_return = rets.mean() * ppy
    ann_std = rets.std(ddof=0) * np.sqrt(ppy)
    sharpe = mean_ann_return / ann_std if ann_std > 0 else np.nan

    # Drawdown on equity (constant capital base)
    equity = capital + pnl.cumsum()
    dd = equity - equity.cummax()
    avg_drawdown = -dd[dd < 0].mean() if (dd < 0).any() else 0.0

    # Costs (if you tracked them separately); otherwise NaN
    if costs_ccy is not None:
        c = pd.Series(costs_ccy).reindex(pnl.index).fillna(0.0)
        ann_costs = (c / float(capital)).mean() * ppy
    else:
        ann_costs = np.nan

    # Turnover: average |Delta position| per period, annualised
    turnover_pa = np.nan
    if position is not None:
        pos = pd.Series(position).reindex(pnl.index).fillna(0.0)
        dpos = pos.diff().abs().fillna(0.0)
        turnover_pa = dpos.mean() * ppy

    # Distribution shape
    skew = rets.skew()
    lower_tail = rets.quantile(0.05) * ppy**0.5  # report as ann-equivalent per sqrt(T)
    upper_tail = rets.quantile(0.95) * ppy**0.5

    return {
        "mean_ann_return": float(mean_ann_return),
        "ann_costs": float(ann_costs) if np.isfinite(ann_costs) else np.nan,
        "avg_drawdown": float(avg_drawdown),
        "ann_std": float(ann_std),
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "turnover_pa": float(turnover_pa) if np.isfinite(turnover_pa) else np.nan,
        "skew": float(skew) if np.isfinite(skew) else np.nan,
        "lower_tail": float(lower_tail),
        "upper_tail": float(upper_tail),
    }


def summarize_runner_stats(backtest_runner) -> pd.DataFrame:
    """
    Build a stats table across all tickers held inside your runner.
    Assumes `backtest_runner.run(...)` has been executed.
    """
    pnl_store = getattr(backtest_runner, "pnl_dict", None) or {}
    position_store = getattr(backtest_runner, "position_dict", None) or {}

    rows = {}
    for t in backtest_runner.tickers:
        pnl_entry = pnl_store.get(t)
        if not isinstance(pnl_entry, dict):
            continue

        pnl = pnl_entry.get("pnl_ccy")
        if pnl is None:
            continue

        pos = position_store.get(t) if position_store else pnl_entry.get("positions")
        costs = pnl_entry.get("costs_ccy")

        rows[t] = carver_backtest_stats(
            pnl_ccy=pnl,
            capital=backtest_runner.capital,
            position=pos,
            costs_ccy=costs,
        )

    return pd.DataFrame.from_dict(rows, orient="index").sort_index()
