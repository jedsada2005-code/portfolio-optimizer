"""Portfolio optimisation helpers built on PyPortfolioOpt.

Kept out of app.py so the constraint handling and the covariance
treatment can be unit tested without a Streamlit session.
"""

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier
from pypfopt.exceptions import OptimizationError


DEFAULT_SHRINKAGE = 0.2


def average_correlation(cov):
    """Mean off-diagonal correlation implied by a covariance matrix."""
    sd = np.sqrt(np.diag(cov.values))
    corr = cov.values / np.outer(sd, sd)
    n = corr.shape[0]
    if n < 2:
        return 0.0
    off = corr[~np.eye(n, dtype=bool)]
    return float(off.mean())


def constant_correlation_target(cov):
    """Shrinkage target: every pair shares the average correlation.

    Keeps each asset's own variance and replaces the pairwise structure,
    which is the part a short sample estimates worst.
    """
    sd = np.sqrt(np.diag(cov.values))
    rho = average_correlation(cov)
    target = rho * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(cov.values))
    return pd.DataFrame(target, index=cov.index, columns=cov.columns)


def nearest_psd(cov):
    """Clip negative eigenvalues so the solver gets a usable matrix."""
    values = (cov.values + cov.values.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(values)
    if eigvals.min() >= 0:
        repaired = values
    else:
        repaired = eigvecs @ np.diag(np.clip(eigvals, 0.0, None)) @ eigvecs.T
        repaired = (repaired + repaired.T) / 2.0
    return pd.DataFrame(repaired, index=cov.index, columns=cov.columns)


def shrink_covariance(cov, intensity=DEFAULT_SHRINKAGE):
    """Blend the sample covariance towards a constant-correlation target.

    PyPortfolioOpt's own Ledoit-Wolf runs ``np.nan_to_num`` over the
    return matrix, which turns every pre-inception period into a 0%
    return -- the same distortion the backtest was fixed for. Shrinking
    the already-computed pairwise covariance avoids touching the raw
    returns at all.
    """
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("intensity ต้องอยู่ระหว่าง 0 ถึง 1")
    if cov.shape[0] < 2:
        return cov.copy()
    if intensity == 0.0:
        return nearest_psd(cov)
    target = constant_correlation_target(cov)
    blended = (1.0 - intensity) * cov.values + intensity * target.values
    return nearest_psd(pd.DataFrame(blended, index=cov.index, columns=cov.columns))


def max_achievable_return(expected_returns, max_weight):
    """Highest expected return reachable under a per-asset weight cap."""
    n = len(expected_returns)
    if max_weight * n < 1.0 - 1e-9:
        raise ValueError(
            f"น้ำหนักสูงสุดต่อสินทรัพย์ {max_weight:.0%} × {n} ตัว ไม่ถึง 100% "
            "— ต้องเพิ่มเพดานน้ำหนัก หรือเพิ่มจำนวนสินทรัพย์"
        )
    remaining = 1.0
    total = 0.0
    for value in expected_returns.sort_values(ascending=False):
        take = min(max_weight, remaining)
        total += take * value
        remaining -= take
        if remaining <= 1e-12:
            break
    return float(total)


def _frontier(expected_returns, cov, max_weight):
    return EfficientFrontier(expected_returns, cov, weight_bounds=(0.0, max_weight))


def optimize_weights(expected_returns, cov, objective, risk_free_rate, max_weight=1.0):
    """Solve for one objective, raising ValueError with a readable message."""
    max_achievable_return(expected_returns, max_weight)  # validates the cap
    ef = _frontier(expected_returns, cov, max_weight)
    if objective == "Min Volatility":
        ef.min_volatility()
    else:
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    return dict(ef.clean_weights())


def portfolio_performance(expected_returns, cov, weights, risk_free_rate):
    """Expected return, volatility and Sharpe for an explicit weight set."""
    w = np.array([float(weights.get(a, 0.0)) for a in expected_returns.index])
    ret = float(w @ expected_returns.values)
    vol = float(np.sqrt(max(w @ cov.values @ w, 0.0)))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def frontier_curve(expected_returns, cov, max_weight=1.0, n_points=60):
    """Trace the efficient frontier by solving for target returns.

    Replaces reading the line data back out of a throwaway matplotlib
    figure, which depended on pypfopt's internal plot ordering.
    """
    ef = _frontier(expected_returns, cov, max_weight)
    ef.min_volatility()
    low = ef.portfolio_performance()[0]
    high = max_achievable_return(expected_returns, max_weight)
    if high <= low:
        vol = ef.portfolio_performance()[1]
        return np.array([vol]), np.array([low])

    vols, rets = [], []
    for target in np.linspace(low, high, n_points):
        try:
            point = _frontier(expected_returns, cov, max_weight)
            point.efficient_return(target)
            ret, vol, _ = point.portfolio_performance()
        except (ValueError, OptimizationError):
            continue
        vols.append(vol)
        rets.append(ret)
    return np.array(vols), np.array(rets)


def sample_weights(rng, n_assets, n_samples, max_weight=1.0, max_passes=12):
    """Random long-only weight vectors, optionally capped per asset.

    Plain rejection sampling empties out fast once the cap approaches
    equal weight, so capped draws are clipped and renormalised until
    they fit, which keeps the scatter cloud populated.
    """
    if max_weight * n_assets < 1.0 - 1e-9:
        return np.empty((0, n_assets))

    w = rng.dirichlet([0.5] * n_assets, n_samples)
    if max_weight >= 1.0:
        return w

    for _ in range(max_passes):
        over = w > max_weight
        if not over.any():
            break
        w = np.minimum(w, max_weight)
        deficit = 1.0 - w.sum(axis=1, keepdims=True)
        headroom = max_weight - w
        total_headroom = headroom.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(total_headroom > 0, headroom / total_headroom, 0.0)
        w = w + share * deficit

    w = np.minimum(w, max_weight)
    totals = w.sum(axis=1, keepdims=True)
    keep = np.isclose(totals[:, 0], 1.0, atol=1e-6)
    return w[keep]
