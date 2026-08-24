"""Portfolio optimisation helpers built on PyPortfolioOpt.

Kept out of app.py so the constraint handling and the covariance
treatment can be unit tested without a Streamlit session.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, HRPOpt, expected_returns as pypfopt_returns
from pypfopt.exceptions import OptimizationError

try:  # cvxpy raises this straight through pypfopt on a degenerate problem
    from cvxpy.error import SolverError
except ImportError:  # pragma: no cover - cvxpy ships with pypfopt
    class SolverError(Exception):
        pass

# SolverError descends from neither ValueError nor OptimizationError, so
# handlers catching those let it escape all the way to the page.
SOLVER_ERRORS = (ValueError, OptimizationError, SolverError)

import metrics


DEFAULT_SHRINKAGE = 0.2

WEEKS_PER_YEAR = 52

# How to estimate what each asset will return. This is the single
# largest influence on the resulting weights -- swapping the method can
# move an allocation from 88% in one asset to 26% in another -- so it
# belongs in the user's hands rather than hardcoded.
RETURN_METHODS = {
    "ค่าเฉลี่ยผลตอบแทน 1 ปี (ทับซ้อน)": "rolling_annual",
    "ค่าเฉลี่ยตลอดช่วง": "mean_historical",
    "ถ่วงน้ำหนักข้อมูลล่าสุด (EMA)": "ema_historical",
    "CAPM (อิงความเสี่ยงเทียบตลาด)": "capm",
}
DEFAULT_RETURN_METHOD = "ค่าเฉลี่ยผลตอบแทน 1 ปี (ทับซ้อน)"

HRP_OBJECTIVE = "HRP (Risk Parity)"


def estimate_returns(weekly_prices, method):
    """Annualised expected returns from weekly prices."""
    key = RETURN_METHODS.get(method)
    if key is None:
        raise ValueError(f"ไม่รู้จักวิธีประมาณผลตอบแทน: {method}")
    if key == "rolling_annual":
        return metrics.annual_return_estimates(weekly_prices)[0]
    if key == "mean_historical":
        return pypfopt_returns.mean_historical_return(
            weekly_prices, frequency=WEEKS_PER_YEAR
        )
    if key == "ema_historical":
        return pypfopt_returns.ema_historical_return(
            weekly_prices, frequency=WEEKS_PER_YEAR, span=2 * WEEKS_PER_YEAR
        )
    return pypfopt_returns.capm_return(weekly_prices, frequency=WEEKS_PER_YEAR)


def estimate_returns_with_counts(weekly_prices, method):
    """Expected returns plus how many observations back each one.

    The count always comes from the trailing-year windows, whatever the
    estimator, because that is what the history floor is expressed in.
    """
    _, counts = metrics.annual_return_estimates(weekly_prices)
    return estimate_returns(weekly_prices, method), counts


def hrp_weights(weekly_prices):
    """Hierarchical risk parity: allocation from the covariance alone.

    Needs no expected returns at all, which sidesteps the noisiest input
    in the whole exercise and never leaves an asset at zero.
    """
    returns = weekly_prices.pct_change().dropna(how="all")
    optimiser = HRPOpt(returns)
    optimiser.optimize()
    weights = dict(optimiser.clean_weights())
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()} if total > 0 else weights


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


def validate_bounds(n_assets, max_weight, min_weight=0.0):
    """Reject weight bounds no allocation can satisfy.

    The solver's own message for an infeasible problem is "Please check
    your objectives/constraints", which does not say which bound is
    wrong or what would work.
    """
    if min_weight > max_weight:
        raise ValueError(
            f"น้ำหนักขั้นต่ำ {min_weight:.0%} มากกว่าน้ำหนักสูงสุด {max_weight:.0%}"
        )
    if min_weight * n_assets > 1.0 + 1e-9:
        raise ValueError(
            f"น้ำหนักขั้นต่ำ {min_weight:.0%} × {n_assets} ตัว = "
            f"{min_weight * n_assets:.0%} เกิน 100% — ตั้งได้สูงสุด {1 / n_assets:.1%}"
        )
    if max_weight * n_assets < 1.0 - 1e-9:
        raise ValueError(
            f"น้ำหนักสูงสุดต่อสินทรัพย์ {max_weight:.0%} × {n_assets} ตัว ไม่ถึง 100% "
            f"— ต้องตั้งอย่างน้อย {1 / n_assets:.1%} หรือเพิ่มจำนวนสินทรัพย์"
        )


def max_achievable_return(expected_returns, max_weight, min_weight=0.0):
    """Highest expected return reachable under the weight bounds.

    Every asset is funded to the floor first; only what remains can be
    steered towards the best performers, and no asset may take more
    than ``max_weight - min_weight`` of it.
    """
    n = len(expected_returns)
    validate_bounds(n, max_weight, min_weight)
    remaining = 1.0 - min_weight * n
    total = min_weight * float(expected_returns.sum())
    headroom = max_weight - min_weight
    for value in expected_returns.sort_values(ascending=False):
        take = min(headroom, remaining)
        total += take * value
        remaining -= take
        if remaining <= 1e-12:
            break
    return float(total)


def _frontier(expected_returns, cov, max_weight, min_weight=0.0):
    return EfficientFrontier(
        expected_returns, cov, weight_bounds=(min_weight, max_weight)
    )


def optimize_weights(expected_returns, cov, objective, risk_free_rate,
                     max_weight=1.0, min_weight=0.0, weekly=None):
    """Solve for one objective, raising ValueError with a readable message."""
    if objective == HRP_OBJECTIVE:
        if weekly is None:
            raise ValueError("HRP ต้องใช้ประวัติผลตอบแทน ไม่ใช่แค่ค่าคาดหวัง")
        return hrp_weights(weekly)
    validate_bounds(len(expected_returns), max_weight, min_weight)
    ef = _frontier(expected_returns, cov, max_weight, min_weight)
    try:
        if objective == "Min Volatility":
            ef.min_volatility()
        else:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
    except SolverError as exc:
        raise OptimizationError(
            "solver แก้ปัญหานี้ไม่ได้ — ข้อมูลอาจสั้นเกินไปจนเมทริกซ์ความแปรปรวนร่วมผิดรูป "
            f"({exc})"
        ) from exc
    # clean_weights rounds to five places, which can leave the set
    # summing to 1.00002 -- harmless to the simulator, which normalises
    # anyway, but it shows up in the weights table and the workbook.
    cleaned = dict(ef.clean_weights())
    total = sum(cleaned.values())
    if total > 0:
        cleaned = {asset: weight / total for asset, weight in cleaned.items()}
    return cleaned


def portfolio_performance(expected_returns, cov, weights, risk_free_rate):
    """Expected return, volatility and Sharpe for an explicit weight set."""
    w = np.array([float(weights.get(a, 0.0)) for a in expected_returns.index])
    ret = float(w @ expected_returns.values)
    vol = float(np.sqrt(max(w @ cov.values @ w, 0.0)))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def frontier_curve(expected_returns, cov, max_weight=1.0, n_points=60, min_weight=0.0):
    """Trace the efficient frontier by solving for target returns.

    Replaces reading the line data back out of a throwaway matplotlib
    figure, which depended on pypfopt's internal plot ordering.
    """
    ef = _frontier(expected_returns, cov, max_weight, min_weight)
    try:
        ef.min_volatility()
    except SOLVER_ERRORS:
        return np.array([]), np.array([])
    low = ef.portfolio_performance()[0]
    high = max_achievable_return(expected_returns, max_weight, min_weight)
    if high - low <= 1e-9:
        vol = ef.portfolio_performance()[1]
        return np.array([vol]), np.array([low])

    vols, rets = [], []
    for target in np.linspace(low, high, n_points):
        try:
            point = _frontier(expected_returns, cov, max_weight, min_weight)
            point.efficient_return(target)
            ret, vol, _ = point.portfolio_performance()
        except SOLVER_ERRORS:
            continue
        vols.append(vol)
        rets.append(ret)
    return np.array(vols), np.array(rets)


def sample_weights(rng, n_assets, n_samples, max_weight=1.0, max_passes=12, min_weight=0.0):
    """Random long-only weight vectors, optionally capped per asset.

    Plain rejection sampling empties out fast once the cap approaches
    equal weight, so capped draws are clipped and renormalised until
    they fit, which keeps the scatter cloud populated.
    """
    try:
        validate_bounds(n_assets, max_weight, min_weight)
    except ValueError:
        return np.empty((0, n_assets))

    # Fund the floor first, then spread what is left. Rejecting draws
    # that breach the floor would discard nearly all of them.
    w = rng.dirichlet([0.5] * n_assets, n_samples)
    if min_weight > 0:
        w = min_weight + (1.0 - min_weight * n_assets) * w
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

    w = np.clip(w, min_weight, max_weight)
    totals = w.sum(axis=1, keepdims=True)
    keep = np.isclose(totals[:, 0], 1.0, atol=1e-6)
    return w[keep]


WalkForwardResult = namedtuple("WalkForwardResult", "returns weight_history turnover")


def fit_weights(prices, risk_free_rate, objective, max_weight, shrinkage,
                min_weight=0.0, return_method=DEFAULT_RETURN_METHOD,
                min_observations=None):
    """Expected returns and covariance from a price window, then solve."""
    weekly = prices.resample("W-FRI").last()
    expected, observations = estimate_returns_with_counts(weekly, return_method)
    floor = (
        metrics.MIN_ANNUAL_OBSERVATIONS if min_observations is None else min_observations
    )
    usable = [asset for asset in expected.index if int(observations[asset]) >= floor]
    if len(usable) < 2:
        return None
    expected = expected[usable]
    cov = shrink_covariance(weekly[usable].pct_change().cov() * 52, shrinkage)
    try:
        return optimize_weights(
            expected, cov, objective, risk_free_rate, max_weight, min_weight,
            weekly=weekly[usable],
        )
    except SOLVER_ERRORS:
        return None


def walk_forward(
    prices, risk_free_rate, objective, max_weight, shrinkage,
    refit_freq="Y", cost_bps=0.0, min_train_years=2.0,
    cash_fraction=0.0, rebalance_freq=None, min_weight=0.0,
    return_method=DEFAULT_RETURN_METHOD, min_observations=None,
):
    """Re-fit on a schedule using only data available at that moment.

    A single train/test split gives one out-of-sample observation. This
    re-solves at every refit date from history up to that date and holds
    the result until the next one, so the whole tested period is out of
    sample and the weights never see their own future.

    ``prices`` must hold the risky assets only. The cash sleeve is added
    here, after each fit, so it stays the fixed allocation the user
    asked for rather than becoming another asset the optimiser is free
    to decline -- and so a near-zero-variance column never reaches the
    covariance matrix.
    """
    empty = pd.Series(dtype=float)
    refit_dates = metrics.rebalance_dates(prices.index, refit_freq)
    earliest = prices.index[0] + pd.Timedelta(days=min_train_years * metrics.DAYS_PER_YEAR)
    refit_dates = [d for d in refit_dates if d >= earliest]
    if not refit_dates:
        return WalkForwardResult(empty, [], empty)

    sim_prices = prices
    if cash_fraction > 0:
        sim_prices = prices.assign(**{
            metrics.CASH_SYMBOL: metrics.cash_price_series(prices.index, risk_free_rate)
        })

    segments, history, turnovers = [], [], []
    previous_weights = {}
    for position, refit_date in enumerate(refit_dates):
        weights = fit_weights(
            prices.loc[:refit_date], risk_free_rate, objective, max_weight,
            shrinkage, min_weight, return_method, min_observations,
        )
        if weights is None:
            continue
        weights = metrics.blend_with_cash(weights, cash_fraction)
        history.append((refit_date, weights))

        end = refit_dates[position + 1] if position + 1 < len(refit_dates) else prices.index[-1]
        window = sim_prices.loc[refit_date:end]
        if len(window) < 2:
            continue

        # Between refits the portfolio follows the user's own rebalance
        # schedule; the refit itself is charged separately below.
        segment = metrics.simulate_portfolio(window, weights, rebalance_freq, cost_bps)
        if segment.returns.empty:
            continue

        # The opening row of every segment is a 0.0 base day that repeats
        # the previous segment's final date, so drop it before charging
        # the refit -- otherwise the cost is applied to a row that is
        # then discarded as a duplicate.
        returns = segment.returns.iloc[1:] if segments else segment.returns.copy()
        if returns.empty:
            continue

        assets = set(previous_weights) | set(weights)
        traded = sum(
            abs(weights.get(a, 0.0) - previous_weights.get(a, 0.0)) for a in assets
        ) if previous_weights else 0.0
        if cost_bps and traded:
            returns = returns.copy()
            returns.iloc[0] -= traded * cost_bps / 10_000.0

        # Turnover has two sources now: the refit itself, and whatever
        # the rebalance schedule traded inside the segment.
        segment_turnover = segment.turnover.reindex(returns.index).fillna(0.0)
        if traded:
            segment_turnover.iloc[0] += traded
        turnovers.append(segment_turnover)

        segments.append(returns)
        previous_weights = segment.final_weights or weights

    if not segments:
        return WalkForwardResult(empty, history, empty)

    chained = pd.concat(segments)
    chained = chained[~chained.index.duplicated(keep="first")].sort_index()
    turnover = pd.concat(turnovers) if turnovers else empty
    turnover = turnover[~turnover.index.duplicated(keep="first")].sort_index()
    return WalkForwardResult(chained, history, turnover)
