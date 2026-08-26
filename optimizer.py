"""Portfolio optimisation helpers built on PyPortfolioOpt.

Kept out of app.py so the constraint handling and the covariance
treatment can be unit tested without a Streamlit session.
"""

from collections import namedtuple

import numpy as np
import pandas as pd
from pypfopt import (
    EfficientCVaR,
    EfficientFrontier,
    EfficientSemivariance,
    HRPOpt,
    expected_returns as pypfopt_returns,
)
from pypfopt.exceptions import OptimizationError

import cvxpy
import scipy.cluster.hierarchy as _sch

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

# Two families. The first works on the mean-variance frontier and
# answers "given this much risk"; the second changes what risk means, or
# stops estimating returns altogether.
TARGET_VOLATILITY = "กำหนดความเสี่ยงเป้าหมาย"
TARGET_RETURN = "กำหนดผลตอบแทนเป้าหมาย"
HRP_OBJECTIVE = "HRP (Risk Parity)"
MIN_CVAR = "Min CVaR"
MIN_SEMIVARIANCE = "Min Semivariance"

# Not an objective: the naive 1/N portfolio, carried in comparisons as
# the baseline every optimiser has to beat to have earned its complexity.
EQUAL_WEIGHT = "ลงเท่ากันทุกตัว (1/N)"

MPT_OBJECTIVES = ["Max Sharpe", "Min Volatility", TARGET_VOLATILITY, TARGET_RETURN]
ALTERNATIVE_OBJECTIVES = [HRP_OBJECTIVE, MIN_CVAR, MIN_SEMIVARIANCE]
OBJECTIVES = MPT_OBJECTIVES + ALTERNATIVE_OBJECTIVES

DOWNSIDE_SOLVER = "CLARABEL"

NEEDS_TARGET = {TARGET_VOLATILITY, TARGET_RETURN}
NEEDS_HISTORY = {HRP_OBJECTIVE, MIN_CVAR, MIN_SEMIVARIANCE}


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


# The seven methods scipy.cluster.hierarchy.linkage documents. pypfopt
# 1.6.0 -- still its latest release -- validates its linkage argument
# against scipy's private _LINKAGE_METHODS mapping, which newer scipy no
# longer defines. The name is used for that check and nothing else;
# every line after it in HRPOpt.optimize is public API. Since app.py
# solves every objective on every run, the AttributeError took the whole
# page down rather than merely disabling HRP.
LINKAGE_METHODS = (
    "single", "complete", "average", "weighted", "centroid", "median", "ward",
)


def _restore_linkage_methods():
    """Put back the private name pypfopt validates against, if it is gone.

    Deliberately done per call rather than once at import, so it holds
    however scipy is loaded or reloaded around us.
    """
    if not hasattr(_sch, "_LINKAGE_METHODS"):
        _sch._LINKAGE_METHODS = {
            name: index for index, name in enumerate(LINKAGE_METHODS)
        }


def hrp_weights(weekly_prices):
    """Hierarchical risk parity: allocation from the covariance alone.

    Needs no expected returns at all, which sidesteps the noisiest input
    in the whole exercise and never leaves an asset at zero.
    """
    _restore_linkage_methods()
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


def achievable_range(expected_returns, cov, max_weight=1.0, min_weight=0.0):
    """(lowest volatility, its return, highest return) under the bounds.

    Used to tell someone asking for an impossible target what they could
    have asked for instead.
    """
    ef = _frontier(expected_returns, cov, max_weight, min_weight)
    ef.min_volatility()
    min_return, min_volatility, _ = ef.portfolio_performance()
    return min_volatility, min_return, max_achievable_return(
        expected_returns, max_weight, min_weight
    )


def _downside_optimizer(objective, expected_returns, weekly, max_weight, min_weight):
    returns = weekly.pct_change().dropna(how="all")
    bounds = (min_weight, max_weight)
    # These build one variable per observation, which the default OSQP
    # backend abandons at its iteration limit once weight bounds tighten.
    # CLARABEL solves the same problems in hundredths of a second.
    backend = DOWNSIDE_SOLVER if DOWNSIDE_SOLVER in cvxpy.installed_solvers() else None
    options = {"solver": backend} if backend else {}
    if objective == MIN_CVAR:
        solver = EfficientCVaR(expected_returns, returns, weight_bounds=bounds, **options)
        solver.min_cvar()
    else:
        solver = EfficientSemivariance(
            expected_returns, returns, frequency=WEEKS_PER_YEAR,
            weight_bounds=bounds, **options,
        )
        solver.min_semivariance()
    weights = dict(solver.clean_weights())
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()} if total > 0 else weights


def optimize_weights(expected_returns, cov, objective, risk_free_rate,
                     max_weight=1.0, min_weight=0.0, weekly=None, target=None):
    """Solve for one objective, raising ValueError with a readable message."""
    if objective in NEEDS_HISTORY and weekly is None:
        raise ValueError(f"{objective} ต้องใช้ประวัติผลตอบแทน ไม่ใช่แค่ค่าคาดหวัง")
    if objective == HRP_OBJECTIVE:
        return hrp_weights(weekly)

    validate_bounds(len(expected_returns), max_weight, min_weight)
    if objective in (MIN_CVAR, MIN_SEMIVARIANCE):
        return _downside_optimizer(
            objective, expected_returns, weekly, max_weight, min_weight
        )

    if objective in NEEDS_TARGET and target is None:
        raise ValueError(f"{objective} ต้องระบุค่าเป้าหมาย")

    ef = _frontier(expected_returns, cov, max_weight, min_weight)
    try:
        if objective == "Min Volatility":
            ef.min_volatility()
        elif objective == TARGET_VOLATILITY:
            ef.efficient_risk(target)
        elif objective == TARGET_RETURN:
            ef.efficient_return(target)
        else:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
    except SOLVER_ERRORS as exc:
        if objective in NEEDS_TARGET:
            low_vol, low_ret, high_ret = achievable_range(
                expected_returns, cov, max_weight, min_weight
            )
            # pypfopt's OptimizationError prepends its own generic
            # sentence, which would bury the useful half in a tuple.
            if objective == TARGET_VOLATILITY:
                raise ValueError(
                    f"ความเสี่ยงเป้าหมาย {target:.2%} อยู่นอกช่วงที่ทำได้ "
                    f"— ต่ำสุดที่เป็นไปได้คือ {low_vol:.2%}"
                ) from exc
            raise ValueError(
                f"ผลตอบแทนเป้าหมาย {target:.2%} อยู่นอกช่วงที่ทำได้ "
                f"— ต่ำสุด {low_ret:.2%} สูงสุด {high_ret:.2%}"
            ) from exc
        raise ValueError(
            "solver แก้ปัญหานี้ไม่ได้ — ข้อมูลอาจสั้นเกินไปจนเมทริกซ์ความแปรปรวนร่วมผิดรูป"
        ) from exc
    # clean_weights rounds to five places, which can leave the set
    # summing to 1.00002 -- harmless to the simulator, which normalises
    # anyway, but it shows up in the weights table and the workbook.
    cleaned = dict(ef.clean_weights())
    total = sum(cleaned.values())
    if total > 0:
        cleaned = {asset: weight / total for asset, weight in cleaned.items()}
    return cleaned


def target_gap(objective, target, expected_return, volatility, tolerance=1e-3):
    """``(requested, achieved)`` when a target objective missed its target.

    ``efficient_risk`` maximises return subject to a volatility ceiling
    and ``efficient_return`` minimises volatility subject to a return
    floor, so a target on the slack side of either constraint is simply
    non-binding: the solver succeeds and hands back the nearest
    reachable portfolio without a word. Asking for 35% volatility on a
    frontier that ends at 19% is answered with the 19% portfolio.

    Returns None when the objective takes no target, none was given, or
    the target was reached.
    """
    if objective not in NEEDS_TARGET or target is None:
        return None
    achieved = volatility if objective == TARGET_VOLATILITY else expected_return
    if abs(achieved - target) <= tolerance:
        return None
    return float(target), float(achieved)


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


def compare_fixed_weights(weights_by_name, prices, risk_free_rate,
                          rebalance_freq=None, cost_bps=0.0, cash_fraction=0.0):
    """Backtest several weight sets through one identical simulation.

    Every row of a comparison has to come out of the same procedure on
    the same window, or the table ranks the procedures rather than the
    objectives it claims to be comparing.

    ``prices`` must already carry the cash column when ``cash_fraction``
    is non-zero, exactly as the single-portfolio backtest receives it.
    """
    table = {}
    for name, weights in weights_by_name.items():
        blended = metrics.blend_with_cash(weights, cash_fraction)
        simulation = metrics.simulate_portfolio(
            prices, blended, rebalance_freq, cost_bps
        )
        if simulation.returns.empty:
            continue
        table[name] = metrics.backtest_stats(simulation.returns, risk_free_rate)
    return table


def compare_walk_forward(objectives, prices, risk_free_rate, max_weight, shrinkage,
                         refit_freq="Y", cost_bps=0.0, targets=None, **walk_kwargs):
    """Walk each objective forward through the identical schedule.

    Reusing a fixed-weight backtest for the objectives the user did not
    select would put a re-fitted portfolio in one row and a static one in
    the next, which is the comparison this table exists to avoid.
    """
    targets = targets or {}
    table = {}
    for name in objectives:
        try:
            result = walk_forward(
                prices, risk_free_rate, name, max_weight, shrinkage,
                refit_freq, cost_bps, target=targets.get(name), **walk_kwargs
            )
        except Exception:  # noqa: BLE001 - one row must not cost the table
            continue
        if result.returns.empty:
            continue
        table[name] = metrics.backtest_stats(result.returns, risk_free_rate)
    return table


WalkForwardResult = namedtuple("WalkForwardResult", "returns weight_history turnover")


def fit_weights(prices, risk_free_rate, objective, max_weight, shrinkage,
                min_weight=0.0, return_method=DEFAULT_RETURN_METHOD,
                min_observations=None, target=None):
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
            weekly=weekly[usable], target=target,
        )
    except SOLVER_ERRORS:
        return None


def walk_forward(
    prices, risk_free_rate, objective, max_weight, shrinkage,
    refit_freq="Y", cost_bps=0.0, min_train_years=2.0,
    cash_fraction=0.0, rebalance_freq=None, min_weight=0.0,
    return_method=DEFAULT_RETURN_METHOD, min_observations=None, target=None,
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
            shrinkage, min_weight, return_method, min_observations, target,
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
