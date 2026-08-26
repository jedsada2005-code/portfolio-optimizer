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


def estimate_returns(weekly_prices, method, risk_free_rate=0.0):
    """Annualised expected returns from weekly prices.

    ``risk_free_rate`` reaches CAPM only, which is defined in terms of
    it: R_i = R_f + beta_i (E[R_m] - R_f). It was omitted from the call,
    so pypfopt used its 0.0 default however the sidebar was set, which
    understated low-beta holdings and overstated high-beta ones. The
    rate is annual, matching the annualised market return pypfopt
    computes from ``frequency``.
    """
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
    return pypfopt_returns.capm_return(
        weekly_prices, frequency=WEEKS_PER_YEAR, risk_free_rate=risk_free_rate
    )


def estimate_returns_with_counts(weekly_prices, method, risk_free_rate=0.0):
    """Expected returns plus how many observations back each one.

    The count always comes from the trailing-year windows, whatever the
    estimator, because that is what the history floor is expressed in.
    """
    _, counts = metrics.annual_return_estimates(weekly_prices)
    return estimate_returns(weekly_prices, method, risk_free_rate), counts


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


def bound_pairs(assets, min_weight, max_weight, floors=None, caps=None):
    """Per-asset ``(lower, upper)`` pairs, defaulting to the shared pair.

    pypfopt takes either one pair for everything or one per asset, and
    only the shared pair was ever passed. A floor meant to keep a core
    holding funded therefore also forced money into holdings the
    optimiser had good reason to decline -- two of them at exactly the
    floor in a real run, purely because the rule could not tell them
    apart.
    """
    floors = floors or {}
    caps = caps or {}
    return [
        (float(floors.get(a, min_weight)), float(caps.get(a, max_weight)))
        for a in assets
    ]


def validate_bounds(pairs, assets=None):
    """Reject weight bounds no allocation can satisfy.

    The solver's own message for an infeasible problem is "Please check
    your objectives/constraints", which does not say which bound is
    wrong or what would work.
    """
    names = list(assets) if assets is not None else list(range(len(pairs)))
    n = len(pairs)
    for name, (low, high) in zip(names, pairs):
        if low > high:
            raise ValueError(
                f"น้ำหนักขั้นต่ำของ {name} ({low:.0%}) มากกว่าน้ำหนักสูงสุด ({high:.0%})"
            )
    floor_total = sum(low for low, _ in pairs)
    cap_total = sum(high for _, high in pairs)
    if floor_total > 1.0 + 1e-9:
        raise ValueError(
            f"น้ำหนักขั้นต่ำรวมกัน {floor_total:.0%} เกิน 100% "
            f"— ลดขั้นต่ำลง หรือยกเว้นบางตัว (เฉลี่ยตั้งได้ไม่เกิน {1 / n:.1%} ต่อตัว)"
        )
    if cap_total < 1.0 - 1e-9:
        raise ValueError(
            f"น้ำหนักสูงสุดรวมกันได้แค่ {cap_total:.0%} ไม่ถึง 100% "
            f"— เพิ่มเพดาน หรือเพิ่มจำนวนสินทรัพย์ (เฉลี่ยต้องอย่างน้อย {1 / n:.1%} ต่อตัว)"
        )


def max_achievable_return(expected_returns, max_weight, min_weight=0.0,
                          floors=None, caps=None):
    """Highest expected return reachable under the weight bounds.

    Every asset is funded to its own floor first; only what remains can
    be steered towards the best performers, and no asset may take more
    than its own remaining headroom.
    """
    assets = list(expected_returns.index)
    pairs = bound_pairs(assets, min_weight, max_weight, floors, caps)
    validate_bounds(pairs, assets)

    low = dict(zip(assets, (p[0] for p in pairs)))
    high = dict(zip(assets, (p[1] for p in pairs)))
    remaining = 1.0 - sum(low.values())
    total = sum(low[a] * float(expected_returns[a]) for a in assets)
    for asset in expected_returns.sort_values(ascending=False).index:
        take = min(high[asset] - low[asset], remaining)
        total += take * float(expected_returns[asset])
        remaining -= take
        if remaining <= 1e-12:
            break
    return float(total)


def _frontier(expected_returns, cov, max_weight, min_weight=0.0,
              floors=None, caps=None):
    return EfficientFrontier(
        expected_returns, cov,
        weight_bounds=bound_pairs(
            list(expected_returns.index), min_weight, max_weight, floors, caps
        ),
    )


def achievable_range(expected_returns, cov, max_weight=1.0, min_weight=0.0,
                     floors=None, caps=None):
    """(lowest volatility, its return, highest return) under the bounds.

    Used to tell someone asking for an impossible target what they could
    have asked for instead.
    """
    ef = _frontier(expected_returns, cov, max_weight, min_weight, floors, caps)
    ef.min_volatility()
    min_return, min_volatility, _ = ef.portfolio_performance()
    return min_volatility, min_return, max_achievable_return(
        expected_returns, max_weight, min_weight, floors, caps
    )


def _downside_optimizer(objective, expected_returns, weekly, max_weight, min_weight,
                        floors=None, caps=None):
    returns = weekly.pct_change().dropna(how="all")
    bounds = bound_pairs(
        list(expected_returns.index), min_weight, max_weight, floors, caps
    )
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
                     max_weight=1.0, min_weight=0.0, weekly=None, target=None,
                     floors=None, caps=None):
    """Solve for one objective, raising ValueError with a readable message."""
    if objective in NEEDS_HISTORY and weekly is None:
        raise ValueError(f"{objective} ต้องใช้ประวัติผลตอบแทน ไม่ใช่แค่ค่าคาดหวัง")
    if objective == HRP_OBJECTIVE:
        return hrp_weights(weekly)

    validate_bounds(
        bound_pairs(list(expected_returns.index), min_weight, max_weight, floors, caps),
        list(expected_returns.index),
    )
    if objective in (MIN_CVAR, MIN_SEMIVARIANCE):
        return _downside_optimizer(
            objective, expected_returns, weekly, max_weight, min_weight, floors, caps
        )

    if objective in NEEDS_TARGET and target is None:
        raise ValueError(f"{objective} ต้องระบุค่าเป้าหมาย")

    ef = _frontier(expected_returns, cov, max_weight, min_weight, floors, caps)
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
                expected_returns, cov, max_weight, min_weight, floors, caps
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


def risk_contributions(weights, cov):
    """Each holding's share of the portfolio's variance.

    A weights table says where the money went; it does not say where the
    risk went, and the two come apart badly. On one eight-ETF fit a
    37.2% holding carried 8.3% of the variance while a 28.9% one carried
    49.4% -- so the position that would hurt in a drawdown was not the
    biggest line in the table.

    The cash sleeve is dropped first: it has no variance and belongs to
    no row of the covariance matrix.
    """
    risky = [a for a in cov.index if float(weights.get(a, 0.0)) != 0.0]
    if not risky:
        return pd.Series(dtype=float)

    w = np.array([float(weights[a]) for a in risky])
    total = w.sum()
    if total > 0:
        w = w / total
    matrix = cov.loc[risky, risky].values
    variance = float(w @ matrix @ w)
    if variance <= 0:
        return pd.Series(dtype=float)

    # Euler decomposition: each holding's weight times its marginal
    # contribution, over the total. These sum to one by construction.
    shares = w * (matrix @ w) / variance
    return pd.Series(shares, index=risky).reindex(cov.index).fillna(0.0)


def frontier_curve(expected_returns, cov, max_weight=1.0, n_points=60, min_weight=0.0,
                   floors=None, caps=None):
    """Trace the efficient frontier by solving for target returns.

    Replaces reading the line data back out of a throwaway matplotlib
    figure, which depended on pypfopt's internal plot ordering.
    """
    ef = _frontier(expected_returns, cov, max_weight, min_weight, floors, caps)
    try:
        ef.min_volatility()
    except SOLVER_ERRORS:
        return np.array([]), np.array([])
    low = ef.portfolio_performance()[0]
    high = max_achievable_return(expected_returns, max_weight, min_weight, floors, caps)
    if high - low <= 1e-9:
        vol = ef.portfolio_performance()[1]
        return np.array([vol]), np.array([low])

    vols, rets = [], []
    for target in np.linspace(low, high, n_points):
        try:
            point = _frontier(expected_returns, cov, max_weight, min_weight, floors, caps)
            point.efficient_return(target)
            ret, vol, _ = point.portfolio_performance()
        except SOLVER_ERRORS:
            continue
        vols.append(vol)
        rets.append(ret)
    return np.array(vols), np.array(rets)


def sample_weights(rng, n_assets, n_samples, max_weight=1.0, max_passes=12,
                   min_weight=0.0, pairs=None):
    """Random long-only weight vectors, optionally capped per asset.

    Plain rejection sampling empties out fast once the cap approaches
    equal weight, so capped draws are clipped and renormalised until
    they fit, which keeps the scatter cloud populated.
    """
    if pairs is None:
        pairs = [(min_weight, max_weight)] * n_assets
    try:
        validate_bounds(pairs)
    except ValueError:
        return np.empty((0, n_assets))

    low = np.array([p[0] for p in pairs], dtype=float)
    high = np.array([p[1] for p in pairs], dtype=float)

    # Fund each floor first, then spread what is left. Rejecting draws
    # that breach a floor would discard nearly all of them.
    w = rng.dirichlet([0.5] * n_assets, n_samples)
    w = low + (1.0 - low.sum()) * w
    if (high >= 1.0).all():
        return w

    for _ in range(max_passes):
        if not (w > high).any():
            break
        w = np.minimum(w, high)
        deficit = 1.0 - w.sum(axis=1, keepdims=True)
        headroom = high - w
        total_headroom = headroom.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(total_headroom > 0, headroom / total_headroom, 0.0)
        w = w + share * deficit

    w = np.clip(w, low, high)
    totals = w.sum(axis=1, keepdims=True)
    keep = np.isclose(totals[:, 0], 1.0, atol=1e-6)
    return w[keep]


def stats_over_one_window(returns_by_name, risk_free_rate):
    """Statistics for several return series over the window they share.

    simulate_portfolio starts at the common_start of whatever a row
    happens to hold, so a row that avoided a late-listing holding was
    handed years the others never traded -- fifteen against seven in the
    case that prompted this. Trimming every row to the overlap makes the
    table rank objectives rather than calendars.
    """
    live = {n: s for n, s in returns_by_name.items() if not s.empty}
    if not live:
        return {}
    start = max(s.index[0] for s in live.values())
    end = min(s.index[-1] for s in live.values())
    if start > end:
        return {}

    table = {}
    for name, series in live.items():
        window = series.loc[start:end].copy()
        if len(window) < 2:
            continue
        # Every row measures growth from the shared start, so the first
        # row is a base day on each rather than a real return on some.
        window.iloc[0] = 0.0
        table[name] = metrics.backtest_stats(window, risk_free_rate)
    return table


def compare_fixed_weights(weights_by_name, prices, risk_free_rate,
                          rebalance_freq=None, cost_bps=0.0, cash_fraction=0.0):
    """Backtest several weight sets through one identical simulation.

    Every row of a comparison has to come out of the same procedure on
    the same window, or the table ranks the procedures rather than the
    objectives it claims to be comparing.

    ``prices`` must already carry the cash column when ``cash_fraction``
    is non-zero, exactly as the single-portfolio backtest receives it.
    """
    returns = {}
    for name, weights in weights_by_name.items():
        blended = metrics.blend_with_cash(weights, cash_fraction)
        simulation = metrics.simulate_portfolio(
            prices, blended, rebalance_freq, cost_bps
        )
        if not simulation.returns.empty:
            returns[name] = simulation.returns
    return stats_over_one_window(returns, risk_free_rate)


def compare_walk_forward(objectives, prices, risk_free_rate, max_weight, shrinkage,
                         refit_freq="Y", cost_bps=0.0, targets=None, **walk_kwargs):
    """Walk each objective forward through the identical schedule.

    Reusing a fixed-weight backtest for the objectives the user did not
    select would put a re-fitted portfolio in one row and a static one in
    the next, which is the comparison this table exists to avoid.
    """
    targets = targets or {}
    returns = {}
    for name in objectives:
        try:
            result = walk_forward(
                prices, risk_free_rate, name, max_weight, shrinkage,
                refit_freq, cost_bps, target=targets.get(name), **walk_kwargs
            )
        except Exception:  # noqa: BLE001 - one row must not cost the table
            continue
        if not result.returns.empty:
            returns[name] = result.returns
    return stats_over_one_window(returns, risk_free_rate)


WalkForwardResult = namedtuple("WalkForwardResult", "returns weight_history turnover")


def fit_weights(prices, risk_free_rate, objective, max_weight, shrinkage,
                min_weight=0.0, return_method=DEFAULT_RETURN_METHOD,
                min_observations=None, target=None, floors=None, caps=None):
    """Expected returns and covariance from a price window, then solve."""
    weekly = prices.resample("W-FRI").last()
    expected, observations = estimate_returns_with_counts(
        weekly, return_method, risk_free_rate
    )
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
            weekly=weekly[usable], target=target, floors=floors, caps=caps,
        )
    except SOLVER_ERRORS:
        return None


def walk_forward(
    prices, risk_free_rate, objective, max_weight, shrinkage,
    refit_freq="Y", cost_bps=0.0, min_train_years=2.0,
    cash_fraction=0.0, rebalance_freq=None, min_weight=0.0,
    return_method=DEFAULT_RETURN_METHOD, min_observations=None, target=None,
    floors=None, caps=None,
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
            floors, caps,
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
