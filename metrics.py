"""Performance statistics shared by the backtesting and NAV views.

These live outside app.py so the formulas can be unit tested directly
instead of only through the Streamlit UI.
"""

from collections import namedtuple

import numpy as np
import pandas as pd


DAYS_PER_YEAR = 365.25
FALLBACK_PERIODS_PER_YEAR = 252.0

_MONTH_ABBR = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def years_elapsed(index):
    """Calendar years covered by a DatetimeIndex.

    Counting rows and dividing by 252 overstates the elapsed time
    whenever the index is a union of several trading calendars (a Thai
    fund merged with US equities easily reaches ~300 rows per year),
    which in turn understates CAGR.
    """
    if len(index) < 2:
        return 1.0 / DAYS_PER_YEAR
    span_days = (index[-1] - index[0]).days
    if span_days <= 0:
        return 1.0 / DAYS_PER_YEAR
    return span_days / DAYS_PER_YEAR


def periods_per_year(index):
    """Observations per calendar year implied by the index itself.

    Used to annualise volatility so the scaling matches the data
    actually being measured rather than an assumed 252-day US calendar.
    """
    if len(index) < 2:
        return FALLBACK_PERIODS_PER_YEAR
    span_days = (index[-1] - index[0]).days
    if span_days <= 0:
        return FALLBACK_PERIODS_PER_YEAR
    return len(index) / (span_days / DAYS_PER_YEAR)


def downside_deviation(returns, periods_per_year_value, target=0.0):
    """Annualised downside deviation.

    Squares only the shortfalls below ``target`` but averages over every
    period. Taking ``.std()`` of the negative returns alone instead both
    de-means them and divides by the count of losing periods, which
    understates downside risk and inflates Sortino.
    """
    r = pd.Series(returns).dropna()
    if r.empty:
        return 0.0
    shortfall = np.minimum(r - target, 0.0)
    return float(np.sqrt((shortfall ** 2).mean()) * np.sqrt(periods_per_year_value))


def sortino_ratio(annual_return, downside_dev, risk_free_rate):
    if downside_dev <= 0:
        return 0.0
    return (annual_return - risk_free_rate) / downside_dev


def cagr(total_return, years):
    """Compound annual growth rate from a cumulative return."""
    if years <= 0:
        return total_return
    growth = 1.0 + total_return
    if growth <= 0:
        return -1.0
    return growth ** (1.0 / years) - 1.0


def month_labels(month_numbers):
    """Map real month numbers to abbreviations.

    Slicing a fixed Jan..Dec list by column count mislabels every month
    when the backtest does not happen to start in January.
    """
    return [_MONTH_ABBR[int(m)] for m in month_numbers]


MIN_ANNUAL_OBSERVATIONS = 52
ANNUAL_LOOKBACK_WEEKS = 52


PortfolioReturns = namedtuple("PortfolioReturns", "portfolio assets held start")


def first_valid_dates(prices):
    """First date each column actually has a price."""
    return prices.apply(lambda col: col.first_valid_index())


def common_start(prices):
    """First date on which every column has a price, or None.

    A portfolio cannot be backtested before all of its holdings exist.
    """
    if prices.empty or prices.shape[1] == 0:
        return None
    complete = prices.notna().all(axis=1)
    if not complete.any():
        return None
    return complete.idxmax()


def portfolio_daily_returns(prices, weights):
    """Daily returns of a constant-weight portfolio.

    Starts only once every held asset has a real price. Filling missing
    pre-inception prices with a zero return instead keeps the asset's
    full weight in the portfolio while contributing nothing, which
    silently dilutes the result towards cash.
    """
    empty = pd.Series(dtype=float)
    held = [c for c in prices.columns if float(weights.get(c, 0.0) or 0.0) > 0]
    if not held:
        return PortfolioReturns(empty, pd.DataFrame(), [], None)

    sub = prices[held]
    start = common_start(sub)
    if start is None:
        return PortfolioReturns(empty, pd.DataFrame(), held, None)

    sub = sub.loc[start:].ffill()
    w = np.array([float(weights[c]) for c in held], dtype=float)
    total = w.sum()
    if total > 0:
        w = w / total

    asset_returns = sub.pct_change().fillna(0.0)
    portfolio = asset_returns.dot(pd.Series(w, index=held))
    return PortfolioReturns(portfolio, asset_returns, held, start)


def annual_return_estimates(weekly, lookback=ANNUAL_LOOKBACK_WEEKS):
    """Mean trailing annual return per asset, plus how many observations
    each estimate rests on.

    An asset with barely over a year of history yields only a handful of
    overlapping windows, so its mean is nearly meaningless -- yet it is
    not NaN, so a NaN-only guard lets it reach the optimiser, which then
    chases the noise.
    """
    changes = weekly.pct_change(lookback)
    return changes.mean(), changes.notna().sum()


def unreliable_assets(observation_counts, minimum=MIN_ANNUAL_OBSERVATIONS):
    """Assets whose expected return rests on too few observations."""
    return {
        asset: int(count)
        for asset, count in observation_counts.items()
        if int(count) < minimum
    }
