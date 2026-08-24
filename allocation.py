"""Turning target weights into an order you can actually place.

The optimiser answers "hold 45.9% SPY". Placing that order needs a
number of units and the change left over, which is where a percentage
stops being useful.
"""

from collections import namedtuple

import pandas as pd

import fx
import metrics


AllocationPlan = namedtuple("AllocationPlan", "rows leftover unaffordable unpriced")


def _is_fractional(symbol, fractional):
    """Thai funds are bought by amount and settle in fractional units."""
    return symbol in fractional or symbol.upper().startswith(fx.MF_PREFIX)


def allocate_units(weights, latest_prices, total_value, fractional=()):
    """Convert weights into units to buy against a real budget.

    Whole-unit holdings are floored and then topped up greedily, always
    buying the holding that sits furthest below its target, so the
    change left over stays small instead of accumulating one part-unit
    per asset.
    """
    fractional = set(fractional)
    unpriced, unaffordable = [], []
    plan, remaining = {}, float(total_value)

    held = {a: float(w) for a, w in weights.items() if float(w) > 0}
    for asset in list(held):
        if asset == metrics.CASH_SYMBOL:
            continue
        price = latest_prices.get(asset)
        if price is None or not price > 0 or pd.isna(price):
            unpriced.append(asset)
            held.pop(asset)

    for asset, weight in held.items():
        target = total_value * weight
        if asset == metrics.CASH_SYMBOL:
            plan[asset] = (None, target)
            remaining -= target
        elif _is_fractional(asset, fractional):
            price = float(latest_prices[asset])
            plan[asset] = (target / price, target)
            remaining -= target
        else:
            price = float(latest_prices[asset])
            units = int(target // price)
            plan[asset] = (units, units * price)
            remaining -= units * price
            if units == 0:
                unaffordable.append(asset)

    # Spend what is left one unit at a time, always on whichever whole-unit
    # holding is furthest below its target share.
    buyable = [
        a for a in held
        if a != metrics.CASH_SYMBOL and not _is_fractional(a, fractional)
    ]
    while buyable and total_value > 0:
        affordable = [a for a in buyable if float(latest_prices[a]) <= remaining + 1e-9]
        if not affordable:
            break
        choice = min(
            affordable,
            key=lambda a: plan[a][1] / total_value - held[a],
        )
        price = float(latest_prices[choice])
        units, value = plan[choice]
        plan[choice] = (units + 1, value + price)
        remaining -= price
        if choice in unaffordable:
            unaffordable.remove(choice)

    rows = [
        {
            "asset": asset,
            "price": None if asset == metrics.CASH_SYMBOL else float(latest_prices[asset]),
            "units": plan[asset][0],
            "value": plan[asset][1],
            "target_weight": held[asset],
            "actual_weight": plan[asset][1] / total_value if total_value else 0.0,
        }
        for asset in sorted(plan)
    ]
    return AllocationPlan(rows, max(remaining, 0.0), sorted(unaffordable), sorted(unpriced))
