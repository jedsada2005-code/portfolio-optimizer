"""Currency handling for portfolios that mix markets.

Combining a baht NAV with a dollar share price treats both as the same
unit of account, which hides exchange-rate risk entirely -- a Thai
investor holding a US fund is exposed to USDTHB whether the app models
it or not.
"""

import pandas as pd


BASE_CURRENCIES = ["USD", "THB"]

MF_PREFIX = "MF:"

# Yahoo quotes "<CUR>=X" as units of that currency per one US dollar.
FX_SYMBOL_TEMPLATE = "{currency}=X"

SUFFIX_CURRENCIES = {
    "BK": "THB", "L": "GBP", "T": "JPY", "HK": "HKD", "SS": "CNY",
    "SZ": "CNY", "DE": "EUR", "PA": "EUR", "AS": "EUR", "MI": "EUR",
    "MC": "EUR", "BR": "EUR", "AX": "AUD", "TO": "CAD", "SI": "SGD",
    "KS": "KRW", "KQ": "KRW", "SW": "CHF", "ST": "SEK", "OL": "NOK",
    "CO": "DKK", "NZ": "NZD", "TW": "TWD", "NS": "INR", "BO": "INR",
}


class FXError(Exception):
    """Raised when a portfolio needs a rate that could not be fetched."""


def currency_for_symbol(symbol, default="USD"):
    """Native currency of a symbol, inferred from its exchange suffix."""
    if symbol.upper().startswith(MF_PREFIX):
        return "THB"
    _, _, suffix = symbol.rpartition(".")
    if suffix and suffix != symbol:
        return SUFFIX_CURRENCIES.get(suffix.upper(), default)
    return default


def required_currencies(currencies, base):
    """Currencies whose USD rate is needed to reach ``base``."""
    needed = set(currencies.values()) | {base}
    needed.discard("USD")
    return sorted(needed)


def convert_prices(prices, currencies, base, rates):
    """Restate every column in ``base``.

    ``rates`` holds units of each currency per one US dollar, so a price
    is divided by its own rate to reach dollars and multiplied by the
    base rate to leave them.
    """
    if prices.empty:
        return prices

    # ffill only. Back-filling priced the earliest part of the window at
    # a rate that did not exist yet, so the currency move over that
    # stretch came out as exactly zero -- silently, and for 40% of the
    # window in the case that prompted this. A date with no rate has no
    # price in the base currency, and says so by staying NaN.
    aligned = rates.reindex(rates.index.union(prices.index)).ffill()
    aligned = aligned.reindex(prices.index)

    def rate_for(currency):
        if currency == "USD":
            return pd.Series(1.0, index=prices.index)
        if currency not in aligned.columns or aligned[currency].isna().all():
            raise FXError(f"ไม่พบอัตราแลกเปลี่ยนสำหรับสกุลเงิน {currency}")
        return aligned[currency]

    base_rate = rate_for(base)
    out = {}
    for column in prices.columns:
        native = currencies.get(column, "USD")
        if native == base:
            out[column] = prices[column]
        else:
            out[column] = prices[column] / rate_for(native) * base_rate
    return pd.DataFrame(out, index=prices.index)[list(prices.columns)]


def late_rates(prices, currencies, base, rates):
    """Currencies whose rate history begins after the prices do.

    Those dates cannot be converted at all, so the backtest has to start
    later -- worth saying out loud rather than letting it look like the
    holdings themselves were unavailable.
    """
    if prices.empty or rates is None or rates.empty:
        return {}
    late = {}
    for currency in required_currencies(currencies, base):
        if currency not in rates.columns:
            continue
        series = rates[currency].dropna()
        if series.empty:
            continue
        if series.index[0] > prices.index[0]:
            late[currency] = series.index[0]
    return late


def resolve_currencies(symbols, lookup=None, defaults=None):
    """Native currency per symbol, preferring a live lookup.

    The exchange suffix only names the venue, not the currency the
    security trades in: IBTA.L is a dollar-denominated Treasury ETF
    listed in London, and reading ".L" as sterling converts it twice.
    Thai funds and uploaded columns carry their currency already, so
    they never reach the lookup.
    """
    defaults = defaults or {}
    resolved = {}
    for symbol in symbols:
        if symbol.upper().startswith(MF_PREFIX):
            resolved[symbol] = "THB"
            continue
        if symbol in defaults:
            resolved[symbol] = defaults[symbol]
            continue
        found = None
        if lookup is not None:
            try:
                found = lookup(symbol)
            except Exception:
                found = None
        resolved[symbol] = found.upper() if found else currency_for_symbol(symbol)
    return resolved


def corrected_guesses(resolved):
    """Symbols whose resolved currency contradicts the suffix guess."""
    corrections = {}
    for symbol, currency in resolved.items():
        guess = currency_for_symbol(symbol)
        if guess != currency:
            corrections[symbol] = (guess, currency)
    return corrections
