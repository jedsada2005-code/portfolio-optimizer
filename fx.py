"""Currency handling for portfolios that mix markets.

Combining a baht NAV with a dollar share price treats both as the same
unit of account, which hides exchange-rate risk entirely -- a Thai
investor holding a US fund is exposed to USDTHB whether the app models
it or not.
"""

import pandas as pd


BASE_CURRENCIES = ["USD", "THB"]

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
    if symbol.upper().startswith("MF:"):
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

    aligned = rates.reindex(rates.index.union(prices.index)).ffill().bfill()
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
