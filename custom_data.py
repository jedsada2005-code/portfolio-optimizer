from pathlib import Path
import warnings

import pandas as pd


DATE_COLUMNS = {"date", "datetime", "timestamp", "time"}
SYMBOL_COLUMNS = {"symbol", "ticker", "asset", "name"}
PRICE_COLUMNS = {
    "adj close",
    "adj_close",
    "close",
    "price",
    "nav",
    "value",
}


class CSVPriceDataError(ValueError):
    """Raised when an uploaded CSV cannot be converted to asset prices."""


def _normalized_columns(df):
    return {str(col).strip().lower(): col for col in df.columns}


def _fallback_asset_name(filename):
    stem = Path(filename or "CSV_ASSET").stem.strip()
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
    return cleaned.upper() or "CSV_ASSET"


def _find_date_column(df):
    columns = _normalized_columns(df)
    for label in DATE_COLUMNS:
        if label in columns:
            return columns[label]

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return col
    raise CSVPriceDataError("CSV must include a date column.")


def _to_price_frame(df, date_col):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    if out.empty:
        raise CSVPriceDataError("CSV has no valid dates.")
    return out


def _numeric_series(values):
    return pd.to_numeric(values, errors="coerce")


def parse_price_csv(file_obj, filename=None):
    """Parse one uploaded CSV into a daily price DataFrame.

    Supported shapes:
      - wide: Date, AAPL, SPY, ...
      - long: Date, Symbol, Close
      - single asset: Date, Close (asset name comes from the file name)
    """
    try:
        raw = pd.read_csv(file_obj)
    except Exception as exc:
        raise CSVPriceDataError(f"Could not read CSV: {exc}") from exc

    if raw.empty:
        raise CSVPriceDataError("CSV is empty.")

    raw.columns = [str(col).strip() for col in raw.columns]
    date_col = _find_date_column(raw)
    df = _to_price_frame(raw, date_col)
    columns = _normalized_columns(df)

    symbol_col = next((columns[c] for c in SYMBOL_COLUMNS if c in columns), None)
    price_col = next((columns[c] for c in PRICE_COLUMNS if c in columns), None)

    if symbol_col is not None and price_col is not None:
        long_df = df[[date_col, symbol_col, price_col]].copy()
        long_df[symbol_col] = long_df[symbol_col].astype(str).str.strip()
        long_df = long_df[long_df[symbol_col] != ""]
        long_df[price_col] = _numeric_series(long_df[price_col])
        long_df = long_df.dropna(subset=[price_col])
        if long_df.empty:
            raise CSVPriceDataError("CSV has no valid price values.")
        prices = long_df.pivot_table(
            index=date_col,
            columns=symbol_col,
            values=price_col,
            aggfunc="last",
        )
    else:
        value_cols = [col for col in df.columns if col != date_col]
        numeric = df[value_cols].apply(_numeric_series)
        numeric.index = df[date_col]
        numeric = numeric.dropna(axis=1, how="all")
        if numeric.empty:
            raise CSVPriceDataError("CSV has no numeric price columns.")

        if len(numeric.columns) == 1 and str(numeric.columns[0]).strip().lower() in PRICE_COLUMNS:
            numeric = numeric.rename(columns={numeric.columns[0]: _fallback_asset_name(filename)})
        prices = numeric

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="all")
    prices.columns = [str(col).strip().upper() for col in prices.columns]
    prices = prices.loc[:, prices.columns != ""]

    if prices.empty:
        raise CSVPriceDataError("CSV produced no usable price data.")
    if not prices.columns.is_unique:
        duplicates = sorted(set(prices.columns[prices.columns.duplicated()]))
        raise CSVPriceDataError(f"Duplicate asset columns in CSV: {', '.join(duplicates)}")

    return prices


def merge_uploaded_prices(data_close, uploaded_prices):
    """Outer-join uploaded prices with market data and forward-fill gaps only."""
    if uploaded_prices.empty:
        return data_close

    overlap = set(data_close.columns).intersection(uploaded_prices.columns)
    if overlap:
        raise CSVPriceDataError(
            "Uploaded CSV duplicates existing symbols: " + ", ".join(sorted(overlap))
        )

    merged = (
        data_close.join(uploaded_prices, how="outer")
        if not data_close.empty
        else uploaded_prices.copy()
    )
    return merged.sort_index().ffill()
