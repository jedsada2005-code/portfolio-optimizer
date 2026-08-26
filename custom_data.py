from pathlib import Path
import warnings

import pandas as pd

import metrics


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


class PriceDataError(ValueError):
    """Raised when an uploaded price file cannot be converted to asset prices."""


CSVPriceDataError = PriceDataError


def _normalized_columns(df):
    return {str(col).strip().lower(): col for col in df.columns}


def _fallback_asset_name(label):
    stem = Path(label or "UPLOADED_ASSET").stem.strip()
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in stem)
    return cleaned.upper() or "UPLOADED_ASSET"


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
    raise PriceDataError("Uploaded file must include a date column.")


def _to_price_frame(df, date_col):
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    if out.empty:
        raise PriceDataError("Uploaded file has no valid dates.")
    return out


def _numeric_series(values):
    return pd.to_numeric(values, errors="coerce")


def _parse_price_frame(raw, fallback_asset_label):
    if raw.empty:
        raise PriceDataError("Uploaded file is empty.")

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
            raise PriceDataError("Uploaded file has no valid price values.")
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
            raise PriceDataError("Uploaded file has no numeric price columns.")

        if len(numeric.columns) == 1 and str(numeric.columns[0]).strip().lower() in PRICE_COLUMNS:
            numeric = numeric.rename(columns={numeric.columns[0]: _fallback_asset_name(fallback_asset_label)})
        prices = numeric

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    prices = prices.dropna(how="all")
    prices.columns = [str(col).strip().upper() for col in prices.columns]
    prices = prices.loc[:, prices.columns != ""]

    if prices.empty:
        raise PriceDataError("Uploaded file produced no usable price data.")
    if not prices.columns.is_unique:
        duplicates = sorted(set(prices.columns[prices.columns.duplicated()]))
        raise PriceDataError(f"Duplicate asset columns in uploaded file: {', '.join(duplicates)}")

    return prices


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
        raise PriceDataError(f"Could not read CSV: {exc}") from exc

    return _parse_price_frame(raw, filename)


def parse_price_xlsx(file_obj, filename=None):
    """Parse an uploaded XLSX workbook into a daily price DataFrame.

    Every worksheet is read. Wide and long sheets keep their own asset
    names; single-asset sheets such as Date,Close use the sheet name when
    there are multiple sheets, otherwise the workbook file name.
    """
    try:
        sheets = pd.read_excel(file_obj, sheet_name=None)
    except Exception as exc:
        raise PriceDataError(f"Could not read XLSX: {exc}") from exc

    if not sheets:
        raise PriceDataError("XLSX workbook has no sheets.")

    frames = []
    errors = []
    use_sheet_name = len(sheets) > 1
    for sheet_name, raw in sheets.items():
        fallback = sheet_name if use_sheet_name else filename
        try:
            frames.append(_parse_price_frame(raw, fallback))
        except PriceDataError as exc:
            errors.append(f"{sheet_name}: {exc}")

    if errors and not frames:
        raise PriceDataError("Could not read any XLSX sheet:\n" + "\n".join(errors))

    prices = pd.concat(frames, axis=1)
    if not prices.columns.is_unique:
        duplicates = sorted(set(prices.columns[prices.columns.duplicated()]))
        raise PriceDataError(f"Duplicate asset columns in XLSX: {', '.join(duplicates)}")
    return prices.sort_index()


def parse_price_file(file_obj, filename=None):
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".csv":
        return parse_price_csv(file_obj, filename)
    if suffix == ".xlsx":
        return parse_price_xlsx(file_obj, filename)
    raise PriceDataError("Unsupported file type. Please upload .csv or .xlsx.")


def merge_uploaded_prices(data_close, uploaded_prices):
    """Outer-join uploaded prices with market data and forward-fill gaps only."""
    if uploaded_prices.empty:
        return data_close

    overlap = set(data_close.columns).intersection(uploaded_prices.columns)
    if overlap:
        raise CSVPriceDataError(
            "Uploaded file duplicates existing symbols: " + ", ".join(sorted(overlap))
        )

    merged = (
        data_close.join(uploaded_prices, how="outer")
        if not data_close.empty
        else uploaded_prices.copy()
    )
    # Only inside each column's own life: an uploaded file that stops
    # earlier than the market data must not be stretched flat to meet it.
    return metrics.ffill_within_life(merged.sort_index())
