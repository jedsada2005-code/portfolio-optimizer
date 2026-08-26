import io

import pandas as pd

import custom_data


def test_parse_wide_csv_with_multiple_assets():
    csv = io.StringIO(
        "Date,AAPL,SPY\n"
        "2024-01-01,100,400\n"
        "2024-01-02,101,402\n"
    )

    prices = custom_data.parse_price_csv(csv, "ignored.csv")

    assert list(prices.columns) == ["AAPL", "SPY"]
    assert prices.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 101
    assert prices.loc[pd.Timestamp("2024-01-02"), "SPY"] == 402


def test_parse_long_csv_with_symbol_and_close_columns():
    csv = io.StringIO(
        "Date,Symbol,Close\n"
        "2024-01-01,AAPL,100\n"
        "2024-01-01,SPY,400\n"
        "2024-01-02,AAPL,101\n"
    )

    prices = custom_data.parse_price_csv(csv, "ignored.csv")

    assert list(prices.columns) == ["AAPL", "SPY"]
    assert prices.loc[pd.Timestamp("2024-01-01"), "SPY"] == 400
    assert prices.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 101


def test_parse_single_asset_csv_uses_file_name_when_column_is_close():
    csv = io.StringIO(
        "Date,Close\n"
        "2024-01-01,10.5\n"
        "2024-01-02,10.8\n"
    )

    prices = custom_data.parse_price_csv(csv, "my custom asset.csv")

    assert list(prices.columns) == ["MY_CUSTOM_ASSET"]
    assert prices.loc[pd.Timestamp("2024-01-02"), "MY_CUSTOM_ASSET"] == 10.8


def test_parse_single_sheet_xlsx_with_wide_prices():
    file_obj = io.BytesIO()
    with pd.ExcelWriter(file_obj, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "Date": ["2024-01-01", "2024-01-02"],
                "AAPL": [100, 101],
                "SPY": [400, 402],
            }
        ).to_excel(writer, index=False)
    file_obj.seek(0)

    prices = custom_data.parse_price_file(file_obj, "prices.xlsx")

    assert list(prices.columns) == ["AAPL", "SPY"]
    assert prices.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 101


def test_parse_multi_sheet_xlsx_uses_sheet_names_for_single_asset_sheets():
    file_obj = io.BytesIO()
    with pd.ExcelWriter(file_obj, engine="openpyxl") as writer:
        pd.DataFrame(
            {"Date": ["2024-01-01", "2024-01-02"], "Close": [10.5, 10.8]}
        ).to_excel(writer, sheet_name="Private Fund", index=False)
        pd.DataFrame(
            {"Date": ["2024-01-01", "2024-01-02"], "NAV": [20.0, 20.2]}
        ).to_excel(writer, sheet_name="Bond Sleeve", index=False)
    file_obj.seek(0)

    prices = custom_data.parse_price_file(file_obj, "portfolio.xlsx")

    assert list(prices.columns) == ["PRIVATE_FUND", "BOND_SLEEVE"]
    assert prices.loc[pd.Timestamp("2024-01-02"), "BOND_SLEEVE"] == 20.2


def test_parse_csv_rejects_missing_date_column():
    csv = io.StringIO(
        "Asset,Close\n"
        "AAPL,100\n"
        "SPY,400\n"
    )

    try:
        custom_data.parse_price_csv(csv, "bad.csv")
        assert False, "expected CSVPriceDataError"
    except custom_data.CSVPriceDataError:
        pass


def test_merge_uploaded_prices_bridges_gaps_inside_a_column():
    market = pd.DataFrame(
        {"AAPL": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-03"]),
    )
    uploaded = pd.DataFrame(
        {"PRIVATE_FUND": [50.0, 51.0, 52.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )

    merged = custom_data.merge_uploaded_prices(market, uploaded)

    assert list(merged.columns) == ["AAPL", "PRIVATE_FUND"]
    # AAPL did not trade on the 2nd; carrying the 1st forward is right.
    assert merged.loc[pd.Timestamp("2024-01-02"), "AAPL"] == 100.0


def test_merge_uploaded_prices_does_not_stretch_a_file_that_ends_early():
    """A file ending before the market data must not be filled out to
    meet it: that turns the last uploaded price into a flat line the
    optimiser reads as a zero-volatility holding."""
    market = pd.DataFrame(
        {"AAPL": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-03"]),
    )
    uploaded = pd.DataFrame(
        {"PRIVATE_FUND": [50.0, 51.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    merged = custom_data.merge_uploaded_prices(market, uploaded)

    assert pd.isna(merged.loc[pd.Timestamp("2024-01-03"), "PRIVATE_FUND"])


def test_merge_uploaded_prices_rejects_duplicate_market_symbol():
    market = pd.DataFrame(
        {"AAPL": [100.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )
    uploaded = pd.DataFrame(
        {"AAPL": [99.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )

    try:
        custom_data.merge_uploaded_prices(market, uploaded)
        assert False, "expected CSVPriceDataError"
    except custom_data.CSVPriceDataError as exc:
        assert "AAPL" in str(exc)
