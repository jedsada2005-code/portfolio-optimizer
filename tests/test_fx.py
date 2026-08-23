import numpy as np
import pandas as pd
import pytest

import fx


class TestCurrencyForSymbol:
    def test_thai_mutual_funds_are_baht(self):
        assert fx.currency_for_symbol("MF:K-GOLD-A(A)") == "THB"

    def test_thai_listed_equities_are_baht(self):
        assert fx.currency_for_symbol("PTT.BK") == "THB"

    def test_plain_us_tickers_are_dollars(self):
        for symbol in ("SPY", "AMZN", "BRK-B"):
            assert fx.currency_for_symbol(symbol) == "USD"

    def test_known_exchange_suffixes(self):
        assert fx.currency_for_symbol("VOD.L") == "GBP"
        assert fx.currency_for_symbol("7203.T") == "JPY"
        assert fx.currency_for_symbol("0700.HK") == "HKD"
        assert fx.currency_for_symbol("SAP.DE") == "EUR"

    def test_unknown_suffix_falls_back_to_the_default(self):
        assert fx.currency_for_symbol("FOO.XYZ") == "USD"
        assert fx.currency_for_symbol("FOO.XYZ", default="THB") == "THB"

    def test_uploaded_columns_use_the_declared_currency(self):
        assert fx.currency_for_symbol("MYFUND", default="EUR") == "EUR"


class TestConvertPrices:
    def _rates(self, idx):
        # 35 THB and 150 JPY to the dollar, held flat.
        return pd.DataFrame({"THB": 35.0, "JPY": 150.0, "USD": 1.0}, index=idx)

    def test_converting_to_the_native_currency_changes_nothing(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({"SPY": [100.0] * 5}, index=idx)
        out = fx.convert_prices(prices, {"SPY": "USD"}, "USD", self._rates(idx))
        pd.testing.assert_frame_equal(out, prices)

    def test_baht_prices_become_dollars(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({"MF:X": [35.0] * 5}, index=idx)
        out = fx.convert_prices(prices, {"MF:X": "THB"}, "USD", self._rates(idx))
        assert out["MF:X"].iloc[0] == pytest.approx(1.0)

    def test_dollar_prices_become_baht(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({"SPY": [1.0] * 5}, index=idx)
        out = fx.convert_prices(prices, {"SPY": "USD"}, "THB", self._rates(idx))
        assert out["SPY"].iloc[0] == pytest.approx(35.0)

    def test_cross_rate_between_two_foreign_currencies(self):
        idx = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame({"7203.T": [150.0] * 5}, index=idx)
        out = fx.convert_prices(prices, {"7203.T": "JPY"}, "THB", self._rates(idx))
        assert out["7203.T"].iloc[0] == pytest.approx(35.0)

    def test_currency_moves_show_up_in_the_converted_return(self):
        # A flat baht NAV still loses value in dollars if the baht weakens.
        idx = pd.bdate_range("2020-01-01", periods=3)
        rates = pd.DataFrame({"THB": [35.0, 36.75, 36.75], "USD": 1.0}, index=idx)
        prices = pd.DataFrame({"MF:X": [35.0, 35.0, 35.0]}, index=idx)
        out = fx.convert_prices(prices, {"MF:X": "THB"}, "USD", rates)
        # The baht weakening by 5% costs 1 - 1/1.05 = 4.76% in dollars.
        assert out["MF:X"].pct_change().iloc[1] == pytest.approx(-1 / 21, abs=1e-9)

    def test_missing_rate_days_are_carried_forward(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        rates = pd.DataFrame({"THB": [35.0, np.nan, np.nan, 35.0], "USD": 1.0}, index=idx)
        prices = pd.DataFrame({"MF:X": [35.0] * 4}, index=idx)
        out = fx.convert_prices(prices, {"MF:X": "THB"}, "USD", rates)
        assert out["MF:X"].notna().all()

    def test_an_unavailable_currency_is_reported(self):
        idx = pd.bdate_range("2020-01-01", periods=3)
        prices = pd.DataFrame({"X.L": [1.0] * 3}, index=idx)
        with pytest.raises(fx.FXError, match="GBP"):
            fx.convert_prices(prices, {"X.L": "GBP"}, "USD", self._rates(idx))

    def test_required_currencies_lists_what_must_be_fetched(self):
        assert fx.required_currencies({"A": "THB", "B": "USD", "C": "THB"}, "USD") == ["THB"]
        assert fx.required_currencies({"A": "USD", "B": "USD"}, "USD") == []
        # USD is the reference leg, so only the base rate is needed.
        assert fx.required_currencies({"A": "USD"}, "THB") == ["THB"]
