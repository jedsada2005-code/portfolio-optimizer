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

    def test_a_blend_converts_only_the_columns_that_need_it(self):
        # A benchmark named as "SPY 60, 2801.HK 40" reaches this function
        # as one frame mixing base-currency and foreign columns, where a
        # single ticker only ever presented one or the other.
        idx = pd.bdate_range("2020-01-01", periods=5)
        prices = pd.DataFrame(
            {"SPY": [100.0] * 5, "MF:X": [35.0] * 5}, index=idx
        )
        out = fx.convert_prices(
            prices, {"SPY": "USD", "MF:X": "THB"}, "USD", self._rates(idx)
        )
        pd.testing.assert_series_equal(out["SPY"], prices["SPY"])
        assert out["MF:X"].iloc[0] == pytest.approx(1.0)

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


class TestResolveCurrencies:
    """Guessing a currency from the exchange suffix is wrong for any
    security that trades in a currency other than its venue's own.

    IBTA.L, a dollar-denominated Treasury ETF listed in London, was read
    as sterling and converted twice.
    """

    def test_a_live_lookup_overrides_the_suffix_guess(self):
        resolved = fx.resolve_currencies(["IBTA.L"], lookup=lambda s: "USD")
        assert resolved["IBTA.L"] == "USD"

    def test_the_suffix_guess_stands_in_when_the_lookup_is_silent(self):
        resolved = fx.resolve_currencies(["VOD.L", "SPY"], lookup=lambda s: None)
        assert resolved == {"VOD.L": "GBP", "SPY": "USD"}

    def test_a_failing_lookup_does_not_break_resolution(self):
        def explode(symbol):
            raise RuntimeError("network down")
        assert fx.resolve_currencies(["VOD.L"], lookup=explode) == {"VOD.L": "GBP"}

    def test_thai_funds_never_hit_the_lookup(self):
        seen = []
        def spy(symbol):
            seen.append(symbol)
            return "USD"
        resolved = fx.resolve_currencies(["MF:K-GOLD-A(A)"], lookup=spy)
        assert resolved["MF:K-GOLD-A(A)"] == "THB"
        assert seen == []

    def test_declared_defaults_win_for_uploaded_columns(self):
        resolved = fx.resolve_currencies(
            ["MYFUND"], lookup=lambda s: None, defaults={"MYFUND": "EUR"}
        )
        assert resolved["MYFUND"] == "EUR"

    def test_a_declared_default_also_skips_the_lookup(self):
        seen = []
        fx.resolve_currencies(
            ["MYFUND"], lookup=lambda s: seen.append(s) or "USD", defaults={"MYFUND": "EUR"}
        )
        assert seen == []

    def test_lowercase_lookups_are_normalised(self):
        assert fx.resolve_currencies(["X.L"], lookup=lambda s: "usd")["X.L"] == "USD"

    def test_no_lookup_falls_back_to_guessing(self):
        assert fx.resolve_currencies(["VOD.L"]) == {"VOD.L": "GBP"}

    def test_mismatches_are_reported_for_display(self):
        corrected = fx.corrected_guesses(
            {"IBTA.L": "USD", "VOD.L": "GBP", "SPY": "USD"}
        )
        assert corrected == {"IBTA.L": ("GBP", "USD")}


class TestRatesAreNotInvented:
    """convert_prices back-filled the rate series, so the earliest part
    of a window was priced at a rate that did not exist yet and the
    currency move over that stretch came out as exactly zero -- 40% of
    the window in the case that prompted this, with no warning."""

    @staticmethod
    def _setup():
        index = pd.bdate_range("2020-01-01", "2024-12-31")
        rate = pd.Series(np.nan, index=index, dtype=float)
        live = index[index >= "2022-01-01"]
        rate.loc[live] = np.linspace(30.0, 36.0, len(live))
        prices = pd.DataFrame({"THAI": pd.Series(100.0, index=index)})
        return prices, pd.DataFrame({"THB": rate})

    def test_dates_before_the_first_rate_are_left_unpriced(self):
        prices, rates = self._setup()
        out = fx.convert_prices(prices, {"THAI": "THB"}, "USD", rates)
        assert out.loc[:"2021-12-31", "THAI"].isna().all()

    def test_dates_the_rate_covers_are_still_converted(self):
        prices, rates = self._setup()
        out = fx.convert_prices(prices, {"THAI": "THB"}, "USD", rates)
        assert out.loc["2022-01-03":, "THAI"].notna().all()
        assert out["THAI"].dropna().iloc[0] == pytest.approx(100.0 / 30.0)

    def test_an_interior_gap_is_still_carried_forward(self):
        idx = pd.bdate_range("2020-01-01", periods=4)
        rates = pd.DataFrame({"THB": [35.0, np.nan, np.nan, 35.0]}, index=idx)
        prices = pd.DataFrame({"MF:X": [35.0] * 4}, index=idx)
        out = fx.convert_prices(prices, {"MF:X": "THB"}, "USD", rates)
        assert out["MF:X"].notna().all()

    def test_a_late_starting_rate_is_reported(self):
        prices, rates = self._setup()
        late = fx.late_rates(prices, {"THAI": "THB"}, "USD", rates)
        assert set(late) == {"THB"}
        assert late["THB"] == pd.Timestamp("2022-01-03")

    def test_full_coverage_reports_nothing(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        rates = pd.DataFrame({"THB": 35.0}, index=idx)
        prices = pd.DataFrame({"MF:X": [35.0] * 10}, index=idx)
        assert fx.late_rates(prices, {"MF:X": "THB"}, "USD", rates) == {}

    def test_a_currency_needed_only_as_the_base_is_checked_too(self):
        prices, rates = self._setup()
        late = fx.late_rates(prices, {"US": "USD"}, "THB", rates)
        assert set(late) == {"THB"}
