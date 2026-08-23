import numpy as np
import pandas as pd
import pytest

import metrics


def _bdays(start, periods):
    return pd.bdate_range(start, periods=periods)


class TestYearsElapsed:
    def test_uses_calendar_span_not_row_count(self):
        # A union of US + Thai trading calendars has far more than 252
        # rows per year, so counting rows overstates the elapsed time.
        idx = _bdays("2015-01-01", 1)
        idx = pd.bdate_range("2015-01-01", "2024-12-31")
        extra = pd.date_range("2015-01-01", "2024-12-31", freq="D")
        union = idx.union(extra[extra.dayofweek < 5])
        assert len(union) / 252 == pytest.approx(10.35, abs=0.02)  # the old, wrong way
        assert metrics.years_elapsed(union) == pytest.approx(10.0, abs=0.02)

    def test_single_row_does_not_divide_by_zero(self):
        assert metrics.years_elapsed(_bdays("2020-01-01", 1)) > 0

    def test_empty_index_is_safe(self):
        assert metrics.years_elapsed(pd.DatetimeIndex([])) > 0


class TestPeriodsPerYear:
    def test_plain_business_day_calendar_is_about_261(self):
        # bdate_range keeps public holidays, so it runs ~261 days/year.
        idx = pd.bdate_range("2015-01-01", "2024-12-31")
        assert metrics.periods_per_year(idx) == pytest.approx(261, abs=2)

    def test_real_trading_calendar_is_about_252(self):
        # Drop ~9 holidays a year to approximate a real exchange calendar.
        idx = pd.bdate_range("2015-01-01", "2024-12-31")
        rng = np.random.default_rng(1)
        keep = rng.permutation(len(idx))[: int(len(idx) * 252 / 261)]
        idx = idx[np.sort(keep)]
        assert metrics.periods_per_year(idx) == pytest.approx(252, abs=4)

    def test_union_calendar_is_higher_than_252(self):
        idx = pd.date_range("2015-01-01", "2024-12-31", freq="D")
        assert metrics.periods_per_year(idx) == pytest.approx(365.25, abs=2)

    def test_degenerate_index_falls_back_to_252(self):
        assert metrics.periods_per_year(pd.DatetimeIndex([])) == 252.0


class TestDownsideDeviation:
    def test_uses_all_periods_not_only_negative_ones(self):
        # std() of just the negative days both de-means them and divides
        # by the count of negatives, understating downside risk.
        rng = np.random.default_rng(0)
        r = pd.Series(rng.normal(0.0004, 0.01, 2520))
        naive = r[r < 0].std() * np.sqrt(252)
        correct = metrics.downside_deviation(r, 252)
        assert correct > naive
        assert correct == pytest.approx(
            np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(252)
        )

    def test_no_negative_returns_gives_zero(self):
        r = pd.Series([0.01, 0.02, 0.0])
        assert metrics.downside_deviation(r, 252) == 0.0

    def test_respects_a_nonzero_target(self):
        r = pd.Series([0.01, 0.01, 0.01])
        assert metrics.downside_deviation(r, 252, target=0.02) > 0

    def test_empty_series_gives_zero(self):
        assert metrics.downside_deviation(pd.Series(dtype=float), 252) == 0.0


class TestSortinoRatio:
    def test_matches_manual_formula(self):
        assert metrics.sortino_ratio(0.10, 0.05, 0.02) == pytest.approx(1.6)

    def test_zero_downside_returns_zero_instead_of_inf(self):
        assert metrics.sortino_ratio(0.10, 0.0, 0.02) == 0.0


class TestMonthLabels:
    def test_labels_follow_actual_month_numbers(self):
        # A backtest starting in March must not be labelled Jan..Jun.
        assert metrics.month_labels([3, 4, 5, 6, 7, 8]) == [
            "Mar", "Apr", "May", "Jun", "Jul", "Aug"
        ]

    def test_full_year_is_unchanged(self):
        assert metrics.month_labels(range(1, 13)) == [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]

    def test_non_contiguous_months(self):
        assert metrics.month_labels([1, 6, 12]) == ["Jan", "Jun", "Dec"]


class TestCagr:
    def test_compounds_over_calendar_years(self):
        assert metrics.cagr(0.21, 2.0) == pytest.approx(0.1, abs=1e-9)

    def test_zero_years_falls_back_to_total_return(self):
        assert metrics.cagr(0.21, 0.0) == pytest.approx(0.21)

    def test_total_loss_does_not_raise(self):
        assert metrics.cagr(-1.0, 3.0) == -1.0


class TestFirstValidDates:
    def test_reports_each_assets_own_start(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        prices = pd.DataFrame({"OLD": range(10), "NEW": [np.nan] * 6 + [1, 2, 3, 4]}, index=idx)
        firsts = metrics.first_valid_dates(prices)
        assert firsts["OLD"] == idx[0]
        assert firsts["NEW"] == idx[6]


class TestCommonStart:
    def test_is_the_latest_asset_inception(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        prices = pd.DataFrame({"OLD": range(10), "NEW": [np.nan] * 6 + [1, 2, 3, 4]}, index=idx)
        assert metrics.common_start(prices) == idx[6]

    def test_none_when_histories_never_overlap(self):
        idx = pd.bdate_range("2020-01-01", periods=6)
        prices = pd.DataFrame(
            {"A": [1, 2, 3, np.nan, np.nan, np.nan], "B": [np.nan] * 3 + [1, 2, 3]}, index=idx
        )
        assert metrics.common_start(prices) is None

    def test_none_for_empty_frame(self):
        assert metrics.common_start(pd.DataFrame()) is None


class TestPortfolioDailyReturns:
    def _mixed(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        # OLD compounds at exactly 10% a day, NEW at exactly 100% a day.
        return pd.DataFrame(
            {"OLD": 100.0 * 1.10 ** np.arange(10),
             "NEW": [np.nan] * 6 + list(100.0 * 2.0 ** np.arange(4))},
            index=idx,
        )

    def test_waits_for_every_held_asset_to_exist(self):
        prices = self._mixed()
        res = metrics.portfolio_daily_returns(prices, {"OLD": 0.5, "NEW": 0.5})
        assert res.start == prices.index[6]
        assert res.portfolio.index[0] == prices.index[6]

    def test_pre_inception_asset_no_longer_acts_as_zero_return_cash(self):
        # OLD compounds at exactly 10% a day. With the old fillna(0) the
        # portfolio earned 5% on days NEW did not exist yet; now those
        # days are simply not part of the backtest.
        prices = self._mixed()
        res = metrics.portfolio_daily_returns(prices, {"OLD": 0.5, "NEW": 0.5})
        # first real day after the common start: OLD +10%, NEW +100%
        assert res.portfolio.iloc[1] == pytest.approx(0.5 * 0.10 + 0.5 * 1.00)
        assert 0.05 not in set(np.round(res.portfolio.values, 10))

    def test_ignores_assets_with_zero_weight(self):
        prices = self._mixed()
        res = metrics.portfolio_daily_returns(prices, {"OLD": 1.0, "NEW": 0.0})
        assert res.held == ["OLD"]
        assert res.start == prices.index[0]

    def test_renormalises_weights_that_do_not_sum_to_one(self):
        prices = self._mixed()
        res = metrics.portfolio_daily_returns(prices, {"OLD": 2.0, "NEW": 2.0})
        assert res.portfolio.iloc[1] == pytest.approx(0.5 * 0.10 + 0.5 * 1.00)

    def test_asset_returns_share_the_portfolio_window(self):
        prices = self._mixed()
        res = metrics.portfolio_daily_returns(prices, {"OLD": 0.5, "NEW": 0.5})
        assert list(res.assets.columns) == ["OLD", "NEW"]
        assert res.assets.index.equals(res.portfolio.index)

    def test_no_overlap_gives_an_empty_result(self):
        idx = pd.bdate_range("2020-01-01", periods=6)
        prices = pd.DataFrame(
            {"A": [1, 2, 3, np.nan, np.nan, np.nan], "B": [np.nan] * 3 + [1, 2, 3]}, index=idx
        )
        res = metrics.portfolio_daily_returns(prices, {"A": 0.5, "B": 0.5})
        assert res.start is None
        assert res.portfolio.empty

    def test_no_held_assets_gives_an_empty_result(self):
        res = metrics.portfolio_daily_returns(self._mixed(), {"OLD": 0.0, "NEW": 0.0})
        assert res.held == []
        assert res.portfolio.empty


class TestAnnualReturnEstimates:
    def test_counts_observations_per_asset(self):
        idx = pd.date_range("2013-01-04", periods=600, freq="W-FRI")
        rng = np.random.default_rng(5)
        w = pd.DataFrame({"OLD": np.cumprod(1 + rng.normal(0.002, 0.02, 600)), "NEW": np.nan}, index=idx)
        w.loc[w.index[-55:], "NEW"] = np.cumprod(1 + rng.normal(0.02, 0.02, 55))
        ar, counts = metrics.annual_return_estimates(w)
        assert counts["OLD"] == 548
        assert counts["NEW"] == 3
        assert not np.isnan(ar["NEW"])  # the old NaN-only guard let this through

    def test_flags_assets_below_the_observation_floor(self):
        counts = pd.Series({"OLD": 548, "NEW": 3, "MID": 52})
        assert metrics.unreliable_assets(counts, 52) == {"NEW": 3}

    def test_nothing_flagged_when_all_have_enough(self):
        counts = pd.Series({"A": 100, "B": 60})
        assert metrics.unreliable_assets(counts, 52) == {}
