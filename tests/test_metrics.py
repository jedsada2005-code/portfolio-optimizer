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


class TestSplitIndex:
    def test_splits_on_calendar_time_not_row_count(self):
        # Rows are dense in the second half; splitting by position would
        # land far later than the halfway point in time.
        idx = pd.DatetimeIndex(
            list(pd.date_range("2020-01-01", periods=10, freq="30D"))
            + list(pd.date_range("2021-01-01", periods=200, freq="D"))
        )
        split = metrics.split_index(idx, 0.5)
        span = (idx[-1] - idx[0]).days
        assert abs((split - idx[0]).days - span * 0.5) <= 1

    def test_fraction_of_one_returns_the_last_date(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        assert metrics.split_index(idx, 1.0) == idx[-1]

    def test_rejects_a_fraction_outside_the_range(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        for bad in (0.0, 1.5, -0.1):
            with pytest.raises(ValueError):
                metrics.split_index(idx, bad)

    def test_empty_index_returns_none(self):
        assert metrics.split_index(pd.DatetimeIndex([]), 0.7) is None


class TestBacktestStats:
    def _flat_growth(self, daily, days):
        idx = pd.bdate_range("2020-01-01", periods=days)
        return pd.Series([0.0] + [daily] * (days - 1), index=idx)

    def test_reports_every_headline_number(self):
        stats = metrics.backtest_stats(self._flat_growth(0.001, 500), 0.02)
        assert set(stats) == {
            "total_return", "years", "annual_return", "annual_volatility",
            "sharpe", "max_drawdown", "calmar", "sortino", "periods_per_year",
        }

    def test_a_never_losing_series_has_no_drawdown(self):
        stats = metrics.backtest_stats(self._flat_growth(0.001, 500), 0.02)
        assert stats["max_drawdown"] == pytest.approx(0.0)
        assert stats["sortino"] == 0.0  # no downside at all, not infinity
        assert stats["calmar"] == 0.0

    def test_annual_return_compounds_over_calendar_years(self):
        r = self._flat_growth(0.0005, 505)
        stats = metrics.backtest_stats(r, 0.02)
        expected = metrics.cagr((1 + r).prod() - 1, metrics.years_elapsed(r.index))
        assert stats["annual_return"] == pytest.approx(expected)

    def test_empty_series_is_safe(self):
        stats = metrics.backtest_stats(pd.Series(dtype=float), 0.02)
        assert stats["total_return"] == 0.0
        assert stats["sharpe"] == 0.0


class TestBetaAlpha:
    def test_tracking_the_benchmark_exactly_gives_beta_one_alpha_zero(self):
        rng = np.random.default_rng(4)
        b = pd.Series(rng.normal(0.0004, 0.01, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
        beta, alpha = metrics.beta_alpha(b, b, 0.02, 252)
        assert beta == pytest.approx(1.0)
        assert alpha == pytest.approx(0.0, abs=1e-9)

    def test_double_exposure_gives_beta_two(self):
        rng = np.random.default_rng(4)
        b = pd.Series(rng.normal(0.0004, 0.01, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
        beta, _ = metrics.beta_alpha(2 * b, b, 0.02, 252)
        assert beta == pytest.approx(2.0)

    def test_constant_outperformance_shows_positive_alpha(self):
        rng = np.random.default_rng(4)
        b = pd.Series(rng.normal(0.0004, 0.01, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
        _, alpha = metrics.beta_alpha(b + 0.0002, b, 0.02, 252)
        assert alpha > 0

    def test_a_flat_benchmark_gives_no_beta(self):
        idx = pd.bdate_range("2020-01-01", periods=50)
        beta, alpha = metrics.beta_alpha(pd.Series(0.001, index=idx), pd.Series(0.0, index=idx), 0.02, 252)
        assert beta == 0.0

    def test_only_overlapping_dates_are_compared(self):
        rng = np.random.default_rng(4)
        idx = pd.bdate_range("2020-01-01", periods=300)
        b = pd.Series(rng.normal(0.0004, 0.01, 300), index=idx)
        beta, _ = metrics.beta_alpha(b.iloc[100:], b, 0.02, 252)
        assert beta == pytest.approx(1.0)


class TestRebalanceDates:
    def test_daily_rebalances_every_day_after_the_first(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        assert metrics.rebalance_dates(idx, "D") == list(idx[1:])

    def test_buy_and_hold_never_rebalances(self):
        idx = pd.bdate_range("2020-01-01", periods=10)
        assert metrics.rebalance_dates(idx, None) == []

    def test_monthly_fires_once_per_month_boundary(self):
        idx = pd.bdate_range("2020-01-01", "2020-12-31")
        assert len(metrics.rebalance_dates(idx, "M")) == 11

    def test_quarterly_and_yearly(self):
        idx = pd.bdate_range("2020-01-01", "2023-12-31")
        assert len(metrics.rebalance_dates(idx, "Q")) == 15
        assert len(metrics.rebalance_dates(idx, "Y")) == 3


class TestSimulatePortfolio:
    def _prices(self):
        idx = pd.bdate_range("2020-01-01", periods=260)
        rng = np.random.default_rng(11)
        return pd.DataFrame(
            {"A": 100 * np.cumprod(1 + rng.normal(0.001, 0.02, 260)),
             "B": 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, 260))},
            index=idx,
        )

    def test_daily_rebalancing_matches_constant_weight_returns(self):
        # The old dot-product backtest was an implicit daily rebalance;
        # the simulator must reproduce it exactly at zero cost.
        prices = self._prices()
        w = {"A": 0.6, "B": 0.4}
        sim = metrics.simulate_portfolio(prices, w, "D", 0.0)
        reference = metrics.portfolio_daily_returns(prices, w).portfolio
        pd.testing.assert_series_equal(sim.returns, reference, check_names=False)

    def test_buy_and_hold_lets_weights_drift(self):
        prices = self._prices()
        sim = metrics.simulate_portfolio(prices, {"A": 0.5, "B": 0.5}, None, 0.0)
        assert sim.turnover.sum() == 0.0
        assert sim.final_weights["A"] != pytest.approx(0.5, abs=0.01)

    def test_buy_and_hold_equals_the_weighted_sum_of_each_holding(self):
        prices = self._prices()
        w = {"A": 0.3, "B": 0.7}
        sim = metrics.simulate_portfolio(prices, w, None, 0.0)
        growth = (prices.iloc[-1] / prices.iloc[0])
        expected = w["A"] * growth["A"] + w["B"] * growth["B"]
        assert (1 + sim.returns).prod() == pytest.approx(expected)

    def test_trading_costs_reduce_the_result(self):
        prices = self._prices()
        w = {"A": 0.5, "B": 0.5}
        free = (1 + metrics.simulate_portfolio(prices, w, "M", 0.0).returns).prod()
        charged = (1 + metrics.simulate_portfolio(prices, w, "M", 50.0).returns).prod()
        assert charged < free

    def test_more_frequent_rebalancing_trades_more(self):
        prices = self._prices()
        w = {"A": 0.5, "B": 0.5}
        yearly = metrics.simulate_portfolio(prices, w, "Y", 0.0).turnover.sum()
        monthly = metrics.simulate_portfolio(prices, w, "M", 0.0).turnover.sum()
        daily = metrics.simulate_portfolio(prices, w, "D", 0.0).turnover.sum()
        assert yearly < monthly < daily

    def test_costs_do_nothing_when_nothing_is_traded(self):
        prices = self._prices()
        w = {"A": 0.5, "B": 0.5}
        free = (1 + metrics.simulate_portfolio(prices, w, None, 0.0).returns).prod()
        charged = (1 + metrics.simulate_portfolio(prices, w, None, 100.0).returns).prod()
        assert charged == pytest.approx(free)

    def test_starts_only_once_every_holding_exists(self):
        prices = self._prices()
        prices.loc[prices.index[:100], "B"] = np.nan
        sim = metrics.simulate_portfolio(prices, {"A": 0.5, "B": 0.5}, "M", 0.0)
        assert sim.start == prices.index[100]

    def test_no_held_assets_gives_an_empty_result(self):
        sim = metrics.simulate_portfolio(self._prices(), {"A": 0.0, "B": 0.0}, "M", 0.0)
        assert sim.returns.empty
        assert sim.held == []


class TestCashPriceSeries:
    def test_grows_at_the_annual_rate(self):
        idx = pd.date_range("2020-01-01", "2021-01-01", freq="D")
        cash = metrics.cash_price_series(idx, 0.05)
        assert cash.iloc[0] == pytest.approx(1.0)
        # 2020 is a leap year, so this accrues over 366/365.25 years.
        assert cash.iloc[-1] == pytest.approx(1.05 ** (366 / 365.25), abs=1e-9)

    def test_a_zero_rate_stays_flat(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        assert metrics.cash_price_series(idx, 0.0).nunique() == 1

    def test_never_loses_value_at_a_positive_rate(self):
        idx = pd.bdate_range("2020-01-01", periods=500)
        assert metrics.cash_price_series(idx, 0.03).diff().dropna().min() >= 0

    def test_empty_index_gives_an_empty_series(self):
        assert metrics.cash_price_series(pd.DatetimeIndex([]), 0.02).empty


class TestBlendWithCash:
    def test_scales_the_risky_sleeve_and_adds_the_rest(self):
        blended = metrics.blend_with_cash({"A": 0.6, "B": 0.4}, 0.10)
        assert blended["A"] == pytest.approx(0.54)
        assert blended["B"] == pytest.approx(0.36)
        assert blended[metrics.CASH_SYMBOL] == pytest.approx(0.10)
        assert sum(blended.values()) == pytest.approx(1.0)

    def test_zero_cash_leaves_the_weights_alone(self):
        assert metrics.blend_with_cash({"A": 0.6, "B": 0.4}, 0.0) == {"A": 0.6, "B": 0.4}

    def test_all_cash_empties_the_risky_sleeve(self):
        blended = metrics.blend_with_cash({"A": 1.0}, 1.0)
        assert blended[metrics.CASH_SYMBOL] == pytest.approx(1.0)
        assert blended["A"] == pytest.approx(0.0)
