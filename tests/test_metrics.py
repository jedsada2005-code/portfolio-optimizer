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
        beta, alpha = metrics.beta_alpha(b, b, 0.02)
        assert beta == pytest.approx(1.0)
        assert alpha == pytest.approx(0.0, abs=1e-9)

    def test_double_exposure_gives_beta_two(self):
        rng = np.random.default_rng(4)
        b = pd.Series(rng.normal(0.0004, 0.01, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
        beta, _ = metrics.beta_alpha(2 * b, b, 0.02)
        assert beta == pytest.approx(2.0)

    def test_constant_outperformance_shows_positive_alpha(self):
        rng = np.random.default_rng(4)
        b = pd.Series(rng.normal(0.0004, 0.01, 1000), index=pd.bdate_range("2020-01-01", periods=1000))
        _, alpha = metrics.beta_alpha(b + 0.0002, b, 0.02)
        assert alpha > 0

    def test_a_flat_benchmark_gives_no_beta(self):
        idx = pd.bdate_range("2020-01-01", periods=50)
        beta, alpha = metrics.beta_alpha(pd.Series(0.001, index=idx), pd.Series(0.0, index=idx), 0.02)
        assert beta == 0.0

    def test_only_overlapping_dates_are_compared(self):
        rng = np.random.default_rng(4)
        idx = pd.bdate_range("2020-01-01", periods=300)
        b = pd.Series(rng.normal(0.0004, 0.01, 300), index=idx)
        beta, _ = metrics.beta_alpha(b.iloc[100:], b, 0.02)
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


class TestBlendPerformance:
    def test_cash_scales_return_and_volatility_together(self):
        ret, vol = metrics.blend_performance(0.12, 0.20, 0.02, 0.25)
        assert ret == pytest.approx(0.75 * 0.12 + 0.25 * 0.02)
        assert vol == pytest.approx(0.75 * 0.20)

    def test_no_cash_leaves_the_figures_alone(self):
        assert metrics.blend_performance(0.12, 0.20, 0.02, 0.0) == (0.12, 0.20)

    def test_all_cash_gives_the_risk_free_rate_and_no_risk(self):
        assert metrics.blend_performance(0.12, 0.20, 0.02, 1.0) == pytest.approx((0.02, 0.0))


class TestHistoryRequirement:
    """52 overlapping trailing-year windows sounds like 52 observations,
    but consecutive windows share 51 of their 52 weeks -- autocorrelation
    runs above 0.96 -- so the real sample is the number of independent
    years, which is far smaller."""

    def test_observations_needed_grows_with_the_years_asked_for(self):
        assert metrics.observations_for_years(2) == 52
        assert metrics.observations_for_years(3) == 104
        assert metrics.observations_for_years(5) == 208

    def test_one_year_needs_a_single_window(self):
        assert metrics.observations_for_years(1) == 1

    def test_independent_years_ignores_the_overlap(self):
        # 104 overlapping windows come from three years of weekly data,
        # which is three independent annual observations.
        assert metrics.independent_years(104) == 3
        assert metrics.independent_years(52) == 2
        assert metrics.independent_years(0) == 1

    def test_the_floor_is_expressed_in_years(self):
        counts = pd.Series({"OLD": 400, "NEW": 60})
        flagged = metrics.unreliable_assets(counts, metrics.observations_for_years(3))
        assert flagged == {"NEW": 60}

    def test_a_two_year_floor_still_admits_shorter_history(self):
        counts = pd.Series({"NEW": 60})
        assert metrics.unreliable_assets(counts, metrics.observations_for_years(2)) == {}


class TestAlignBenchmark:
    """A benchmark that listed after the portfolio began leaves NaNs at
    the front of the reindexed price series. Turning those into 0%
    returns made it look flat through years it did not exist for, which
    flattered every comparison drawn against it."""

    @staticmethod
    def _portfolio(index):
        return pd.Series(0.0004, index=index)

    def test_a_later_listing_is_measured_from_its_own_inception(self):
        index = pd.bdate_range("2020-01-01", "2024-12-31")
        prices = pd.Series(
            100.0 * 1.0005 ** np.arange(len(index)), index=index
        )
        prices.loc[:"2022-01-01"] = np.nan

        aligned = metrics.align_benchmark(self._portfolio(index), prices)

        assert aligned.start == prices.first_valid_index()
        # 0.05% a business day, compounded, is what the benchmark really did.
        stats = metrics.backtest_stats(aligned.benchmark, 0.0)
        assert stats["annual_return"] == pytest.approx(1.0005 ** 261 - 1, rel=0.02)

    def test_both_sides_cover_the_same_days(self):
        index = pd.bdate_range("2020-01-01", "2024-12-31")
        prices = pd.Series(100.0 * 1.0005 ** np.arange(len(index)), index=index)
        prices.loc[:"2022-01-01"] = np.nan

        aligned = metrics.align_benchmark(self._portfolio(index), prices)

        assert aligned.portfolio.index.equals(aligned.benchmark.index)
        assert aligned.portfolio.index[0] == aligned.start

    def test_both_sides_start_from_the_same_base(self):
        index = pd.bdate_range("2020-01-01", "2024-12-31")
        prices = pd.Series(100.0 * 1.0005 ** np.arange(len(index)), index=index)
        prices.loc[:"2022-01-01"] = np.nan

        aligned = metrics.align_benchmark(self._portfolio(index), prices)

        assert aligned.portfolio.iloc[0] == 0.0
        assert aligned.benchmark.iloc[0] == 0.0

    def test_a_fully_overlapping_benchmark_keeps_the_whole_window(self):
        index = pd.bdate_range("2020-01-01", "2024-12-31")
        prices = pd.Series(100.0 * 1.0005 ** np.arange(len(index)), index=index)

        aligned = metrics.align_benchmark(self._portfolio(index), prices)

        assert aligned.start == index[0]
        assert len(aligned.portfolio) == len(index)

    def test_a_benchmark_that_never_overlaps_returns_nothing(self):
        index = pd.bdate_range("2020-01-01", "2020-12-31")
        prices = pd.Series(
            100.0, index=pd.bdate_range("2022-01-01", "2022-12-31")
        )

        aligned = metrics.align_benchmark(self._portfolio(index), prices)

        assert aligned.benchmark.empty
        assert aligned.start is None

    def test_an_empty_benchmark_returns_nothing(self):
        index = pd.bdate_range("2020-01-01", "2020-12-31")
        aligned = metrics.align_benchmark(
            self._portfolio(index), pd.Series(dtype=float)
        )
        assert aligned.benchmark.empty
        assert aligned.portfolio.empty


class TestRealisedReturns:
    """The frontier's expected returns and the backtest's realised ones
    are two different numbers on two different tabs, and nothing on
    screen let a reader see how far apart an estimator had put them."""

    def test_a_steady_riser_reports_its_own_growth_rate(self):
        index = pd.bdate_range("2015-01-01", "2024-12-31")
        prices = pd.DataFrame(
            {"UP": 100.0 * 1.10 ** (np.arange(len(index)) / 261.0)}, index=index
        )
        realised = metrics.realised_returns(prices)
        assert realised["UP"] == pytest.approx(0.10, abs=2e-3)

    def test_a_faller_reports_a_negative_rate(self):
        index = pd.bdate_range("2015-01-01", "2024-12-31")
        prices = pd.DataFrame(
            {"DOWN": 100.0 * 0.95 ** (np.arange(len(index)) / 261.0)}, index=index
        )
        assert metrics.realised_returns(prices)["DOWN"] < 0

    def test_each_column_is_measured_over_its_own_history(self):
        index = pd.bdate_range("2015-01-01", "2024-12-31")
        late = 100.0 * 1.20 ** (np.arange(len(index)) / 261.0)
        prices = pd.DataFrame({
            "OLD": 100.0 * 1.10 ** (np.arange(len(index)) / 261.0),
            "NEW": late,
        }, index=index)
        prices.loc[:"2022-01-01", "NEW"] = np.nan

        realised = metrics.realised_returns(prices)
        assert realised["OLD"] == pytest.approx(0.10, abs=2e-3)
        assert realised["NEW"] == pytest.approx(0.20, abs=2e-3)

    def test_a_column_with_no_prices_is_left_out(self):
        index = pd.bdate_range("2015-01-01", "2015-12-31")
        prices = pd.DataFrame({
            "REAL": 100.0 * 1.10 ** (np.arange(len(index)) / 261.0),
            "EMPTY": np.nan,
        }, index=index)
        assert "EMPTY" not in metrics.realised_returns(prices).index


class TestEffectiveHoldings:
    """A weights table shows what is held; it does not show how much of
    the portfolio one position really is."""

    def test_equal_weights_are_worth_their_own_count(self):
        assert metrics.effective_holdings(
            {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        ) == pytest.approx(4.0)

    def test_one_dominant_holding_counts_for_little_more_than_one(self):
        assert metrics.effective_holdings(
            {"A": 0.9, "B": 0.05, "C": 0.05}
        ) == pytest.approx(1.0 / (0.81 + 0.0025 + 0.0025))

    def test_zero_weights_do_not_count(self):
        assert metrics.effective_holdings(
            {"A": 0.5, "B": 0.5, "C": 0.0}
        ) == pytest.approx(2.0)

    def test_unnormalised_weights_are_normalised_first(self):
        assert metrics.effective_holdings({"A": 50.0, "B": 50.0}) == pytest.approx(2.0)

    def test_an_empty_portfolio_is_worth_nothing(self):
        assert metrics.effective_holdings({}) == 0.0
        assert metrics.effective_holdings({"A": 0.0}) == 0.0


class TestRollingCorrelation:
    """A single correlation matrix for the whole window hides the thing
    diversification lives or dies by: SPY and TLT averaged -0.25 over
    2007-2025 but ran at +0.12 through 2022, and the average pair went
    from 0.20 in calm years to 0.51 in the first quarter of 2020."""

    @staticmethod
    def _weekly(periods=300):
        return pd.date_range("2015-01-02", periods=periods, freq="W-FRI")

    def test_two_identical_series_correlate_perfectly_throughout(self):
        index = self._weekly()
        rng = np.random.default_rng(1)
        moves = rng.normal(0.001, 0.02, len(index))
        returns = pd.DataFrame({"A": moves, "B": moves}, index=index)

        rolling = metrics.rolling_correlation(returns, 52)
        assert not rolling.empty
        assert rolling.min() == pytest.approx(1.0)

    def test_mirror_images_correlate_negatively(self):
        index = self._weekly()
        rng = np.random.default_rng(2)
        moves = rng.normal(0.001, 0.02, len(index))
        returns = pd.DataFrame({"A": moves, "B": -moves}, index=index)

        assert metrics.rolling_correlation(returns, 52).max() == pytest.approx(-1.0)

    def test_a_regime_change_shows_up_in_the_line(self):
        index = self._weekly(400)
        rng = np.random.default_rng(3)
        independent = rng.normal(0, 0.02, len(index))
        shared = rng.normal(0, 0.02, len(index))
        # Second half: both assets are driven by the same shock.
        a = independent.copy()
        b = rng.normal(0, 0.02, len(index))
        a[200:], b[200:] = shared[200:], shared[200:]
        returns = pd.DataFrame({"A": a, "B": b}, index=index)

        rolling = metrics.rolling_correlation(returns, 52)
        assert abs(rolling.iloc[0]) < 0.4
        assert rolling.iloc[-1] > 0.9

    def test_every_pair_counts_equally(self):
        index = self._weekly()
        rng = np.random.default_rng(4)
        moves = rng.normal(0.001, 0.02, len(index))
        returns = pd.DataFrame(
            {"A": moves, "B": moves, "C": -moves}, index=index
        )
        # AB is +1, AC and BC are -1, so the mean of the three is -1/3.
        assert metrics.rolling_correlation(returns, 52).iloc[-1] == pytest.approx(-1 / 3)

    def test_a_window_longer_than_the_data_gives_nothing(self):
        index = self._weekly(30)
        rng = np.random.default_rng(5)
        returns = pd.DataFrame(
            {"A": rng.normal(0, 0.02, 30), "B": rng.normal(0, 0.02, 30)}, index=index
        )
        assert metrics.rolling_correlation(returns, 52).empty

    def test_one_holding_has_no_pairs_to_correlate(self):
        index = self._weekly()
        rng = np.random.default_rng(6)
        returns = pd.DataFrame({"A": rng.normal(0, 0.02, len(index))}, index=index)
        assert metrics.rolling_correlation(returns, 52).empty

    def test_a_named_pair_can_be_followed_on_its_own(self):
        index = self._weekly()
        rng = np.random.default_rng(7)
        moves = rng.normal(0.001, 0.02, len(index))
        returns = pd.DataFrame(
            {"A": moves, "B": -moves, "C": rng.normal(0, 0.02, len(index))}, index=index
        )
        pair = metrics.rolling_pair_correlation(returns, "A", "B", 52)
        assert pair.max() == pytest.approx(-1.0)


class TestIndependentAnnualReturns:
    """Trailing-year windows overlap by 51 of their 52 weeks, so treating
    them as a sample badly overstates how much evidence there is."""

    @staticmethod
    def _weekly(years):
        index = pd.date_range("2005-01-07", periods=52 * years, freq="W-FRI")
        return pd.DataFrame(
            {"UP": 100.0 * 1.10 ** (np.arange(len(index)) / 52.0)}, index=index
        )

    def test_ten_years_of_weeks_yield_about_nine_annual_observations(self):
        rows = metrics.independent_annual_returns(self._weekly(10))
        assert 8 <= len(rows) <= 10

    def test_the_windows_do_not_overlap(self):
        rows = metrics.independent_annual_returns(self._weekly(10))
        gaps = rows.index.to_series().diff().dropna().dt.days
        assert (gaps >= 360).all(), gaps.tolist()

    def test_a_steady_riser_reports_its_rate_each_year(self):
        rows = metrics.independent_annual_returns(self._weekly(10))
        assert rows["UP"].mean() == pytest.approx(0.10, abs=5e-3)

    def test_less_than_a_year_yields_nothing(self):
        assert metrics.independent_annual_returns(self._weekly(1)).empty


class TestBootstrapReturnInterval:
    """The optimiser reads an expected return as an exact number and
    allocates to one decimal place off the back of it. On eighteen
    independent years SPY's 11.6% carries a 90% interval of 5.0% to
    18.6% -- which is why 1/N keeps beating the optimisers."""

    @staticmethod
    def _weekly(years, drift, vol, seed=0):
        index = pd.date_range("2005-01-07", periods=52 * years, freq="W-FRI")
        rng = np.random.default_rng(seed)
        moves = rng.normal(drift, vol, len(index))
        return pd.DataFrame({"X": 100.0 * np.cumprod(1 + moves)}, index=index)

    def test_the_interval_brackets_the_estimate(self):
        weekly = self._weekly(20, 0.002, 0.02)
        row = metrics.bootstrap_return_interval(weekly).loc["X"]
        middle = metrics.independent_annual_returns(weekly)["X"].mean()
        assert row["low"] < middle < row["high"]

    def test_more_confidence_needs_a_wider_interval(self):
        weekly = self._weekly(20, 0.002, 0.02)
        narrow = metrics.bootstrap_return_interval(weekly, level=0.50).loc["X"]
        wide = metrics.bootstrap_return_interval(weekly, level=0.95).loc["X"]
        assert (wide["high"] - wide["low"]) > (narrow["high"] - narrow["low"])

    def test_a_noisier_asset_gets_a_wider_interval(self):
        calm = metrics.bootstrap_return_interval(self._weekly(20, 0.002, 0.005)).loc["X"]
        wild = metrics.bootstrap_return_interval(self._weekly(20, 0.002, 0.05)).loc["X"]
        assert (wild["high"] - wild["low"]) > (calm["high"] - calm["low"])

    def test_the_same_prices_give_the_same_interval_twice(self):
        weekly = self._weekly(20, 0.002, 0.02)
        first = metrics.bootstrap_return_interval(weekly).loc["X"]
        second = metrics.bootstrap_return_interval(weekly).loc["X"]
        assert first["low"] == second["low"] and first["high"] == second["high"]

    def test_it_reports_how_many_independent_years_it_had(self):
        row = metrics.bootstrap_return_interval(self._weekly(12, 0.002, 0.02)).loc["X"]
        assert 10 <= row["samples"] <= 12

    def test_an_asset_with_too_little_history_is_left_out(self):
        weekly = self._weekly(20, 0.002, 0.02)
        weekly["NEW"] = np.nan
        weekly.iloc[-40:, weekly.columns.get_loc("NEW")] = 100.0
        table = metrics.bootstrap_return_interval(weekly)
        assert "X" in table.index
        assert "NEW" not in table.index


class TestFfillWithinLife:
    """A holiday is a gap to bridge; a holding that stops reporting is
    not. Filling past the last real price turns it into a flat,
    zero-volatility line the optimiser reads as a safe asset -- its
    measured volatility fell from 32.4% to 22.9% in the case that
    prompted this."""

    @staticmethod
    def _frame():
        index = pd.bdate_range("2020-01-01", "2020-03-31")
        frame = pd.DataFrame({
            "LIVE": np.arange(1.0, len(index) + 1.0),
            "STOPS": np.arange(1.0, len(index) + 1.0),
            "GAPPY": np.arange(1.0, len(index) + 1.0),
        }, index=index)
        frame.loc["2020-02-14":, "STOPS"] = np.nan      # stops reporting
        frame.loc["2020-02-03":"2020-02-05", "GAPPY"] = np.nan  # a holiday run
        return frame

    def test_an_interior_gap_is_still_bridged(self):
        filled = metrics.ffill_within_life(self._frame())
        assert filled.loc["2020-02-04", "GAPPY"] == filled.loc["2020-01-31", "GAPPY"]

    def test_nothing_is_invented_past_the_last_real_price(self):
        filled = metrics.ffill_within_life(self._frame())
        assert filled.loc["2020-02-14":, "STOPS"].isna().all()

    def test_a_live_column_is_untouched(self):
        frame = self._frame()
        filled = metrics.ffill_within_life(frame)
        pd.testing.assert_series_equal(filled["LIVE"], frame["LIVE"])

    def test_a_column_that_never_reported_stays_empty(self):
        frame = self._frame()
        frame["NEVER"] = np.nan
        assert metrics.ffill_within_life(frame)["NEVER"].isna().all()


class TestCommonEnd:
    """common_start guards the day a portfolio can first be held. Nothing
    guarded the day one of its holdings stopped existing."""

    @staticmethod
    def _frame():
        index = pd.bdate_range("2020-01-01", "2020-03-31")
        frame = pd.DataFrame(
            {"A": 1.0, "B": 1.0, "C": 1.0}, index=index
        )
        frame.loc[:"2020-01-15", "B"] = np.nan     # starts late
        frame.loc["2020-03-02":, "C"] = np.nan     # stops early
        return frame

    def test_it_is_the_day_the_first_holding_stops(self):
        frame = self._frame()
        assert metrics.common_end(frame) == pd.Timestamp("2020-02-28")

    def test_it_pairs_with_common_start(self):
        frame = self._frame()
        start, end = metrics.common_start(frame), metrics.common_end(frame)
        assert start == pd.Timestamp("2020-01-16")
        assert start < end
        assert frame.loc[start:end].notna().all().all()

    def test_a_frame_that_never_overlaps_has_no_end(self):
        index = pd.bdate_range("2020-01-01", "2020-03-31")
        frame = pd.DataFrame({"A": 1.0, "B": np.nan}, index=index)
        assert metrics.common_end(frame) is None

    def test_an_empty_frame_has_no_end(self):
        assert metrics.common_end(pd.DataFrame()) is None

    def test_all_columns_live_means_the_last_row(self):
        index = pd.bdate_range("2020-01-01", "2020-03-31")
        frame = pd.DataFrame({"A": 1.0, "B": 1.0}, index=index)
        assert metrics.common_end(frame) == index[-1]


class TestSimulationStopsWhenAHoldingDoes:
    def test_the_backtest_ends_where_the_holdings_do(self):
        index = pd.bdate_range("2020-01-01", "2020-12-31")
        prices = pd.DataFrame(
            {"A": 100.0 * 1.0004 ** np.arange(len(index)),
             "B": 100.0 * 1.0002 ** np.arange(len(index))}, index=index
        )
        prices.loc["2020-07-01":, "B"] = np.nan

        result = metrics.simulate_portfolio(prices, {"A": 0.5, "B": 0.5}, "M", 0.0)
        assert result.returns.index[-1] == metrics.common_end(prices)
        assert result.returns.index[-1] < index[-1]

    def test_a_dead_holding_cannot_flat_line_the_volatility(self):
        index = pd.bdate_range("2015-01-01", "2025-01-01")
        rng = np.random.default_rng(4)
        prices = pd.DataFrame(
            {"A": 100 * np.cumprod(1 + rng.normal(0.0004, 0.011, len(index))),
             "B": 100 * np.cumprod(1 + rng.normal(0.0006, 0.020, len(index)))},
            index=index,
        )
        prices.loc["2020-01-01":, "B"] = np.nan

        truncated = metrics.simulate_portfolio(prices, {"A": 0.5, "B": 0.5}, "Q", 0.0)
        stretched = metrics.simulate_portfolio(
            prices.ffill(), {"A": 0.5, "B": 0.5}, "Q", 0.0
        )
        honest = metrics.backtest_stats(truncated.returns, 0.02)
        invented = metrics.backtest_stats(stretched.returns, 0.02)
        assert honest["annual_volatility"] > invented["annual_volatility"]
        assert honest["years"] < invented["years"]


class TestAnchoredGrowth:
    """Two growth curves each based at 1.0 on different dates cannot be
    compared by eye. A benchmark that listed later restarted at 1.00
    while the portfolio was already at 4.84, so the chart showed the
    portfolio winning 2.6x while the table -- correctly -- had it losing
    by 9.85% a year."""

    @staticmethod
    def _pair():
        index = pd.bdate_range("2010-01-01", "2025-01-01")
        rng = np.random.default_rng(3)
        port = pd.Series(rng.normal(0.0005, 0.010, len(index)), index=index)
        bench = pd.Series(
            100 * np.cumprod(1 + rng.normal(0.0009, 0.012, len(index))), index=index
        )
        bench.loc[:"2018-01-01"] = np.nan
        return port, bench

    def test_the_benchmark_starts_where_the_portfolio_stood(self):
        port, bench_px = self._pair()
        aligned = metrics.align_benchmark(port, bench_px)
        port_curve = (1.0 + port).cumprod()

        curve = metrics.anchored_growth(aligned.benchmark, port_curve)
        assert curve.iloc[0] == pytest.approx(port_curve.loc[aligned.start])

    def test_the_line_that_ends_higher_is_the_one_that_performed_better(self):
        port, bench_px = self._pair()
        aligned = metrics.align_benchmark(port, bench_px)
        port_curve = (1.0 + port).cumprod()
        bench_curve = metrics.anchored_growth(aligned.benchmark, port_curve)

        port_cagr = metrics.backtest_stats(aligned.portfolio, 0.0)["annual_return"]
        bench_cagr = metrics.backtest_stats(aligned.benchmark, 0.0)["annual_return"]

        chart_says_bench_won = bench_curve.iloc[-1] > port_curve.iloc[-1]
        stats_say_bench_won = bench_cagr > port_cagr
        assert chart_says_bench_won == stats_say_bench_won

    def test_a_benchmark_covering_the_whole_window_is_unchanged(self):
        index = pd.bdate_range("2020-01-01", "2022-12-31")
        rng = np.random.default_rng(8)
        port = pd.Series(rng.normal(0.0004, 0.01, len(index)), index=index)
        bench = pd.Series(rng.normal(0.0003, 0.01, len(index)), index=index)
        bench.iloc[0] = 0.0
        port_curve = (1.0 + port).cumprod()

        curve = metrics.anchored_growth(bench, port_curve)
        assert curve.iloc[0] == pytest.approx(port_curve.iloc[0])

    def test_no_returns_gives_no_curve(self):
        index = pd.bdate_range("2020-01-01", periods=5)
        reference = pd.Series(1.0, index=index)
        assert metrics.anchored_growth(pd.Series(dtype=float), reference).empty

    def test_an_empty_reference_falls_back_to_one(self):
        index = pd.bdate_range("2020-01-01", periods=5)
        returns = pd.Series(0.01, index=index)
        curve = metrics.anchored_growth(returns, pd.Series(dtype=float))
        assert curve.iloc[0] == pytest.approx(1.01)


class TestRatiosSurviveANearZeroDenominator:
    """`if vol > 0` passes on 8e-18, which is what pandas returns for the
    standard deviation of a constant series. The result was a Sharpe of
    1.1e17 rendered as a headline figure."""

    def test_a_constant_return_series_reports_no_sharpe(self):
        index = pd.bdate_range("2024-01-01", periods=400)
        stats = metrics.backtest_stats(pd.Series(0.002, index=index), 0.02)
        assert stats["annual_volatility"] < 1e-12
        assert stats["sharpe"] == 0.0

    def test_a_vanishing_drawdown_reports_no_calmar(self):
        index = pd.bdate_range("2024-01-01", periods=400)
        stats = metrics.backtest_stats(pd.Series(0.002, index=index), 0.02)
        assert stats["calmar"] == 0.0

    def test_a_constant_series_reports_no_sortino(self):
        index = pd.bdate_range("2024-01-01", periods=400)
        assert metrics.backtest_stats(pd.Series(0.002, index=index), 0.02)["sortino"] == 0.0

    def test_a_real_portfolio_still_gets_its_ratios(self):
        index = pd.bdate_range("2015-01-01", periods=2000)
        rng = np.random.default_rng(31)
        stats = metrics.backtest_stats(
            pd.Series(rng.normal(0.0005, 0.010, len(index)), index=index), 0.02
        )
        assert 0.0 < abs(stats["sharpe"]) < 10.0
        assert 0.0 < abs(stats["calmar"]) < 100.0
        assert stats["sortino"] != 0.0


class TestEffectiveHoldingsIgnoresCash:
    """The metric measures how concentrated the risky sleeve is. Cash is
    not a diversifying holding, and counting it made a portfolio look
    more spread out the more of it sat idle."""

    def test_cash_does_not_count_as_a_holding(self):
        risky = {"A": 0.5, "B": 0.3, "C": 0.2}
        bare = metrics.effective_holdings(risky)
        blended = metrics.effective_holdings(metrics.blend_with_cash(risky, 0.4))
        assert blended == pytest.approx(bare)

    def test_more_cash_does_not_raise_the_count(self):
        risky = {"A": 0.6, "B": 0.4}
        counts = [
            metrics.effective_holdings(metrics.blend_with_cash(risky, c))
            for c in (0.0, 0.3, 0.6)
        ]
        assert counts[0] == pytest.approx(counts[1]) == pytest.approx(counts[2])

    def test_a_cash_only_portfolio_holds_nothing_risky(self):
        assert metrics.effective_holdings({metrics.CASH_SYMBOL: 1.0}) == 0.0
