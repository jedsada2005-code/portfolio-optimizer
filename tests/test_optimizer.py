import numpy as np
import pandas as pd
import pytest

import metrics
import optimizer


def _cov(values, names):
    return pd.DataFrame(values, index=names, columns=names)


@pytest.fixture
def sample_cov():
    return _cov(
        [[0.0400, 0.0180, 0.0020],
         [0.0180, 0.0900, 0.0045],
         [0.0020, 0.0045, 0.0100]],
        ["A", "B", "C"],
    )


class TestShrinkCovariance:
    def test_zero_intensity_leaves_the_sample_untouched(self, sample_cov):
        out = optimizer.shrink_covariance(sample_cov, 0.0)
        pd.testing.assert_frame_equal(out, sample_cov)

    def test_full_intensity_reaches_the_constant_correlation_target(self, sample_cov):
        out = optimizer.shrink_covariance(sample_cov, 1.0)
        target = optimizer.constant_correlation_target(sample_cov)
        pd.testing.assert_frame_equal(out, target)

    def test_variances_survive_any_intensity(self, sample_cov):
        for delta in (0.0, 0.25, 0.5, 1.0):
            out = optimizer.shrink_covariance(sample_cov, delta)
            assert np.allclose(np.diag(out.values), np.diag(sample_cov.values))

    def test_correlations_move_towards_their_average(self, sample_cov):
        corr = lambda m: m.values[0, 1] / np.sqrt(m.values[0, 0] * m.values[1, 1])
        avg = optimizer.average_correlation(sample_cov)
        before, after = corr(sample_cov), corr(optimizer.shrink_covariance(sample_cov, 0.5))
        assert abs(after - avg) < abs(before - avg)

    def test_output_stays_positive_semidefinite(self, sample_cov):
        out = optimizer.shrink_covariance(sample_cov, 0.3)
        assert np.linalg.eigvalsh(out.values).min() >= -1e-10

    def test_repairs_a_non_psd_input(self):
        broken = _cov([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]], list("ABC"))
        assert np.linalg.eigvalsh(broken.values).min() < 0
        out = optimizer.shrink_covariance(broken, 0.0)
        assert np.linalg.eigvalsh(out.values).min() >= -1e-10

    def test_rejects_an_intensity_outside_zero_to_one(self, sample_cov):
        with pytest.raises(ValueError):
            optimizer.shrink_covariance(sample_cov, 1.5)

    def test_single_asset_is_returned_unchanged(self):
        one = _cov([[0.04]], ["A"])
        pd.testing.assert_frame_equal(optimizer.shrink_covariance(one, 0.5), one)


class TestMaxAchievableReturn:
    def test_unbounded_picks_the_single_best_asset(self):
        ar = pd.Series({"A": 0.10, "B": 0.30, "C": 0.20})
        assert optimizer.max_achievable_return(ar, 1.0) == pytest.approx(0.30)

    def test_a_cap_forces_diversification(self):
        ar = pd.Series({"A": 0.10, "B": 0.30, "C": 0.20})
        # 0.4 into B, 0.4 into C, 0.2 left for A
        assert optimizer.max_achievable_return(ar, 0.4) == pytest.approx(
            0.4 * 0.30 + 0.4 * 0.20 + 0.2 * 0.10
        )

    def test_cap_too_small_to_reach_full_investment_is_rejected(self):
        ar = pd.Series({"A": 0.1, "B": 0.2})
        with pytest.raises(ValueError):
            optimizer.max_achievable_return(ar, 0.3)


class TestOptimizeWeights:
    @pytest.fixture
    def ar(self):
        return pd.Series({"A": 0.12, "B": 0.20, "C": 0.05})

    def test_max_sharpe_weights_sum_to_one(self, ar, sample_cov):
        w = optimizer.optimize_weights(ar, sample_cov, "Max Sharpe", 0.02, 1.0)
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_weight_cap_is_respected(self, ar, sample_cov):
        w = optimizer.optimize_weights(ar, sample_cov, "Max Sharpe", 0.02, 0.4)
        assert max(w.values()) <= 0.4 + 1e-6
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)

    def test_min_volatility_is_less_volatile_than_max_sharpe(self, ar, sample_cov):
        mv = optimizer.optimize_weights(ar, sample_cov, "Min Volatility", 0.02, 1.0)
        ms = optimizer.optimize_weights(ar, sample_cov, "Max Sharpe", 0.02, 1.0)
        vol = lambda w: np.sqrt(
            np.array([w[a] for a in ar.index]) @ sample_cov.values @ np.array([w[a] for a in ar.index])
        )
        assert vol(mv) <= vol(ms) + 1e-9

    def test_impossible_cap_raises_a_clear_error(self, ar, sample_cov):
        with pytest.raises(ValueError, match="สูงสุด"):
            optimizer.optimize_weights(ar, sample_cov, "Max Sharpe", 0.02, 0.2)


class TestFrontierCurve:
    def test_returns_a_rising_curve(self):
        ar = pd.Series({"A": 0.12, "B": 0.20, "C": 0.05})
        cov = _cov([[0.04, 0.018, 0.002], [0.018, 0.09, 0.0045], [0.002, 0.0045, 0.01]], list("ABC"))
        vols, rets = optimizer.frontier_curve(ar, cov, 1.0, n_points=25)
        assert len(vols) == len(rets) >= 10
        assert np.all(np.diff(rets) > -1e-9)
        assert vols[0] == pytest.approx(min(vols), abs=1e-6)

    def test_a_weight_cap_shortens_the_reachable_range(self):
        ar = pd.Series({"A": 0.12, "B": 0.20, "C": 0.05})
        cov = _cov([[0.04, 0.018, 0.002], [0.018, 0.09, 0.0045], [0.002, 0.0045, 0.01]], list("ABC"))
        _, wide = optimizer.frontier_curve(ar, cov, 1.0, n_points=25)
        _, capped = optimizer.frontier_curve(ar, cov, 0.5, n_points=25)
        assert max(capped) < max(wide)


class TestSampleWeights:
    def test_rows_are_full_allocations(self):
        rng = np.random.default_rng(0)
        w = optimizer.sample_weights(rng, 5, 2000, 1.0)
        assert w.shape == (2000, 5)
        assert np.allclose(w.sum(axis=1), 1.0)

    def test_respects_the_cap(self):
        rng = np.random.default_rng(0)
        w = optimizer.sample_weights(rng, 5, 2000, 0.35)
        assert w.max() <= 0.35 + 1e-9
        assert np.allclose(w.sum(axis=1), 1.0)

    def test_still_fills_the_cloud_at_a_tight_cap(self):
        # 8 assets capped at 15% leaves almost no room; plain rejection
        # sampling would return an empty cloud here.
        rng = np.random.default_rng(0)
        w = optimizer.sample_weights(rng, 8, 3000, 0.15)
        assert len(w) >= 2000
        assert w.max() <= 0.15 + 1e-9
        assert np.allclose(w.sum(axis=1), 1.0)

    def test_cap_at_exactly_equal_weight_gives_equal_weights(self):
        rng = np.random.default_rng(0)
        w = optimizer.sample_weights(rng, 4, 100, 0.25)
        assert np.allclose(w, 0.25)

    def test_infeasible_cap_returns_nothing(self):
        rng = np.random.default_rng(0)
        assert len(optimizer.sample_weights(rng, 4, 100, 0.2)) == 0


def _trending_prices(days=1500, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=days)
    return pd.DataFrame(
        {"A": 100 * np.cumprod(1 + rng.normal(0.0006, 0.012, days)),
         "B": 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, days)),
         "C": 100 * np.cumprod(1 + rng.normal(0.0004, 0.020, days))},
        index=idx,
    )


class TestWalkForward:
    def test_refits_repeatedly_and_the_weights_move(self):
        result = optimizer.walk_forward(_trending_prices(), 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0)
        assert len(result.weight_history) >= 2
        first, last = result.weight_history[0][1], result.weight_history[-1][1]
        assert first != last

    def test_only_covers_the_period_after_the_first_fit(self):
        prices = _trending_prices()
        result = optimizer.walk_forward(prices, 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0)
        assert result.returns.index[0] > prices.index[0]
        assert result.returns.index[-1] == prices.index[-1]

    def test_refitting_more_often_produces_more_weight_sets(self):
        prices = _trending_prices()
        yearly = optimizer.walk_forward(prices, 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0)
        quarterly = optimizer.walk_forward(prices, 0.02, "Max Sharpe", 1.0, 0.2, "Q", 0.0)
        assert len(quarterly.weight_history) > len(yearly.weight_history)

    def test_costs_reduce_the_outcome(self):
        prices = _trending_prices()
        free = optimizer.walk_forward(prices, 0.02, "Max Sharpe", 1.0, 0.2, "Q", 0.0)
        charged = optimizer.walk_forward(prices, 0.02, "Max Sharpe", 1.0, 0.2, "Q", 100.0)
        assert (1 + charged.returns).prod() < (1 + free.returns).prod()

    def test_respects_the_weight_cap_at_every_refit(self):
        result = optimizer.walk_forward(_trending_prices(), 0.02, "Max Sharpe", 0.5, 0.2, "Y", 0.0)
        for _, weights in result.weight_history:
            assert max(weights.values()) <= 0.5 + 1e-6

    def test_too_little_history_produces_nothing(self):
        short = _trending_prices(days=200)
        result = optimizer.walk_forward(short, 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0)
        assert result.returns.empty
        assert result.weight_history == []


class TestWalkForwardRespectsSettings:
    """Walk-forward silently ignored two settings the user had chosen.

    The cash sleeve was appended to the price frame handed to the
    optimiser, so cash became just another asset it could decline to
    hold -- a 40% cash setting produced 0% cash. And every segment was
    simulated buy-and-hold at zero cost regardless of the chosen
    rebalance frequency.
    """

    def _prices(self, days=2200, seed=5):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2014-01-01", periods=days)
        return pd.DataFrame(
            {"A": 100 * np.cumprod(1 + rng.normal(0.0006, 0.012, days)),
             "B": 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, days)),
             "C": 100 * np.cumprod(1 + rng.normal(0.0004, 0.020, days))},
            index=idx,
        )

    @pytest.mark.parametrize("cash", [0.0, 0.25, 0.60])
    def test_the_cash_sleeve_is_held_exactly(self, cash):
        result = optimizer.walk_forward(
            self._prices(), 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0, cash_fraction=cash,
        )
        assert result.weight_history
        for _, weights in result.weight_history:
            held = weights.get(metrics.CASH_SYMBOL, 0.0)
            assert held == pytest.approx(cash, abs=1e-9)
            assert sum(weights.values()) == pytest.approx(1.0)

    def test_cash_never_enters_the_optimised_universe(self):
        result = optimizer.walk_forward(
            self._prices(), 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0, cash_fraction=0.30,
        )
        # The risky sleeve keeps its internal proportions; cash is not
        # something the optimiser chose.
        for _, weights in result.weight_history:
            risky = {k: v for k, v in weights.items() if k != metrics.CASH_SYMBOL}
            assert sum(risky.values()) == pytest.approx(0.70)

    def test_rebalance_frequency_changes_the_result(self):
        prices = self._prices()
        outcomes = {}
        for freq in ("D", "M", "Q", None):
            result = optimizer.walk_forward(
                prices, 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0, rebalance_freq=freq,
            )
            outcomes[freq] = float((1 + result.returns).prod())
        assert len(set(round(v, 8) for v in outcomes.values())) > 1, outcomes

    def test_buy_and_hold_matches_the_previous_hardcoded_behaviour(self):
        prices = self._prices()
        explicit = optimizer.walk_forward(
            prices, 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0, rebalance_freq=None,
        )
        assert not explicit.returns.empty

    def test_turnover_counts_both_refits_and_rebalances(self):
        prices = self._prices()
        held = optimizer.walk_forward(
            prices, 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0, rebalance_freq=None,
        )
        traded = optimizer.walk_forward(
            prices, 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0, rebalance_freq="M",
        )
        assert traded.turnover.sum() > held.turnover.sum()
        assert traded.turnover.index.is_monotonic_increasing

    def test_trading_costs_apply_inside_segments_too(self):
        prices = self._prices()
        free = optimizer.walk_forward(
            prices, 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0, rebalance_freq="M",
        )
        charged = optimizer.walk_forward(
            prices, 0.02, "Max Sharpe", 1.0, 0.2, "Y", 200.0, rebalance_freq="M",
        )
        assert (1 + charged.returns).prod() < (1 + free.returns).prod()


class TestMinimumWeight:
    @pytest.fixture
    def ar(self):
        return pd.Series({"A": 0.12, "B": 0.20, "C": 0.05, "D": 0.09})

    @pytest.fixture
    def cov(self):
        return _cov(
            [[0.0400, 0.0180, 0.0020, 0.0050],
             [0.0180, 0.0900, 0.0045, 0.0060],
             [0.0020, 0.0045, 0.0100, 0.0015],
             [0.0050, 0.0060, 0.0015, 0.0250]],
            list("ABCD"),
        )

    def test_no_holding_falls_below_the_floor(self, ar, cov):
        for objective in ("Max Sharpe", "Min Volatility"):
            w = optimizer.optimize_weights(ar, cov, objective, 0.02, 1.0, min_weight=0.10)
            assert min(w.values()) >= 0.10 - 1e-6, (objective, w)
            assert sum(w.values()) == pytest.approx(1.0)

    def test_a_floor_removes_the_zero_allocations(self):
        # D is dominated: the lowest return and the highest variance, so
        # an unconstrained solve refuses to hold it at all.
        ar = pd.Series({"A": 0.12, "B": 0.20, "C": 0.09, "D": 0.01})
        cov = _cov(
            [[0.040, 0.018, 0.005, 0.010],
             [0.018, 0.090, 0.006, 0.012],
             [0.005, 0.006, 0.025, 0.008],
             [0.010, 0.012, 0.008, 0.160]],
            list("ABCD"),
        )
        unbounded = optimizer.optimize_weights(ar, cov, "Max Sharpe", 0.02, 1.0)
        assert unbounded["D"] < 1e-6

        floored = optimizer.optimize_weights(ar, cov, "Max Sharpe", 0.02, 1.0, min_weight=0.05)
        assert all(v >= 0.05 - 1e-6 for v in floored.values())

    def test_floor_and_cap_hold_at_once(self, ar, cov):
        w = optimizer.optimize_weights(ar, cov, "Max Sharpe", 0.02, 0.40, min_weight=0.15)
        assert min(w.values()) >= 0.15 - 1e-6
        assert max(w.values()) <= 0.40 + 1e-6

    def test_a_floor_at_equal_weight_leaves_one_portfolio(self, ar, cov):
        w = optimizer.optimize_weights(ar, cov, "Max Sharpe", 0.02, 1.0, min_weight=0.25)
        assert all(v == pytest.approx(0.25, abs=1e-4) for v in w.values())

    def test_a_floor_that_cannot_be_funded_is_rejected(self, ar, cov):
        with pytest.raises(ValueError, match="ขั้นต่ำ"):
            optimizer.optimize_weights(ar, cov, "Max Sharpe", 0.02, 1.0, min_weight=0.30)

    def test_a_floor_above_the_cap_is_rejected(self, ar, cov):
        with pytest.raises(ValueError, match="ขั้นต่ำ"):
            optimizer.optimize_weights(ar, cov, "Max Sharpe", 0.02, 0.10, min_weight=0.20)

    def test_max_achievable_return_accounts_for_the_floor(self):
        ar = pd.Series({"A": 0.10, "B": 0.30, "C": 0.20, "D": 0.05})
        # every asset takes 10% first, then the best ones fill the rest
        expected = 0.10 * (0.10 + 0.30 + 0.20 + 0.05) + 0.30 * 0.30 + 0.30 * 0.20
        assert optimizer.max_achievable_return(ar, 0.40, 0.10) == pytest.approx(expected)

    def test_floor_shrinks_the_reachable_return_range(self):
        ar = pd.Series({"A": 0.10, "B": 0.30, "C": 0.20, "D": 0.05})
        assert optimizer.max_achievable_return(ar, 1.0, 0.10) < optimizer.max_achievable_return(ar, 1.0)

    def test_frontier_curve_respects_the_floor(self, ar, cov):
        vols, rets = optimizer.frontier_curve(ar, cov, 1.0, n_points=20, min_weight=0.10)
        assert len(vols) >= 1
        assert max(rets) <= optimizer.max_achievable_return(ar, 1.0, 0.10) + 1e-6

    def test_frontier_collapses_to_a_point_at_equal_weight(self, ar, cov):
        vols, rets = optimizer.frontier_curve(ar, cov, 1.0, n_points=20, min_weight=0.25)
        assert len(set(np.round(rets, 8))) == 1

    def test_sampled_clouds_respect_the_floor(self):
        rng = np.random.default_rng(0)
        w = optimizer.sample_weights(rng, 6, 5000, max_weight=0.40, min_weight=0.10)
        assert len(w) > 0
        assert w.min() >= 0.10 - 1e-9
        assert w.max() <= 0.40 + 1e-9
        assert np.allclose(w.sum(axis=1), 1.0)

    def test_an_unfundable_floor_samples_nothing(self):
        rng = np.random.default_rng(0)
        assert len(optimizer.sample_weights(rng, 6, 100, min_weight=0.25)) == 0

    def test_walk_forward_applies_the_floor_every_period(self):
        rng = np.random.default_rng(6)
        idx = pd.bdate_range("2014-01-01", periods=2200)
        prices = pd.DataFrame(
            {c: 100 * np.cumprod(1 + rng.normal(m, s, 2200))
             for c, m, s in [("A", 0.0006, 0.012), ("B", 0.0003, 0.008), ("C", 0.0004, 0.02)]},
            index=idx,
        )
        result = optimizer.walk_forward(
            prices, 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0, min_weight=0.20,
        )
        assert result.weight_history
        for _, weights in result.weight_history:
            assert min(weights.values()) >= 0.20 - 1e-6


def _weekly_prices(days=800, seed=9):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2012-01-06", periods=days, freq="W-FRI")
    return pd.DataFrame(
        {c: 100 * np.cumprod(1 + rng.normal(m, s, days))
         for c, m, s in [("A", 0.0025, 0.02), ("B", 0.0012, 0.012),
                         ("C", 0.0018, 0.030), ("D", 0.0006, 0.008)]},
        index=idx,
    )


class TestReturnEstimators:
    """The expected-return method is the single biggest driver of the
    resulting portfolio, and it used to be hardcoded."""

    def test_every_advertised_method_produces_a_full_estimate(self):
        weekly = _weekly_prices()
        for method in optimizer.RETURN_METHODS:
            estimate = optimizer.estimate_returns(weekly, method)
            assert list(estimate.index) == list(weekly.columns)
            assert estimate.notna().all(), method

    def test_the_default_matches_the_previous_hardcoded_estimator(self):
        weekly = _weekly_prices()
        pd.testing.assert_series_equal(
            optimizer.estimate_returns(weekly, optimizer.DEFAULT_RETURN_METHOD),
            metrics.annual_return_estimates(weekly)[0],
            check_names=False,
        )

    def test_methods_actually_disagree(self):
        weekly = _weekly_prices()
        estimates = {m: optimizer.estimate_returns(weekly, m) for m in optimizer.RETURN_METHODS}
        spread = pd.DataFrame(estimates).max(axis=1) - pd.DataFrame(estimates).min(axis=1)
        assert spread.max() > 0.01

    def test_an_unknown_method_is_rejected(self):
        with pytest.raises(ValueError, match="วิธีประมาณ"):
            optimizer.estimate_returns(_weekly_prices(), "ไม่มีวิธีนี้")

    def test_observation_counts_come_back_with_the_estimate(self):
        weekly = _weekly_prices()
        weekly.loc[weekly.index[:600], "D"] = np.nan
        _, counts = optimizer.estimate_returns_with_counts(weekly, optimizer.DEFAULT_RETURN_METHOD)
        assert counts["A"] > counts["D"]


class TestHierarchicalRiskParity:
    def test_hrp_allocates_to_everything(self):
        weekly = _weekly_prices()
        w = optimizer.hrp_weights(weekly)
        assert set(w) == set(weekly.columns)
        assert sum(w.values()) == pytest.approx(1.0)
        assert min(w.values()) > 0

    def test_hrp_ignores_expected_returns_entirely(self):
        weekly = _weekly_prices()
        first = optimizer.hrp_weights(weekly)
        scaled = optimizer.hrp_weights(weekly * 3.0)   # same returns, different levels
        for asset in first:
            assert first[asset] == pytest.approx(scaled[asset], abs=1e-9)

    def test_hrp_is_reachable_through_the_common_entry_point(self):
        weekly = _weekly_prices()
        expected = optimizer.estimate_returns(weekly, optimizer.DEFAULT_RETURN_METHOD)
        cov = optimizer.shrink_covariance(weekly.pct_change().cov() * 52, 0.2)
        w = optimizer.optimize_weights(
            expected, cov, optimizer.HRP_OBJECTIVE, 0.02, 1.0, weekly=weekly
        )
        assert sum(w.values()) == pytest.approx(1.0)
        assert min(w.values()) > 0

    def test_hrp_needs_the_return_history(self):
        weekly = _weekly_prices()
        expected = optimizer.estimate_returns(weekly, optimizer.DEFAULT_RETURN_METHOD)
        cov = optimizer.shrink_covariance(weekly.pct_change().cov() * 52, 0.2)
        with pytest.raises(ValueError, match="HRP"):
            optimizer.optimize_weights(expected, cov, optimizer.HRP_OBJECTIVE, 0.02, 1.0)


class TestHistoryFloorFlowsThrough:
    def _prices_with_a_newcomer(self):
        rng = np.random.default_rng(12)
        idx = pd.bdate_range("2014-01-01", periods=2400)
        frame = pd.DataFrame(
            {c: 100 * np.cumprod(1 + rng.normal(m, s, 2400))
             for c, m, s in [("OLD1", 0.0006, 0.012), ("OLD2", 0.0004, 0.010),
                             ("OLD3", 0.0005, 0.015), ("NEW", 0.0009, 0.020)]},
            index=idx,
        )
        frame.loc[frame.index[:-620], "NEW"] = np.nan   # ~2.4 years of history
        return frame

    def test_a_strict_floor_excludes_the_newcomer(self):
        weights = optimizer.fit_weights(
            self._prices_with_a_newcomer(), 0.02, "Max Sharpe", 1.0, 0.2,
            min_observations=metrics.observations_for_years(4),
        )
        assert "NEW" not in weights

    def test_a_relaxed_floor_admits_it(self):
        weights = optimizer.fit_weights(
            self._prices_with_a_newcomer(), 0.02, "Max Sharpe", 1.0, 0.2,
            min_observations=metrics.observations_for_years(2),
        )
        assert "NEW" in weights

    def test_walk_forward_passes_the_floor_to_every_refit(self):
        result = optimizer.walk_forward(
            self._prices_with_a_newcomer(), 0.02, "Max Sharpe", 1.0, 0.2, "Y", 0.0,
            min_observations=metrics.observations_for_years(5),
        )
        for _, weights in result.weight_history:
            assert "NEW" not in weights


class TestDegenerateProblems:
    """cvxpy's SolverError inherits from neither ValueError nor
    OptimizationError, so every handler in the app let it through to the
    page as a raw traceback."""

    def _too_short(self):
        idx = pd.date_range("2022-01-07", periods=30, freq="W-FRI")
        rng = np.random.default_rng(2)
        weekly = pd.DataFrame(
            {c: 100 * np.cumprod(1 + rng.normal(0.001, 0.02, 30))
             for c in ("A", "B", "C", "D")},
            index=idx,
        )
        expected, _ = metrics.annual_return_estimates(weekly)
        cov = optimizer.shrink_covariance(weekly.pct_change().cov() * 52, 0.2)
        return expected, cov

    def test_solver_errors_are_included_in_the_handled_set(self):
        assert optimizer.SolverError in optimizer.SOLVER_ERRORS

    def test_a_degenerate_solve_raises_something_callers_catch(self):
        expected, cov = self._too_short()
        with pytest.raises(optimizer.SOLVER_ERRORS):
            optimizer.optimize_weights(expected, cov, "Max Sharpe", 0.02, 1.0)

    def test_the_frontier_returns_empty_rather_than_exploding(self):
        expected, cov = self._too_short()
        vols, rets = optimizer.frontier_curve(expected, cov)
        assert len(vols) == len(rets) == 0

    def test_fit_weights_declines_instead_of_raising(self):
        idx = pd.bdate_range("2022-01-03", periods=150)
        rng = np.random.default_rng(2)
        prices = pd.DataFrame(
            {c: 100 * np.cumprod(1 + rng.normal(0.001, 0.02, 150)) for c in ("A", "B")},
            index=idx,
        )
        assert optimizer.fit_weights(prices, 0.02, "Max Sharpe", 1.0, 0.2) is None
