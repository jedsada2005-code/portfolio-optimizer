import numpy as np
import pandas as pd
import pytest

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
