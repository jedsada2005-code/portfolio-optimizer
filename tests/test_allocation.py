import numpy as np
import pandas as pd
import pytest

import allocation
import metrics


PRICES = pd.Series({"SPY": 578.32, "GLD": 240.63, "TLT": 81.87, "EEM": 40.76})


class TestAllocateUnits:
    def test_whole_units_only_and_the_rest_stays_as_change(self):
        plan = allocation.allocate_units({"SPY": 0.5, "GLD": 0.5}, PRICES, 100_000)
        for row in plan.rows:
            assert float(row["units"]).is_integer()
        spent = sum(row["value"] for row in plan.rows)
        assert spent + plan.leftover == pytest.approx(100_000)
        assert 0 <= plan.leftover < max(PRICES)

    def test_greedy_spending_keeps_the_change_small(self):
        plan = allocation.allocate_units(
            {"SPY": 0.3, "GLD": 0.3, "TLT": 0.2, "EEM": 0.2}, PRICES, 100_000
        )
        assert plan.leftover < 100_000 * 0.01

    def test_fractional_assets_take_their_exact_share(self):
        plan = allocation.allocate_units(
            {"MF:FUND": 0.4, "SPY": 0.6},
            pd.Series({"MF:FUND": 12.3456, "SPY": 578.32}),
            100_000,
            fractional={"MF:FUND"},
        )
        fund = next(r for r in plan.rows if r["asset"] == "MF:FUND")
        assert fund["value"] == pytest.approx(40_000)
        assert fund["units"] == pytest.approx(40_000 / 12.3456)

    def test_thai_funds_are_fractional_without_being_told(self):
        plan = allocation.allocate_units(
            {"MF:FUND": 1.0}, pd.Series({"MF:FUND": 12.3456}), 1_000
        )
        assert plan.rows[0]["units"] == pytest.approx(1_000 / 12.3456)
        assert plan.leftover == pytest.approx(0.0)

    def test_a_cash_sleeve_is_held_not_bought(self):
        plan = allocation.allocate_units(
            {"SPY": 0.7, metrics.CASH_SYMBOL: 0.3}, PRICES, 100_000
        )
        cash_row = next(r for r in plan.rows if r["asset"] == metrics.CASH_SYMBOL)
        assert cash_row["units"] is None
        assert cash_row["value"] == pytest.approx(30_000)

    def test_an_asset_priced_above_the_budget_is_reported(self):
        plan = allocation.allocate_units({"SPY": 1.0}, pd.Series({"SPY": 578.32}), 100)
        assert plan.rows[0]["units"] == 0
        assert plan.unaffordable == ["SPY"]

    def test_zero_weights_are_left_out(self):
        plan = allocation.allocate_units({"SPY": 1.0, "GLD": 0.0}, PRICES, 50_000)
        assert [r["asset"] for r in plan.rows] == ["SPY"]

    def test_actual_weights_come_back_for_comparison(self):
        plan = allocation.allocate_units({"SPY": 0.5, "GLD": 0.5}, PRICES, 100_000)
        for row in plan.rows:
            assert row["actual_weight"] == pytest.approx(row["value"] / 100_000, abs=1e-9)
            assert abs(row["actual_weight"] - row["target_weight"]) < 0.02

    def test_a_missing_price_is_skipped_rather_than_crashing(self):
        plan = allocation.allocate_units(
            {"SPY": 0.5, "GHOST": 0.5}, PRICES, 100_000
        )
        assert "GHOST" not in [r["asset"] for r in plan.rows]
        assert plan.unpriced == ["GHOST"]

    def test_no_budget_allocates_nothing(self):
        plan = allocation.allocate_units({"SPY": 1.0}, PRICES, 0)
        assert plan.rows[0]["units"] == 0
        assert plan.leftover == 0


class TestRollingPerformance:
    def _returns(self, days=1500, seed=3):
        rng = np.random.default_rng(seed)
        return pd.Series(
            rng.normal(0.0005, 0.01, days), index=pd.bdate_range("2018-01-01", periods=days)
        )

    def test_columns_and_alignment(self):
        rolling = metrics.rolling_performance(self._returns(), 0.02, window_years=1.0)
        assert list(rolling.columns) == ["annual_return", "annual_volatility", "sharpe"]
        assert rolling.index[-1] == self._returns().index[-1]

    def test_the_first_window_is_dropped(self):
        r = self._returns()
        rolling = metrics.rolling_performance(r, 0.02, window_years=1.0)
        assert len(rolling) < len(r)
        assert rolling.notna().all().all()

    def test_a_steady_series_gives_a_steady_reading(self):
        idx = pd.bdate_range("2018-01-01", periods=800)
        r = pd.Series(0.0004, index=idx)
        rolling = metrics.rolling_performance(r, 0.02, window_years=1.0)
        assert rolling["annual_volatility"].max() == pytest.approx(0.0, abs=1e-9)
        assert rolling["annual_return"].std() < 1e-9

    def test_the_last_window_agrees_with_a_direct_calculation(self):
        r = self._returns()
        rolling = metrics.rolling_performance(r, 0.02, window_years=1.0)
        window = int(round(metrics.periods_per_year(r.index)))
        direct = metrics.backtest_stats(r.iloc[-window:], 0.02)
        assert rolling["annual_return"].iloc[-1] == pytest.approx(direct["annual_return"], rel=0.05)

    def test_too_little_history_gives_an_empty_frame(self):
        rolling = metrics.rolling_performance(self._returns(days=100), 0.02, window_years=1.0)
        assert rolling.empty

    def test_an_empty_series_is_safe(self):
        assert metrics.rolling_performance(pd.Series(dtype=float), 0.02).empty
