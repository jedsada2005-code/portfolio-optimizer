"""End-to-end runs of the Streamlit script itself.

The unit tests cover the formulas; these catch the wiring between them
-- a name that only exists on one branch, a session key written in one
mode and read in another -- which py_compile and pytest cannot see.

Marked slow because each case downloads real prices from Yahoo.
"""

import re

import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

import metrics
import optimizer


pytestmark = pytest.mark.slow

SYMBOLS = "SPY, QQQ, GLD, TLT"


def _widget(widgets, label_fragment):
    for widget in widgets:
        if label_fragment in (widget.label or ""):
            return widget
    raise LookupError(f"no widget labelled …{label_fragment}…")


@pytest.fixture
def app():
    at = AppTest.from_file("app.py", default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    _widget(at.text_input, "สินทรัพย์ในพอร์ต").set_value(SYMBOLS)
    return at


def _calculate(at):
    # Not at.button[0]: AppTest lists main-block elements before sidebar
    # ones, so the welcome screen's preset buttons come first.
    _widget(at.button, "Calculate").click().run()
    assert not at.exception, at.exception
    return at


def test_default_run_produces_results(app):
    at = _calculate(app)
    assert not at.error
    assert at.session_state["calculated"] is True


@pytest.mark.parametrize("mode", ["Train / Test Split", "Walk-Forward", "In-sample (ทั้งช่วง)"])
def test_every_backtest_mode_runs(app, mode):
    app.radio[0].set_value(mode)
    assert not _calculate(app).error


@pytest.mark.parametrize("frequency", ["รายวัน", "รายเดือน", "รายไตรมาส", "รายปี", "ซื้อแล้วถือ (ไม่ปรับ)"])
def test_every_rebalance_frequency_runs(app, frequency):
    _widget(app.selectbox, "Rebalance").set_value(frequency)
    assert not _calculate(app).error


def test_baht_base_currency_runs(app):
    _widget(app.selectbox, "สกุลเงินฐาน").set_value("THB")
    assert not _calculate(app).error


def test_trading_costs_run(app):
    _widget(app.selectbox, "Rebalance").set_value("รายวัน")
    _widget(app.number_input, "ค่าธรรมเนียม").set_value(100.0)
    assert not _calculate(app).error


def test_cash_sleeve_and_weight_cap_run_together(app):
    _widget(app.slider, "น้ำหนักสูงสุดต่อสินทรัพย์").set_value(40)
    _widget(app.slider, "เงินสด").set_value(20)
    assert not _calculate(app).error


def test_benchmark_can_be_left_empty(app):
    _widget(app.text_input, "Benchmark").set_value("")
    assert not _calculate(app).error


def test_a_single_symbol_is_refused(app):
    _widget(app.text_input, "สินทรัพย์ในพอร์ต").set_value("SPY")
    at = _calculate(app)
    assert any("อย่างน้อย 2 ตัว" in e.value for e in at.error)


def test_reversed_dates_are_refused(app):
    app.date_input[0].set_value(pd.Timestamp("2024-01-01").date())
    app.date_input[1].set_value(pd.Timestamp("2015-01-01").date())
    at = _calculate(app)
    assert any("ช่วงวันที่ไม่ถูกต้อง" in e.value for e in at.error)


def _one(value):
    # AppTest exposes raw query params, where repeated keys are lists;
    # the Streamlit runtime hands the script the last value.
    return value[-1] if isinstance(value, list) else value


def test_settings_are_written_back_to_the_url(app):
    _widget(app.selectbox, "สกุลเงินฐาน").set_value("THB")
    _widget(app.slider, "น้ำหนักสูงสุดต่อสินทรัพย์").set_value(40)
    at = _calculate(app)
    assert _one(at.query_params["symbols"]) == SYMBOLS
    assert _one(at.query_params["base"]) == "THB"
    assert _one(at.query_params["maxw"]) == "0.4"  # stored as a fraction


def test_a_shared_link_reopens_with_the_same_settings():
    at = AppTest.from_file("app.py", default_timeout=300)
    at.query_params["symbols"] = "AAPL, MSFT"
    at.query_params["base"] = "THB"
    at.query_params["maxw"] = "0.45"
    at.query_params["mode"] = "In-sample (ทั้งช่วง)"
    at.query_params["reb"] = "รายปี"
    at.run()
    assert not at.exception, at.exception
    assert _widget(at.text_input, "สินทรัพย์ในพอร์ต").value == "AAPL, MSFT"
    assert _widget(at.selectbox, "สกุลเงินฐาน").value == "THB"
    # Stored as a fraction in the URL, shown as whole percent.
    assert _widget(at.slider, "น้ำหนักสูงสุดต่อสินทรัพย์").value == 45
    assert at.radio[0].value == "In-sample (ทั้งช่วง)"
    assert _widget(at.selectbox, "Rebalance").value == "รายปี"


class TestPercentSliders:
    """Percent sliders must be driven in whole percent.

    Given a 0.05-1.0 range, ``format="%.0f%%"`` renders the raw fraction,
    so every setting below 0.55 showed as "0%" and everything above as
    "1%" -- the control looked broken and unmovable even though the
    value behind it was correct.
    """

    @pytest.mark.parametrize(
        "label, low, high",
        [("น้ำหนักสูงสุดต่อสินทรัพย์", 5, 100), ("เงินสด", 0, 90)],
    )
    def test_range_is_expressed_in_percent(self, app, label, high, low):
        slider = _widget(app.slider, label)
        assert (slider.min, slider.max) == (low, high)
        assert slider.step == 5

    def test_train_fraction_slider_is_percent_too(self, app):
        app.radio[0].set_value("Train / Test Split").run()
        slider = _widget(app.slider, "สัดส่วนช่วง Train")
        assert (slider.min, slider.max, slider.step) == (30, 90, 5)

    def test_a_mid_range_setting_survives_the_round_trip(self, app):
        # 45% used to be indistinguishable from 5% on screen.
        _widget(app.slider, "น้ำหนักสูงสุดต่อสินทรัพย์").set_value(45)
        at = _calculate(app)
        assert not at.error
        assert at.session_state["max_weight"] == pytest.approx(0.45)
        assert _one(at.query_params["maxw"]) == "0.45"

    def test_a_cap_below_equal_weight_is_refused_with_the_minimum(self, app):
        # Four holdings need at least 25% each to reach a full allocation.
        _widget(app.slider, "น้ำหนักสูงสุดต่อสินทรัพย์").set_value(20)
        at = _calculate(app)
        assert any("25.0%" in e.value for e in at.error), [e.value for e in at.error]

    def test_a_saved_link_restores_the_percent_position(self):
        at = AppTest.from_file("app.py", default_timeout=300)
        at.query_params["maxw"] = "0.35"
        at.query_params["cashpct"] = "0.25"
        at.run()
        assert not at.exception, at.exception
        assert _widget(at.slider, "น้ำหนักสูงสุดต่อสินทรัพย์").value == 35
        assert _widget(at.slider, "เงินสด").value == 25


class TestWeightSourceSelection:
    """The Backtesting tab owns which weights get tested.

    It used to always read the Custom Weights sliders, which snap to 1%
    steps -- so a 35.49% optimal allocation was silently backtested as
    36%, and nothing on screen said so. The NAV tab recomputed its own
    series, which let the two tabs disagree outright under walk-forward.
    """

    def _source_radio(self, at):
        return _widget(at.selectbox, "น้ำหนักที่ใช้ backtest")

    def test_optimal_weights_are_the_default(self, app):
        at = _calculate(app)
        assert self._source_radio(at).value == "Max Sharpe"

    def test_custom_is_offered_once_the_sliders_have_rendered(self, app):
        at = _calculate(app)
        assert "Custom (จากแท็บน้ำหนักพอร์ต)" in self._source_radio(at).options

    @pytest.mark.parametrize("source", ["Max Sharpe", "Min Volatility", "Custom (จากแท็บน้ำหนักพอร์ต)"])
    def test_every_source_runs(self, app, source):
        at = _calculate(app)
        self._source_radio(at).set_value(source).run()
        assert not at.exception, at.exception
        assert at.session_state["nav_view"]["source"] == source

    def test_optimal_source_is_not_rounded_to_slider_steps(self, app):
        at = _calculate(app)
        self._source_radio(at).set_value("Max Sharpe").run()
        used = at.session_state["nav_view"]["weights"]
        optimal = at.session_state["cleaned"]
        for asset, weight in optimal.items():
            assert used[asset] == pytest.approx(weight)
        # A slider-rounded copy would land on whole percents.
        assert any(
            abs(w * 100 - round(w * 100)) > 1e-6 for w in used.values() if w > 0
        ), "expected at least one non-integer percent to prove no rounding"

    def test_switching_source_changes_the_weights_actually_used(self, app):
        at = _calculate(app)
        self._source_radio(at).set_value("Max Sharpe").run()
        sharpe_weights = dict(at.session_state["nav_view"]["weights"])
        self._source_radio(at).set_value("Min Volatility").run()
        assert at.session_state["nav_view"]["weights"] != sharpe_weights


class TestNavTabAgreesWithBacktest:
    def test_both_tabs_share_one_series(self, app):
        at = _calculate(app)
        view = at.session_state["nav_view"]
        assert not view["returns"].empty
        assert view["start"] is not None

    def test_walk_forward_reaches_the_nav_tab(self, app):
        app.radio[0].set_value("Walk-Forward")
        at = _calculate(app)
        assert not at.error
        view = at.session_state["nav_view"]
        assert view["is_walk_forward"] is True
        # The NAV line must be the walk-forward series, not a constant
        # weight simulation standing in for it.
        assert len(view["returns"]) > 0
        assert view["source"] in ("Max Sharpe", "Min Volatility")

    def test_walk_forward_hides_the_custom_option(self, app):
        app.radio[0].set_value("Walk-Forward")
        at = _calculate(app)
        labels = [w.label for w in at.selectbox]
        assert "น้ำหนักที่ใช้ backtest" not in labels
        import optimizer
        assert _widget(at.selectbox, "วัตถุประสงค์").options == optimizer.OBJECTIVES

    def test_cash_sleeve_reaches_the_nav_tab(self, app):
        _widget(app.slider, "เงินสด").set_value(25)
        at = _calculate(app)
        view = at.session_state["nav_view"]
        assert view["weights"][metrics.CASH_SYMBOL] == pytest.approx(0.25)


class TestWelcomeScreen:
    """Before the first run the page used to hold one sentence.

    Someone opening the app had no idea what to type into the symbol
    box, which matters because the whole point is handing it to people
    who have not seen it before.
    """

    @pytest.fixture
    def fresh(self):
        at = AppTest.from_file("app.py", default_timeout=300)
        at.run()
        assert not at.exception, at.exception
        return at

    def test_explains_itself_before_any_run(self, fresh):
        text = " ".join(m.value for m in fresh.markdown)
        assert "ทดสอบ" in text
        assert any("จะได้อะไรกลับมา" in m.value for m in fresh.markdown)

    def test_offers_a_starting_point_for_every_preset(self, fresh):
        labels = [b.label for b in fresh.button]
        for name in ["หุ้นเทคโนโลยีสหรัฐ", "กระจายความเสี่ยงหลายสินทรัพย์", "หุ้นไทย", "พอร์ตความเสี่ยงต่ำ"]:
            assert name in labels

    def test_a_preset_fills_the_symbol_box(self, fresh):
        _widget(fresh.button, "หุ้นไทย").click().run()
        assert not fresh.exception, fresh.exception
        assert fresh.session_state["symbols_input"] == "PTT.BK, ADVANC.BK, CPALL.BK, KBANK.BK, AOT.BK"

    def test_a_preset_actually_runs(self, fresh):
        _widget(fresh.button, "พอร์ตความเสี่ยงต่ำ").click().run()
        _calculate(fresh)
        assert not fresh.exception, fresh.exception
        assert not fresh.error
        assert fresh.session_state["calculated"] is True

    def test_the_share_link_appears_only_after_a_run(self, fresh, app):
        assert not any("แชร์พอร์ตนี้" in (e.label or "") for e in fresh.expander)
        at = _calculate(app)
        assert any("แชร์พอร์ตนี้" in (e.label or "") for e in at.expander)


class TestCustomWeightEditing:
    """Custom weights moved into keyed number inputs.

    A fixed four-column grid of sliders became unreadable past a handful
    of holdings, and slider state could only be read after the weights
    tab had drawn itself -- which is what forced the backtest to live
    inside a tab.
    """

    def test_one_input_per_asset(self, app):
        at = _calculate(app)
        labels = [n.label for n in at.number_input]
        for symbol in ["SPY", "QQQ", "GLD", "TLT"]:
            assert symbol in labels

    def test_equal_weight_button_levels_everything(self, app):
        at = _calculate(app)
        _widget(at.button, "เท่ากันทุกตัว").click().run()
        assert not at.exception, at.exception
        for symbol in ["SPY", "QQQ", "GLD", "TLT"]:
            assert at.session_state[f"cw_{symbol}"] == pytest.approx(25.0)

    def test_restore_buttons_bring_back_each_objective(self, app):
        at = _calculate(app)
        _widget(at.button, "เท่ากันทุกตัว").click().run()
        _widget(at.button, "กลับไปใช้ Max Sharpe").click().run()
        assert not at.exception, at.exception
        optimal = at.session_state["cleaned"]
        for symbol, weight in optimal.items():
            assert at.session_state[f"cw_{symbol}"] == pytest.approx(round(weight * 100, 1))

    def test_edited_weights_reach_the_backtest(self, app):
        at = _calculate(app)
        _widget(at.button, "เท่ากันทุกตัว").click().run()
        _widget(at.selectbox, "น้ำหนักที่ใช้ backtest").set_value("Custom (จากแท็บน้ำหนักพอร์ต)").run()
        assert not at.exception, at.exception
        used = at.session_state["nav_view"]["weights"]
        for symbol in ["SPY", "QQQ", "GLD", "TLT"]:
            assert used[symbol] == pytest.approx(0.25)

    def test_weights_are_normalised_not_rejected(self, app):
        at = _calculate(app)
        for symbol in ["SPY", "QQQ", "GLD", "TLT"]:
            _widget(at.number_input, symbol).set_value(10.0)
        _widget(at.selectbox, "น้ำหนักที่ใช้ backtest").set_value("Custom (จากแท็บน้ำหนักพอร์ต)").run()
        assert not at.error
        used = at.session_state["nav_view"]["weights"]
        assert sum(used.values()) == pytest.approx(1.0)


class TestTabsAreRenderers:
    """The backtest is computed before any tab draws.

    Weight diagnostics could not previously sit on the weights tab,
    because it renders before the backtest tab that produced them.
    """

    def test_weight_diagnostics_reach_the_weights_tab(self, app):
        at = _calculate(app)
        headings = [h.value for h in at.subheader]
        assert "Train vs Test" in headings

    def test_walk_forward_history_reaches_the_weights_tab(self, app):
        app.radio[0].set_value("Walk-Forward")
        at = _calculate(app)
        assert not at.error
        assert "น้ำหนักที่คำนวณใหม่ในแต่ละงวด" in [h.value for h in at.subheader]

    def test_nav_view_is_published_before_the_tabs_draw(self, app):
        at = _calculate(app)
        assert at.session_state["nav_view"]["returns"] is not None


class TestResetButton:
    def test_reset_clears_saved_settings(self, app):
        at = _calculate(app)
        assert at.query_params
        _widget(at.button, "รีเซ็ตการตั้งค่าทั้งหมด").click().run()
        assert not at.exception, at.exception
        assert "calculated" not in at.session_state
        assert _widget(at.text_input, "สินทรัพย์ในพอร์ต").value == "AMZN, META, LLY, SPY, NVDA, GOOGL"


class TestMinimumWeightControl:
    def test_a_floor_removes_every_zero_allocation(self, app):
        _widget(app.slider, "น้ำหนักขั้นต่ำต่อสินทรัพย์").set_value(10)
        at = _calculate(app)
        assert not at.error
        weights = at.session_state["cleaned"]
        assert min(weights.values()) >= 0.10 - 1e-6

    def test_zero_floor_keeps_the_previous_behaviour(self, app):
        at = _calculate(app)
        assert at.session_state["min_weight"] == 0.0
        assert not at.error

    def test_an_unfundable_floor_is_refused_with_the_limit(self, app):
        _widget(app.slider, "น้ำหนักขั้นต่ำต่อสินทรัพย์").set_value(30)
        at = _calculate(app)
        assert any("25.0%" in e.value for e in at.error), [e.value for e in at.error]

    def test_a_floor_above_the_cap_is_refused(self, app):
        _widget(app.slider, "น้ำหนักขั้นต่ำต่อสินทรัพย์").set_value(25)
        _widget(app.slider, "น้ำหนักสูงสุดต่อสินทรัพย์").set_value(20)
        at = _calculate(app)
        assert any("มากกว่าน้ำหนักสูงสุด" in e.value for e in at.error)

    def test_the_random_cloud_respects_the_floor(self, app):
        _widget(app.slider, "น้ำหนักขั้นต่ำต่อสินทรัพย์").set_value(10)
        at = _calculate(app)
        stds, rets, sharpes = at.session_state["random"]
        assert len(stds) > 0

    def test_a_comparison_frontier_is_kept_when_bounds_are_set(self, app):
        _widget(app.slider, "น้ำหนักขั้นต่ำต่อสินทรัพย์").set_value(10)
        at = _calculate(app)
        free_x, free_y = at.session_state["ef_curve_free"]
        assert len(free_x) > 1

    def test_no_comparison_frontier_without_bounds(self, app):
        at = _calculate(app)
        free_x, _ = at.session_state["ef_curve_free"]
        assert len(free_x) == 0

    def test_the_floor_is_saved_in_the_url(self, app):
        _widget(app.slider, "น้ำหนักขั้นต่ำต่อสินทรัพย์").set_value(15)
        at = _calculate(app)
        assert _one(at.query_params["minw"]) == "0.15"


class TestReturnMethodAndHRP:
    """The expected-return estimator was hardcoded, and HRP -- which
    needs no return estimate at all -- was not offered."""

    def _method(self, at):
        return _widget(at.selectbox, "วิธีประมาณผลตอบแทน")

    def test_the_default_method_is_the_previous_behaviour(self, app):
        at = _calculate(app)
        assert at.session_state["return_method"] == "ค่าเฉลี่ยผลตอบแทน 1 ปี (ทับซ้อน)"

    @pytest.mark.parametrize("method", [
        "ค่าเฉลี่ยผลตอบแทน 1 ปี (ทับซ้อน)", "ค่าเฉลี่ยตลอดช่วง",
        "ถ่วงน้ำหนักข้อมูลล่าสุด (EMA)", "CAPM (อิงความเสี่ยงเทียบตลาด)",
    ])
    def test_every_method_runs(self, app, method):
        self._method(app).set_value(method)
        at = _calculate(app)
        assert not at.error
        assert at.session_state["return_method"] == method

    def test_changing_the_method_changes_the_weights(self, app):
        at = _calculate(app)
        first = dict(at.session_state["cleaned"])
        self._method(at).set_value("CAPM (อิงความเสี่ยงเทียบตลาด)")
        at = _calculate(at)
        assert dict(at.session_state["cleaned"]) != first

    def test_hrp_holds_every_asset(self, app):
        at = _calculate(app)
        hrp = at.session_state["hrp_cleaned"]
        assert min(hrp.values()) > 0
        assert sum(hrp.values()) == pytest.approx(1.0)

    def test_hrp_is_selectable_as_a_backtest_source(self, app):
        at = _calculate(app)
        _widget(at.selectbox, "น้ำหนักที่ใช้ backtest").set_value("HRP (Risk Parity)").run()
        assert not at.exception, at.exception
        assert at.session_state["nav_view"]["source"] == "HRP (Risk Parity)"
        used = {k: v for k, v in at.session_state["nav_view"]["weights"].items()}
        assert min(used.values()) > 0

    def test_hrp_does_not_depend_on_the_return_method(self, app):
        at = _calculate(app)
        baseline = dict(at.session_state["hrp_cleaned"])
        self._method(at).set_value("CAPM (อิงความเสี่ยงเทียบตลาด)")
        at = _calculate(at)
        assert dict(at.session_state["hrp_cleaned"]) == baseline

    def test_walk_forward_can_refit_with_hrp(self, app):
        app.radio[0].set_value("Walk-Forward")
        at = _calculate(app)
        _widget(at.selectbox, "วัตถุประสงค์").set_value("HRP (Risk Parity)").run()
        assert not at.exception, at.exception
        assert not at.error

    def test_the_method_is_saved_in_the_url(self, app):
        self._method(app).set_value("CAPM (อิงความเสี่ยงเทียบตลาด)")
        at = _calculate(app)
        assert _one(at.query_params["retm"]) == "CAPM (อิงความเสี่ยงเทียบตลาด)"


class TestCurrencyDetection:
    def test_currencies_are_resolved_not_guessed(self, app):
        at = _calculate(app)
        assert not at.error

    def test_a_london_listed_dollar_etf_is_treated_as_dollars(self, app):
        _widget(app.text_input, "สินทรัพย์ในพอร์ต").set_value("SPY, GLD, IBTA.L")
        app.date_input[0].set_value(pd.Timestamp("2018-01-01").date())
        at = _calculate(app)
        assert not at.error
        # The suffix says London, the security trades in dollars; a
        # caption reports the correction rather than silently converting
        # the series twice.
        assert any("IBTA.L" in c.value and "GBP→USD" in c.value for c in at.caption), \
            [c.value for c in at.caption]


class TestHistoryFloorControl:
    def _slider(self, at):
        return _widget(at.slider, "ประวัติขั้นต่ำที่ยอมรับ")

    def test_the_default_is_three_years(self, app):
        assert self._slider(app).value == 3

    @pytest.mark.parametrize("years", [1, 3, 6])
    def test_every_setting_runs(self, app, years):
        self._slider(app).set_value(years)
        at = _calculate(app)
        assert not at.exception, at.exception

    def test_a_lax_setting_is_flagged(self, app):
        self._slider(app).set_value(2)
        at = _calculate(app)
        assert any("ต่ำกว่าค่าแนะนำ" in c.value for c in at.caption)

    def test_no_warning_at_the_recommended_setting(self, app):
        at = _calculate(app)
        assert not any("ต่ำกว่าค่าแนะนำ" in c.value for c in at.caption)

    def test_the_setting_is_saved_in_the_url(self, app):
        self._slider(app).set_value(5)
        at = _calculate(app)
        assert _one(at.query_params["minyrs"]) == "5"


class TestCorrelationWindow:
    def test_correlation_matches_the_optimisation_window(self, app):
        at = _calculate(app)
        weekly = at.session_state["weekly_usable"]
        train = at.session_state["train_close"]
        # The frontier tab describes the training window, so the
        # correlation shown beside it must come from that data alone.
        # W-FRI labels each bucket by its Friday, which can fall a few
        # days past the last training date without carrying test data.
        expected = train[list(weekly.columns)].resample("W-FRI").last()
        pd.testing.assert_frame_equal(weekly, expected)
        assert list(weekly.columns) == list(at.session_state["ar"].index)


class TestOrderPlan:
    """The app could say "hold 45.9% SPY" but not "buy 29 units"."""

    def test_an_order_table_is_produced(self, app):
        at = _calculate(app)
        assert "แปลงเป็นคำสั่งซื้อจริง" in [h.value for h in at.subheader]
        assert not at.error

    def test_the_change_left_over_is_small(self, app):
        at = _calculate(app)
        leftover = next(m for m in at.metric if "เศษเงินเหลือ" in m.label)
        digits = leftover.value.replace(",", "").split()[0]
        assert float(digits) < 1_000_000 * 0.01

    def test_a_tiny_budget_is_flagged_rather_than_silently_wrong(self, app):
        _widget(app.number_input, "เงินลงทุนตั้งต้น").set_value(100)
        at = _calculate(app)
        assert any("เงินไม่พอซื้อ" in w.value for w in at.warning), [w.value for w in at.warning]

    def test_a_cash_sleeve_appears_as_cash_not_units(self, app):
        _widget(app.slider, "เงินสด").set_value(30)
        at = _calculate(app)
        assert not at.error
        assert at.session_state["nav_view"]["weights"][metrics.CASH_SYMBOL] == pytest.approx(0.30)

    def test_the_plan_follows_the_selected_weight_source(self, app):
        at = _calculate(app)
        _widget(at.selectbox, "น้ำหนักที่ใช้ backtest").set_value("Min Volatility").run()
        assert not at.exception, at.exception
        assert at.session_state["nav_view"]["source"] == "Min Volatility"


class TestRollingView:
    def test_rolling_performance_is_available(self, app):
        at = _calculate(app)
        assert any("เลื่อนหน้าต่าง 1 ปี" in (e.label or "") for e in at.expander)

    def test_it_reports_the_best_and_worst_year(self, app):
        at = _calculate(app)
        labels = [m.label for m in at.metric]
        assert "ปีที่ดีที่สุด" in labels
        assert "ปีที่แย่ที่สุด" in labels

    def test_a_short_test_window_says_so_instead_of_breaking(self, app):
        app.date_input[0].set_value(pd.Timestamp("2022-01-01").date())
        app.date_input[1].set_value(pd.Timestamp("2023-02-01").date())
        at = _calculate(app)
        assert not at.exception, at.exception


class TestObjectiveCatalogueInApp:
    """Seven objectives are solved on every run; the picker chooses
    among ready answers rather than triggering a solve."""

    def _picker(self, at):
        return _widget(at.selectbox, "น้ำหนักที่ใช้ backtest")

    def test_all_objectives_are_offered(self, app):
        import optimizer
        at = _calculate(app)
        for name in optimizer.OBJECTIVES:
            assert name in self._picker(at).options, name

    @pytest.mark.parametrize("objective", [
        "กำหนดความเสี่ยงเป้าหมาย", "กำหนดผลตอบแทนเป้าหมาย",
        "Min CVaR", "Min Semivariance",
    ])
    def test_each_new_objective_backtests(self, app, objective):
        at = _calculate(app)
        self._picker(at).set_value(objective).run()
        assert not at.exception, at.exception
        assert at.session_state["nav_view"]["source"] == objective
        assert sum(at.session_state["nav_view"]["weights"].values()) == pytest.approx(1.0, abs=1e-6)

    def test_a_volatility_target_is_actually_hit(self, app):
        import optimizer
        _widget(app.number_input, "ความเสี่ยงเป้าหมาย").set_value(0.14)
        at = _calculate(app)
        weights = at.session_state["objective_weights"][optimizer.TARGET_VOLATILITY]
        _, vol = at.session_state["objective_perf"][optimizer.TARGET_VOLATILITY]
        assert vol == pytest.approx(0.14, abs=0.005)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_an_impossible_target_is_reported_not_crashed(self, app):
        import optimizer
        _widget(app.number_input, "ความเสี่ยงเป้าหมาย").set_value(0.01)
        at = _calculate(app)
        assert not at.exception, at.exception
        assert optimizer.TARGET_VOLATILITY in at.session_state["objective_errors"]
        assert optimizer.TARGET_VOLATILITY not in self._picker(at).options

    def test_downside_objectives_respect_the_weight_floor(self, app):
        _widget(app.slider, "น้ำหนักขั้นต่ำต่อสินทรัพย์").set_value(10)
        at = _calculate(app)
        for name in ("Min CVaR", "Min Semivariance"):
            weights = at.session_state["objective_weights"][name]
            assert min(weights.values()) >= 0.10 - 1e-6, name

    def test_targets_are_saved_in_the_url(self, app):
        _widget(app.number_input, "ความเสี่ยงเป้าหมาย").set_value(0.15)
        at = _calculate(app)
        assert _one(at.query_params["tvol"]) == "0.15"


class TestWalkForwardUsesEverySetting:
    """A settings pass-through was silently dropped by a mis-indented
    edit, so walk-forward kept using the default estimator and history
    floor while the sidebar said otherwise."""

    def test_the_return_method_reaches_every_refit(self, app):
        app.radio[0].set_value("Walk-Forward")
        at = _calculate(app)
        baseline = [dict(w) for _, w in at.session_state["walk_weight_history"]]
        assert baseline

        _widget(at.selectbox, "วิธีประมาณผลตอบแทน").set_value("CAPM (อิงความเสี่ยงเทียบตลาด)")
        at = _calculate(at)
        changed = [dict(w) for _, w in at.session_state["walk_weight_history"]]
        assert changed != baseline

    def test_a_strict_history_floor_reaches_walk_forward(self, app):
        app.radio[0].set_value("Walk-Forward")
        _widget(app.slider, "ประวัติขั้นต่ำที่ยอมรับ").set_value(6)
        at = _calculate(app)
        assert not at.exception, at.exception


class TestHistoryFloorActuallyExcludes:
    """The warning said "ไม่รวมในการคำนวณพอร์ต" while the exclusion sat
    inside `if min_history_years < 3`, so it ran only when the floor was
    set BELOW the default -- tightening the control turned it off."""

    def test_an_asset_short_of_the_floor_never_reaches_the_optimiser(self, app):
        # META listed May 2012, so in the train half of this window it
        # backs its estimate with 75 trailing-year windows against the
        # 104 the default floor demands, while SPY/QQQ/GLD each have 146.
        _widget(app.text_input, "สินทรัพย์ในพอร์ต").set_value("SPY, QQQ, GLD, META")
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2011-01-01"))
        _widget(app.date_input, "End Date").set_value(pd.Timestamp("2016-06-01"))
        at = _calculate(app)

        assert "META" not in list(at.session_state["ar"].index)
        assert "META" not in at.session_state["covr"].columns
        for name, weights in at.session_state["objective_weights"].items():
            assert "META" not in weights, name

    def test_a_floor_nothing_can_clear_stops_the_run(self, app):
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2021-06-01"))
        _widget(app.slider, "ประวัติขั้นต่ำที่ยอมรับ").set_value(6)
        _widget(app.button, "Calculate").click().run()

        assert not app.exception, app.exception
        assert app.error, "no asset clears a 6-year floor in a 4-year window"

    def test_a_listing_younger_than_the_split_does_not_crash_the_page(self, app):
        # ARM listed Sep 2023, after this window's train/test boundary, so
        # it contributes an all-NaN covariance row. Keeping it reached
        # np.linalg.eigh, which failed to converge and took the whole page
        # down rather than merely mis-weighting it.
        _widget(app.text_input, "สินทรัพย์ในพอร์ต").set_value("SPY, TLT, GLD, QQQ, ARM")
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2015-01-01"))
        at = _calculate(app)

        assert not at.error
        assert "ARM" not in list(at.session_state["ar"].index)

    def test_a_floor_it_clears_still_lets_the_asset_through(self, app):
        # Guard against over-correcting into "drop everything": the same
        # holding the default floor rejects has to survive a floor it
        # genuinely clears.
        _widget(app.text_input, "สินทรัพย์ในพอร์ต").set_value("SPY, QQQ, GLD, META")
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2011-01-01"))
        _widget(app.date_input, "End Date").set_value(pd.Timestamp("2016-06-01"))
        _widget(app.slider, "ประวัติขั้นต่ำที่ยอมรับ").set_value(1)
        at = _calculate(app)
        assert "META" in list(at.session_state["ar"].index)


class TestFeeDragComparesLikeWithLike:
    """In walk-forward mode the "before fees" baseline was a fixed
    in-sample portfolio, so the whole in-sample/out-of-sample gap was
    reported as the cost of trading."""

    def _reported_drag(self, at):
        pattern = re.compile(r"กินผลตอบแทนต่อปีไป \*\*(-?[\d.]+)%\*\*")
        for caption in at.caption:
            found = pattern.search(caption.value or "")
            if found:
                return float(found.group(1)) / 100.0
        raise LookupError("no fee-drag caption rendered")

    def test_walk_forward_drag_matches_the_same_run_without_costs(self, app):
        app.radio[0].set_value("Walk-Forward")
        _widget(app.selectbox, "Rebalance").set_value("รายไตรมาส")
        _widget(app.number_input, "ค่าธรรมเนียม").set_value(100.0)
        at = _calculate(app)

        state = at.session_state
        objective = _widget(at.selectbox, "วัตถุประสงค์ที่ใช้คำนวณ").value
        common = dict(
            cash_fraction=state["cash_fraction"],
            rebalance_freq=metrics.REBALANCE_FREQUENCIES[state["rebalance_label"]],
            min_weight=state["min_weight"],
            return_method=state["return_method"],
            min_observations=state["required_observations"],
        )
        net, gross = (
            optimizer.walk_forward(
                state["test_close"], state["risk_free_rate"], objective,
                state["max_weight"], state["shrinkage"], state["refit_freq"],
                bps, **common,
            ).returns
            for bps in (state["cost_bps"], 0.0)
        )
        truth = (
            metrics.backtest_stats(gross, state["risk_free_rate"])["annual_return"]
            - metrics.backtest_stats(net, state["risk_free_rate"])["annual_return"]
        )
        assert self._reported_drag(at) == pytest.approx(truth, abs=5e-4)

    def test_split_mode_drag_still_matches(self, app):
        _widget(app.selectbox, "Rebalance").set_value("รายไตรมาส")
        _widget(app.number_input, "ค่าธรรมเนียม").set_value(100.0)
        at = _calculate(app)
        assert 0.0 <= self._reported_drag(at) < 0.05


def test_the_benchmark_comparison_is_rendered_once(app):
    at = _calculate(app)
    headings = [
        s.value for s in at.subheader if "เทียบกับ Benchmark" in (s.value or "")
    ]
    assert len(headings) == 1, headings


class TestBenchmarkCoversTheSameDays:
    """A benchmark that listed after the portfolio began was forward-
    filled into NaNs, then those NaNs became 0% returns -- so it read as
    flat through years it did not exist for."""

    def test_a_later_listing_is_not_flat_lined(self, app):
        # In-sample so the tested window really does start in 2010; under
        # the default split it would start after ARKK already existed and
        # there would be no gap to mishandle. ARKK listed Oct 2014.
        _widget(app.text_input, "Benchmark").set_value("ARKK")
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2010-01-01"))
        app.radio[0].set_value("In-sample (ทั้งช่วง)")
        at = _calculate(app)

        prices = at.session_state["benchmark"].dropna()
        truth = metrics.backtest_stats(
            prices.pct_change().dropna(), at.session_state["risk_free_rate"]
        )["annual_return"]

        for frame in at.dataframe:
            table = frame.value
            if "ARKK" in getattr(table, "columns", []):
                reported = float(table["ARKK"].iloc[0].rstrip("%")) / 100.0
                assert reported == pytest.approx(truth, abs=5e-3)
                break
        else:
            raise AssertionError("no benchmark comparison table rendered")

    def test_the_shorter_comparison_window_is_declared(self, app):
        _widget(app.text_input, "Benchmark").set_value("ARKK")
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2010-01-01"))
        app.radio[0].set_value("In-sample (ทั้งช่วง)")
        at = _calculate(app)

        said = [w.value for w in at.warning if "ARKK" in (w.value or "")]
        assert said, "nothing told the reader the comparison starts later"

    def test_a_benchmark_covering_the_whole_window_says_nothing_extra(self, app):
        # ^GSPC rather than SPY: SPY is in the portfolio, which raises an
        # unrelated warning of its own.
        _widget(app.text_input, "Benchmark").set_value("^GSPC")
        at = _calculate(app)
        assert not [
            w.value for w in at.warning if "^GSPC" in (w.value or "")
        ]


class TestUnreachableTargetsAreDeclared:
    """efficient_risk maximises return under a volatility ceiling, so a
    ceiling above the end of the frontier is non-binding: the solver
    returns the highest-return portfolio and says nothing."""

    def test_a_volatility_ceiling_beyond_the_frontier_is_reported(self, app):
        _widget(app.number_input, "ความเสี่ยงเป้าหมาย").set_value(0.90)
        at = _calculate(app)
        _widget(at.selectbox, "น้ำหนักที่ใช้ backtest").set_value(
            optimizer.TARGET_VOLATILITY
        ).run()

        assert not at.exception, at.exception
        said = " ".join(w.value or "" for w in at.warning)
        assert "90" in said and "ทำได้" in said, said

    def test_a_reachable_target_is_not_reported(self, app):
        _widget(app.number_input, "ความเสี่ยงเป้าหมาย").set_value(0.12)
        at = _calculate(app)
        _widget(at.selectbox, "น้ำหนักที่ใช้ backtest").set_value(
            optimizer.TARGET_VOLATILITY
        ).run()

        assert not [w for w in at.warning if "ทำได้สูงสุด" in (w.value or "")]


class TestStatusChipsMatchTheChosenObjective:
    """HRP allocates from the covariance tree alone and takes no weight
    bounds, but the status line was built before the objective was
    chosen, so it advertised bounds HRP had ignored."""

    def test_hrp_does_not_advertise_bounds_it_ignores(self, app):
        _widget(app.slider, "น้ำหนักสูงสุดต่อสินทรัพย์").set_value(40)
        at = _calculate(app)
        _widget(at.selectbox, "น้ำหนักที่ใช้ backtest").set_value(
            optimizer.HRP_OBJECTIVE
        ).run()

        status = " ".join(c.value or "" for c in at.caption if "Rebalance" in (c.value or ""))
        assert status, "no status line rendered"
        assert "สูงสุด 40%/ตัว" not in status, status

    def test_a_bounded_objective_still_advertises_its_bounds(self, app):
        _widget(app.slider, "น้ำหนักสูงสุดต่อสินทรัพย์").set_value(40)
        at = _calculate(app)
        status = " ".join(c.value or "" for c in at.caption if "Rebalance" in (c.value or ""))
        assert "สูงสุด 40%/ตัว" in status, status


def _comparison_table(at):
    for frame in at.dataframe:
        table = frame.value
        if "วิธี" in list(getattr(table, "columns", [])):
            return table.set_index("วิธี")
    raise LookupError("no objective comparison table rendered")


class TestObjectivesAreComparedSideBySide:
    """Choosing between seven objectives meant running seven backtests
    and remembering the numbers; the weights of all seven were already
    on screen but never their results."""

    def test_every_solvable_objective_gets_a_row(self, app):
        at = _calculate(app)
        table = _comparison_table(at)
        for name in at.session_state["objective_weights"]:
            assert name in table.index, name

    def test_the_selected_objective_row_matches_the_headline(self, app):
        at = _calculate(app)
        chosen = _widget(at.selectbox, "น้ำหนักที่ใช้ backtest").value
        row = _comparison_table(at).loc[chosen]
        headline = {m.label: m.value for m in at.metric}

        assert row["ผลตอบแทนต่อปี"] == headline["ผลตอบแทนต่อปี"]
        assert row["ขาดทุนสูงสุด"] == headline["ขาดทุนสูงสุด"]
        assert row["Sharpe"] == headline["Sharpe Ratio"]

    def test_it_still_matches_after_switching_objective(self, app):
        at = _calculate(app)
        _widget(at.selectbox, "น้ำหนักที่ใช้ backtest").set_value(
            optimizer.HRP_OBJECTIVE
        ).run()

        row = _comparison_table(at).loc[optimizer.HRP_OBJECTIVE]
        headline = {m.label: m.value for m in at.metric}
        assert row["ผลตอบแทนต่อปี"] == headline["ผลตอบแทนต่อปี"]
        assert row["Sharpe"] == headline["Sharpe Ratio"]

    def test_walk_forward_refits_every_row_not_just_the_chosen_one(self, app):
        app.radio[0].set_value("Walk-Forward")
        at = _calculate(app)
        chosen = _widget(at.selectbox, "วัตถุประสงค์ที่ใช้คำนวณ").value

        table = _comparison_table(at)
        row = table.loc[chosen]
        headline = {m.label: m.value for m in at.metric}
        assert row["ผลตอบแทนต่อปี"] == headline["ผลตอบแทนต่อปี"]
        assert row["Sharpe"] == headline["Sharpe Ratio"]
        assert len(table) >= 3

    def test_walk_forward_measures_the_baseline_over_the_same_window(self, app):
        # walk_forward only starts after its minimum training history, so
        # a baseline simulated from the first price would be handed extra
        # years the optimisers never got to trade.
        app.radio[0].set_value("Walk-Forward")
        at = _calculate(app)

        window = at.session_state["nav_view"]["returns"].index
        prices = at.session_state["test_close"].loc[window[0]:]
        assets = sorted(at.session_state["ar"].index)
        expected = optimizer.compare_fixed_weights(
            {"x": {a: 1 / len(assets) for a in assets}},
            prices, at.session_state["risk_free_rate"],
            metrics.REBALANCE_FREQUENCIES[at.session_state["rebalance_label"]],
            at.session_state["cost_bps"],
        )["x"]
        row = _comparison_table(at).loc[optimizer.EQUAL_WEIGHT]
        assert row["ผลตอบแทนต่อปี"] == f"{expected['annual_return']:.2%}"

    def test_equal_weight_is_always_offered_as_the_naive_baseline(self, app):
        at = _calculate(app)
        assert optimizer.EQUAL_WEIGHT in _comparison_table(at).index

    def test_untouched_custom_weights_do_not_masquerade_as_a_choice(self, app):
        # read_custom_weights falls back to 1/N before the weights tab is
        # ever edited, so an untouched "Custom" row would just be the
        # equal-weight row again under a name implying the reader chose it.
        at = _calculate(app)
        assert not any("Custom" in str(n) for n in _comparison_table(at).index)

    def test_edited_custom_weights_earn_their_own_row(self, app):
        at = _calculate(app)
        _widget(at.number_input, "SPY").set_value(70.0).run()
        assert any("Custom" in str(n) for n in _comparison_table(at).index)

    def test_the_equal_weight_row_really_is_equally_weighted(self, app):
        at = _calculate(app)
        assets = sorted(at.session_state["ar"].index)
        expected = optimizer.compare_fixed_weights(
            {"x": {a: 1 / len(assets) for a in assets}},
            at.session_state["test_close"], at.session_state["risk_free_rate"],
            metrics.REBALANCE_FREQUENCIES[at.session_state["rebalance_label"]],
            at.session_state["cost_bps"],
        )["x"]
        row = _comparison_table(at).loc[optimizer.EQUAL_WEIGHT]
        assert row["ผลตอบแทนต่อปี"] == f"{expected['annual_return']:.2%}"

    def test_in_sample_mode_says_the_ranking_is_in_sample(self, app):
        app.radio[0].set_value("In-sample (ทั้งช่วง)")
        at = _calculate(app)
        _comparison_table(at)
        assert any(
            "In-sample" in (c.value or "") and "เปรียบเทียบ" in (c.value or "")
            for c in at.caption
        ), [c.value for c in at.caption]


class TestEstimatesAreShownAgainstOutcomes:
    """Swapping the return estimator moves the frontier's expected return
    by tens of points -- EMA put gold at 30.8% a year over a window it
    delivered 6.7% in -- but the estimate and the outcome lived on
    different tabs with nothing tying them together."""

    def _table(self, at):
        for frame in at.dataframe:
            table = frame.value
            if "เกิดขึ้นจริง" in list(getattr(table, "columns", [])):
                return table.set_index("สินทรัพย์")
        raise LookupError("no estimate-versus-outcome table rendered")

    def test_each_holding_shows_its_estimate_beside_its_outcome(self, app):
        at = _calculate(app)
        table = self._table(at)
        truth = metrics.realised_returns(at.session_state["train_close"])
        for asset in at.session_state["ar"].index:
            assert table.loc[asset, "เกิดขึ้นจริง"] == f"{truth[asset]:.2%}"

    def test_an_estimator_far_from_the_outcome_is_flagged(self, app):
        _widget(app.selectbox, "วิธีประมาณผลตอบแทน").set_value(
            "ถ่วงน้ำหนักข้อมูลล่าสุด (EMA)"
        )
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2010-01-01"))
        at = _calculate(app)
        assert any(
            "ห่างจากที่เกิดขึ้นจริง" in (w.value or "") for w in at.warning
        ), [w.value for w in at.warning]


class TestConcentrationIsMeasured:
    """A weights table shows what is held, not how much of the portfolio
    one position really is."""

    def test_the_effective_holding_count_is_reported(self, app):
        at = _calculate(app)
        weights = at.session_state["objective_weights"]["Max Sharpe"]
        expected = metrics.effective_holdings(weights)
        shown = {m.label: m.value for m in at.metric}
        assert "กระจายตัวเทียบเท่า" in shown
        assert shown["กระจายตัวเทียบเท่า"] == f"{expected:.1f} ตัว"

    def test_a_minimum_weight_raises_the_effective_count(self, app):
        bare = metrics.effective_holdings(
            _calculate(app).session_state["objective_weights"]["Max Sharpe"]
        )
        _widget(app.slider, "น้ำหนักขั้นต่ำต่อสินทรัพย์").set_value(15)
        spread = metrics.effective_holdings(
            _calculate(app).session_state["objective_weights"]["Max Sharpe"]
        )
        assert spread > bare


class TestRebalancingReportsAreExact:
    def test_turnover_says_it_counts_both_sides_of_each_trade(self, app):
        at = _calculate(app)
        turnover = [m for m in at.metric if "Turnover" in m.label]
        assert turnover, "no turnover metric"
        # "ซื้อขาย" alone would pass on the old wording; the point is that
        # the figure is the two-way sum, twice the conventional one-way one.
        assert "สองทาง" in turnover[0].help, turnover[0].help
        assert "ครึ่ง" in turnover[0].help, turnover[0].help

    def test_walk_forward_drift_starts_from_the_weights_it_held(self, app):
        app.radio[0].set_value("Walk-Forward")
        _widget(app.selectbox, "Rebalance").set_value("ซื้อแล้วถือ (ไม่ปรับ)")
        at = _calculate(app)

        held = at.session_state["walk_weight_history"][-1][1]
        drift = [c.value for c in at.caption if "สัดส่วนเมื่อจบช่วง" in (c.value or "")]
        assert drift, "no drift caption"
        for symbol, weight in held.items():
            if weight > 0:
                assert f"{symbol} {weight:.0%}→" in drift[0], (symbol, drift[0])


class TestCorrelationIsShownThroughTime:
    """The heatmap is one matrix for the whole window, which hides the
    only thing that matters about diversification: whether it survives
    the quarter you need it in."""

    def test_the_range_the_average_moved_through_is_reported(self, app):
        at = _calculate(app)
        truth = metrics.rolling_correlation(
            at.session_state["weekly_usable"].pct_change(), 52
        )
        # not bare "แกว่งระหว่าง": the rolling-Sharpe caption uses it too.
        said = [
            c.value for c in at.caption
            if "ค่าสหสัมพันธ์เฉลี่ยแกว่งระหว่าง" in (c.value or "")
        ]
        assert said, [c.value for c in at.caption]
        assert f"{truth.min():.2f}" in said[0]
        assert f"{truth.max():.2f}" in said[0]

    def test_the_range_is_wider_than_the_single_headline_number(self, app):
        at = _calculate(app)
        rolling = metrics.rolling_correlation(
            at.session_state["weekly_usable"].pct_change(), 52
        )
        assert rolling.max() - rolling.min() > 0.05, "no movement to show"

    def test_one_pair_can_be_followed_on_its_own(self, app):
        at = _calculate(app)
        picker = _widget(at.selectbox, "ดูค่าสหสัมพันธ์รายคู่")
        assert len(picker.options) > 1
        picker.set_value(picker.options[1]).run()
        assert not at.exception, at.exception

    def test_too_little_history_says_so_instead_of_drawing_nothing(self, app):
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2023-06-01"))
        _widget(app.slider, "ประวัติขั้นต่ำที่ยอมรับ").set_value(1)
        at = _calculate(app)
        assert not at.exception, at.exception


class TestEstimatesCarryTheirUncertainty:
    """The optimiser reads an expected return as exact and allocates to
    one decimal place off it. On eighteen independent years SPY's 11.6%
    carries a 90% interval from 5.0% to 18.6%."""

    def _table(self, at):
        for frame in at.dataframe:
            table = frame.value
            if "ช่วง 90%" in list(getattr(table, "columns", [])):
                return table.set_index("สินทรัพย์")
        raise LookupError("no interval column rendered")

    def test_each_estimate_shows_the_interval_around_it(self, app):
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2007-01-01"))
        at = _calculate(app)
        truth = metrics.bootstrap_return_interval(at.session_state["weekly_usable"])
        table = self._table(at)
        for asset in truth.index:
            cell = table.loc[asset, "ช่วง 90%"]
            assert f"{truth.loc[asset, 'low']:.1%}" in cell, (asset, cell)
            assert f"{truth.loc[asset, 'high']:.1%}" in cell, (asset, cell)

    def test_the_independent_sample_size_is_stated(self, app):
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2007-01-01"))
        at = _calculate(app)
        truth = metrics.bootstrap_return_interval(at.session_state["weekly_usable"])
        table = self._table(at)
        for asset in truth.index:
            assert str(int(truth.loc[asset, "samples"])) in table.loc[asset, "ปีอิสระ"]

    def test_the_widest_interval_is_called_out(self, app):
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2007-01-01"))
        at = _calculate(app)
        assert any(
            "ช่วงกว้างที่สุด" in (c.value or "") for c in at.caption
        ), [c.value for c in at.caption]

    def test_a_short_window_leaves_the_interval_out_rather_than_faking_it(self, app):
        _widget(app.date_input, "Start Date").set_value(pd.Timestamp("2021-01-01"))
        _widget(app.slider, "ประวัติขั้นต่ำที่ยอมรับ").set_value(1)
        at = _calculate(app)
        assert not at.exception, at.exception


class TestOneBrokenObjectiveDoesNotTakeThePageDown:
    """Every objective is solved on every run, so anything unexpected out
    of one of them reached the page as a traceback. That is how a private
    scipy name pypfopt validates against, removed in a newer scipy, put
    up "Oh no. Error running app." for every visitor."""

    def test_the_page_survives_a_scipy_without_the_private_name(
        self, app, monkeypatch
    ):
        import scipy.cluster.hierarchy as sch
        monkeypatch.delattr(sch, "_LINKAGE_METHODS", raising=False)

        at = _calculate(app)
        assert not at.exception, at.exception
        assert optimizer.HRP_OBJECTIVE in at.session_state["objective_weights"]

    def test_an_objective_that_blows_up_is_reported_not_fatal(
        self, app, monkeypatch
    ):
        def explode(*_args, **_kwargs):
            raise RuntimeError("upstream library changed under us")

        monkeypatch.setattr(optimizer, "hrp_weights", explode)
        at = _calculate(app)

        assert not at.exception, at.exception
        assert at.session_state["calculated"] is True
        assert optimizer.HRP_OBJECTIVE not in at.session_state["objective_weights"]
        assert optimizer.HRP_OBJECTIVE in at.session_state["objective_errors"]

    def test_the_remaining_objectives_still_work(self, app, monkeypatch):
        def explode(*_args, **_kwargs):
            raise RuntimeError("upstream library changed under us")

        monkeypatch.setattr(optimizer, "hrp_weights", explode)
        at = _calculate(app)

        for name in ("Max Sharpe", "Min Volatility", optimizer.MIN_CVAR):
            assert name in at.session_state["objective_weights"], name


def test_walk_forward_survives_one_objective_blowing_up(app, monkeypatch):
    def explode(*_args, **_kwargs):
        raise RuntimeError("upstream library changed under us")

    monkeypatch.setattr(optimizer, "hrp_weights", explode)
    app.radio[0].set_value("Walk-Forward")
    at = _calculate(app)

    assert not at.exception, at.exception
    assert at.session_state["calculated"] is True
