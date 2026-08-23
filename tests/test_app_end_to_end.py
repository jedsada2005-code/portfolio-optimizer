"""End-to-end runs of the Streamlit script itself.

The unit tests cover the formulas; these catch the wiring between them
-- a name that only exists on one branch, a session key written in one
mode and read in another -- which py_compile and pytest cannot see.

Marked slow because each case downloads real prices from Yahoo.
"""

import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest

import metrics


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
    at.button[0].click().run()
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
        assert any("25%" in e.value for e in at.error)

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
        return _widget(at.radio, "น้ำหนักที่ใช้ backtest")

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
        labels = [r.label for r in at.radio]
        assert "น้ำหนักที่ใช้ backtest" not in labels
        assert _widget(at.radio, "วัตถุประสงค์").options == ["Max Sharpe", "Min Volatility"]

    def test_cash_sleeve_reaches_the_nav_tab(self, app):
        _widget(app.slider, "เงินสด").set_value(25)
        at = _calculate(app)
        view = at.session_state["nav_view"]
        assert view["weights"][metrics.CASH_SYMBOL] == pytest.approx(0.25)
