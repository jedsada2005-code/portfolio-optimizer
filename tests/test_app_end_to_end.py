"""End-to-end runs of the Streamlit script itself.

The unit tests cover the formulas; these catch the wiring between them
-- a name that only exists on one branch, a session key written in one
mode and read in another -- which py_compile and pytest cannot see.

Marked slow because each case downloads real prices from Yahoo.
"""

import pandas as pd
import pytest

from streamlit.testing.v1 import AppTest


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
    _widget(at.text_input, "Stock Symbols").set_value(SYMBOLS)
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
    _widget(app.slider, "น้ำหนักสูงสุด").set_value(0.4)
    _widget(app.slider, "เงินสด").set_value(0.2)
    assert not _calculate(app).error


def test_benchmark_can_be_left_empty(app):
    _widget(app.text_input, "Benchmark").set_value("")
    assert not _calculate(app).error


def test_a_single_symbol_is_refused(app):
    _widget(app.text_input, "Stock Symbols").set_value("SPY")
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
    _widget(app.slider, "น้ำหนักสูงสุด").set_value(0.4)
    at = _calculate(app)
    assert _one(at.query_params["symbols"]) == SYMBOLS
    assert _one(at.query_params["base"]) == "THB"
    assert _one(at.query_params["maxw"]) == "0.4"


def test_a_shared_link_reopens_with_the_same_settings():
    at = AppTest.from_file("app.py", default_timeout=300)
    at.query_params["symbols"] = "AAPL, MSFT"
    at.query_params["base"] = "THB"
    at.query_params["maxw"] = "0.45"
    at.query_params["mode"] = "In-sample (ทั้งช่วง)"
    at.query_params["reb"] = "รายปี"
    at.run()
    assert not at.exception, at.exception
    assert _widget(at.text_input, "Stock Symbols").value == "AAPL, MSFT"
    assert _widget(at.selectbox, "สกุลเงินฐาน").value == "THB"
    assert _widget(at.slider, "น้ำหนักสูงสุด").value == 0.45
    assert at.radio[0].value == "In-sample (ทั้งช่วง)"
    assert _widget(at.selectbox, "Rebalance").value == "รายปี"
