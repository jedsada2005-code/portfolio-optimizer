import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
import io
from pypfopt.exceptions import OptimizationError

import custom_data
import fx
import metrics
import optimizer
import thai_mf

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# Random portfolios are sampled at n_samples but only PLOT_SAMPLE of
# them are drawn, which is what keeps the frontier tab responsive.
PLOT_SAMPLE = 6_000


MODES = ["Train / Test Split", "Walk-Forward", "In-sample (ทั้งช่วง)"]
OBJECTIVES = ["Max Sharpe", "Min Volatility"]
CUSTOM_SOURCE = "Custom (จากแท็บน้ำหนักพอร์ต)"
DEFAULT_SYMBOLS = "AMZN, META, LLY, SPY, NVDA, GOOGL"

# Starting points for people who have no idea what to type into an
# empty symbol box.
PRESETS = {
    "หุ้นเทคโนโลยีสหรัฐ": (
        "AMZN, META, LLY, SPY, NVDA, GOOGL",
        "หุ้นใหญ่ที่คนรู้จัก ผลตอบแทนสูงแต่ผันผวนแรงและเคลื่อนไหวไปทางเดียวกัน",
    ),
    "กระจายความเสี่ยงหลายสินทรัพย์": (
        "SPY, QQQ, GLD, TLT, EEM, XLE",
        "หุ้น ทองคำ พันธบัตร ตลาดเกิดใหม่ พลังงาน — ค่าสหสัมพันธ์ต่ำกว่ามาก",
    ),
    "หุ้นไทย": (
        "PTT.BK, ADVANC.BK, CPALL.BK, KBANK.BK, AOT.BK",
        "หุ้นใหญ่ใน SET ลองตั้งสกุลเงินฐานเป็น THB และ benchmark เป็น ^SET.BK",
    ),
    "พอร์ตความเสี่ยงต่ำ": (
        "AGG, TLT, GLD, SPY",
        "เน้นพันธบัตรและทองคำ ลองเพิ่มสัดส่วนเงินสดในการตั้งค่าขั้นสูง",
    ),
}


def qp_text(key, default):
    """Widget default taken from the URL, falling back to the built-in."""
    value = st.query_params.get(key)
    return value if value not in (None, "") else default


def qp_number(key, default, cast=float):
    try:
        return cast(st.query_params[key])
    except (KeyError, TypeError, ValueError):
        return default


def qp_date(key, default):
    try:
        return pd.Timestamp(st.query_params[key])
    except (KeyError, TypeError, ValueError):
        return default


def read_custom_weights(assets):
    """Normalised custom weights, read straight from the widget state.

    Keyed widgets keep their value in session_state from the start of a
    run, so the backtest can be computed before the tabs render without
    waiting for the weights tab to draw itself.
    """
    raw = {sym: float(st.session_state.get(f"cw_{sym}", 0.0)) for sym in assets}
    total = sum(raw.values())
    if total <= 0:
        return {sym: 1.0 / len(assets) for sym in assets}
    return {sym: value / total for sym, value in raw.items()}


def current_url():
    """Full page URL including the saved settings, for copying."""
    try:
        return st.context.url
    except Exception:
        query = "&".join(f"{k}={v}" for k, v in st.query_params.items())
        return f"?{query}"


def qp_choice(key, options, default_index=0):
    value = st.query_params.get(key)
    return options.index(value) if value in options else default_index


@st.cache_data(show_spinner=False, ttl=3600)
def download_fx_rates(currencies, start, end):
    """Units of each currency per one US dollar, from Yahoo."""
    if not currencies:
        return pd.DataFrame()
    symbols = [fx.FX_SYMBOL_TEMPLATE.format(currency=c) for c in currencies]
    frame = download_yf_close(tuple(symbols), start, end)
    if frame.empty:
        return pd.DataFrame()
    rename = {
        fx.FX_SYMBOL_TEMPLATE.format(currency=c): c
        for c in currencies
    }
    return frame.rename(columns=rename)


@st.cache_data(show_spinner=False, ttl=3600)
def download_yf_close(symbols, start, end):
    """Adjusted closes from Yahoo, cached so that re-running with the
    same symbols and dates does not re-download."""
    df = yf.download(
        tickers=list(symbols),
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
    )
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].ffill()
    else:
        close = df[["Close"]].rename(columns={"Close": symbols[0]}).ffill()
    close = close.dropna(how="all")
    return close.dropna(axis=1, how="all")
st.title("Portfolio Optimizer & Backtesting")

# ─── Sidebar: Inputs ───
with st.sidebar:
    st.header("Settings")

    # Only the inputs a typical run actually touches stay visible. Every
    # default below is usable as-is, so grouping the rest behind
    # expanders costs nothing and stops Calculate from sitting at the
    # bottom of nineteen controls.
    if "pending_symbols" in st.session_state:
        st.session_state["symbols_input"] = st.session_state.pop("pending_symbols")
    elif "symbols_input" not in st.session_state:
        st.session_state["symbols_input"] = qp_text("symbols", DEFAULT_SYMBOLS)
    symbols_input = st.text_input(
        "สินทรัพย์ในพอร์ต (คั่นด้วยจุลภาค)",
        key="symbols_input",
        help="หุ้น/ETF เช่น `SPY` · หุ้นไทยเติม `.BK` เช่น `PTT.BK` · กองทุนไทยเติม `MF:` เช่น `MF:K-GOLD-A(A)`",
    )
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=qp_date("start", pd.Timestamp("2010-01-01")),
            min_value=pd.Timestamp("1990-01-01"),
            max_value=pd.Timestamp.today(),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=qp_date("end", pd.Timestamp.today()),
            min_value=pd.Timestamp("1990-01-01"),
            max_value=pd.Timestamp.today(),
        )

    run_btn = st.button("Calculate", type="primary", use_container_width=True)
    if st.button("↺ รีเซ็ตการตั้งค่าทั้งหมด", use_container_width=True):
        st.query_params.clear()
        st.session_state.clear()
        st.rerun()

    with st.expander("💱 เงินและสกุลเงิน"):
        base_currency = st.selectbox(
            "สกุลเงินฐาน", fx.BASE_CURRENCIES, index=qp_choice("base", fx.BASE_CURRENCIES),
            help=(
                "แปลงทุกสินทรัพย์เป็นสกุลนี้ก่อนคำนวณ กองทุนไทยเป็น THB หุ้น US เป็น USD "
                "ถ้าไม่แปลง ผลของค่าเงินจะหายไปทั้งหมดและ volatility จะต่ำกว่าความจริง"
            ),
        )
        total_cash = st.number_input(
            f"เงินลงทุนตั้งต้น ({base_currency})",
            value=qp_number("cash", 1_000_000, int), step=100_000,
        )
        risk_free_rate = st.number_input(
            "Risk-Free Rate", value=qp_number("rf", 0.02), step=0.01, format="%.4f",
            help="ผลตอบแทนที่ได้โดยไม่มีความเสี่ยง ใช้เป็นฐานคำนวณ Sharpe และเป็นดอกเบี้ยของสัดส่วนเงินสด",
        )

    with st.expander("🔍 วิธีทดสอบ"):
        backtest_mode = st.radio(
            "โหมด Backtest",
            MODES,
            index=qp_choice("mode", MODES),
            help=(
                "In-sample หาน้ำหนักและทดสอบบนข้อมูลชุดเดียวกัน ผลลัพธ์จะสวยเกินจริงเสมอ "
                "Train/Test หาน้ำหนักจากช่วงแรก แล้วทดสอบบนช่วงหลังที่ไม่เคยเห็น "
                "Walk-Forward คำนวณน้ำหนักใหม่เป็นงวดๆ โดยใช้เฉพาะข้อมูลที่มีอยู่ ณ ตอนนั้น"
            ),
        )
        train_fraction = 0.7
        refit_label = "รายปี"
        if backtest_mode == "Train / Test Split":
            train_fraction = st.slider(
                "สัดส่วนช่วง Train", 30, 90, 70, step=5, format="%d%%",
                help="ที่เหลือใช้เป็นช่วง Test สำหรับวัดผลจริงแบบ out-of-sample",
            ) / 100
        elif backtest_mode == "Walk-Forward":
            refit_label = st.selectbox(
                "ความถี่การคำนวณน้ำหนักใหม่", ["รายไตรมาส", "รายปี"], index=1,
                help="ต้องมีข้อมูลอย่างน้อย 2 ปีก่อนการคำนวณครั้งแรก",
            )

        rebalance_label = st.selectbox(
            "ความถี่การ Rebalance",
            list(metrics.REBALANCE_FREQUENCIES),
            index=qp_choice("reb", list(metrics.REBALANCE_FREQUENCIES), 2),
            help=(
                "การคำนวณแบบเดิมสมมติว่าปรับพอร์ตกลับสัดส่วนเดิมทุกวันทำการโดยไม่มีค่าใช้จ่าย "
                "ซึ่งทำไม่ได้จริงและดันผลตอบแทนสูงเกินจริง"
            ),
        )
        cost_bps = st.number_input(
            "ค่าธรรมเนียมซื้อขาย (bps ต่อมูลค่าที่เทรด)",
            min_value=0.0, max_value=500.0, value=qp_number("cost", 0.0), step=5.0,
            help=(
                "100 bps = 1% คิดจากมูลค่าที่ซื้อขายจริงในแต่ละรอบ rebalance เท่านั้น "
                "(ไม่คิดตอนซื้อครั้งแรก) หมายเหตุ: NAV กองทุนและราคา ETF หัก "
                "ค่าธรรมเนียมจัดการรายปีไปแล้ว ช่องนี้จึงมีไว้ใส่ค่าธรรมเนียมขาย/รับซื้อคืน "
                "และค่าคอมมิชชั่นเท่านั้น ไม่ต้องใส่ TER ซ้ำ"
            ),
        )
        benchmark_symbol = st.text_input(
            "Benchmark (เว้นว่างได้)", value=qp_text("bench", "SPY"),
            help="สัญลักษณ์ Yahoo สำหรับเทียบผลงาน ไม่ถูกนับรวมเป็นสินทรัพย์ในพอร์ต เช่น SPY หรือ ^SET.BK",
        ).strip().upper()

    with st.expander("⚙️ การตั้งค่าขั้นสูง"):
        min_weight = st.slider(
            "น้ำหนักขั้นต่ำต่อสินทรัพย์", 0, 50,
            int(round(qp_number("minw", 0.0) * 100)), step=1, format="%d%%",
            help=(
                "บังคับให้ทุกตัวได้อย่างน้อยเท่านี้ กันไม่ให้ optimizer ตัดบางตัวเหลือ 0% "
                "ตั้งได้ไม่เกิน 100% ÷ จำนวนสินทรัพย์ และคิดจากส่วนที่เป็นสินทรัพย์เสี่ยง "
                "(ไม่รวมสัดส่วนเงินสด)"
            ),
        ) / 100
        max_weight = st.slider(
            "น้ำหนักสูงสุดต่อสินทรัพย์", 5, 100,
            int(round(qp_number("maxw", 1.0) * 100)), step=5, format="%d%%",
            help=(
                "กันไม่ให้ optimizer ทุ่มน้ำหนักเกือบทั้งหมดลงสินทรัพย์ตัวเดียว "
                "ต้องตั้งไม่ต่ำกว่า 100% หารด้วยจำนวนสินทรัพย์ เช่น 4 ตัวต้องอย่างน้อย 25%"
            ),
        ) / 100
        cash_fraction = st.slider(
            "สัดส่วนเงินสด", 0, 90,
            int(round(qp_number("cashpct", 0.0) * 100)), step=5, format="%d%%",
            help=(
                "กันเงินไว้เป็นเงินสดที่ได้ผลตอบแทนเท่า Risk-Free Rate "
                "สินทรัพย์เสี่ยงที่เหลือคงสัดส่วนภายในเดิม (two-fund separation)"
            ),
        ) / 100
        shrinkage = st.slider(
            "Covariance Shrinkage", 0.0, 1.0,
            qp_number("shrink", optimizer.DEFAULT_SHRINKAGE), step=0.05,
            format="%.2f",
            help=(
                "ดึงค่าสหสัมพันธ์เข้าหาค่าเฉลี่ย ทำให้น้ำหนักที่ได้เสถียรขึ้นและ "
                "ไม่สุดขั้ว 0 = ใช้ค่าจากข้อมูลดิบ, 1 = ใช้ค่าเฉลี่ยทั้งหมด"
            ),
        )

    with st.expander("📄 ข้อมูลของคุณเอง"):
        uploaded_price_files = st.file_uploader(
            "อัปโหลดไฟล์ราคา CSV/XLSX",
            type=["csv", "xlsx"],
            accept_multiple_files=True,
            help=(
                "รองรับ Date,AAPL,SPY... หรือ Date,Symbol,Close หรือ Date,Close "
                "(กรณี Date,Close จะใช้ชื่อไฟล์หรือชื่อ sheet เป็นชื่อสินทรัพย์)"
            ),
        )
        upload_currency = st.selectbox(
            "สกุลเงินของไฟล์ที่อัปโหลด", fx.BASE_CURRENCIES + ["EUR", "JPY", "GBP"],
            help="ใช้เมื่อมีไฟล์ CSV/XLSX เท่านั้น",
        )
        st.caption("กองทุนรวมไทย (`MF:`) ต้องใช้ API Key สองตัวคนละชุด สมัครฟรีที่ secopendata.sec.or.th")
        sec_factsheet_key = st.text_input(
            "SEC Fund Factsheet API Key",
            value="",
            type="password",
            help="จำเป็นเฉพาะเมื่อกรอกกองทุนรวมไทยด้วย prefix MF:",
        )
        sec_daily_info_key = st.text_input(
            "SEC Fund Daily Info API Key",
            value="",
            type="password",
            help="เป็น API key คนละตัวกับ Fund Factsheet ต้อง subscribe แยกกัน",
        )

    with st.expander("📖 คู่มือและข้อควรระวัง"):
        st.caption("**รูปแบบสัญลักษณ์**")
        st.caption("• หุ้น/ETF ต่างประเทศ ใส่ชื่อได้เลย เช่น `AMZN`, `SPY`")
        st.caption("• หุ้นไทยเติม `.BK` เช่น `PTT.BK`")
        st.caption("• กองทุนรวมไทยเติม `MF:` เช่น `MF:K-CHANGE-A(A)` — ต้องกรอก API Key ทั้ง 2 ช่อง")
        st.caption("• ไฟล์อัปโหลดต้องเป็นราคาหรือ NAV ไม่ใช่ daily return และต้องมีคอลัมน์วันที่")
        st.caption("**การอ่านผลลัพธ์**")
        st.caption("• **Train/Test** และ **Walk-Forward** วัดจากข้อมูลที่ optimizer ไม่เคยเห็น ส่วน **In-sample** สวยเกินจริงเสมอ")
        st.caption("• Backtest เริ่มนับจากวันแรกที่ **ทุกตัว** ในพอร์ตมีข้อมูลครบ ไม่ใช่จาก Start Date เสมอไป")
        st.caption("• ทุกสินทรัพย์ถูกแปลงเป็นสกุลเงินฐานก่อนคำนวณ ผลตอบแทนจึงรวมผลของค่าเงินแล้ว")
        st.caption("• NAV กองทุนและราคา ETF หักค่าธรรมเนียมจัดการรายปีไปแล้ว ช่องค่าธรรมเนียมมีไว้ใส่ค่าซื้อขายเท่านั้น")
        st.caption("**อื่นๆ**")
        st.caption("• Custom Weight ไม่ต้องรวมกันเป็น 1.0 ระบบ normalize ให้อัตโนมัติ")
        st.caption("• สินทรัพย์บางตัวอาจโหลดไม่สำเร็จเพราะเขียนชื่อผิด หรือปีนั้นยังไม่มีข้อมูล")
        st.caption("• กด Calculate แล้ว URL จะเก็บการตั้งค่าทั้งหมด bookmark หรือส่งต่อได้")

# ─── Parse symbols ───
stock_list = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

# Identifies the inputs a displayed result was produced from, so edits
# made without pressing Calculate can be flagged as stale.
input_signature = (
    tuple(stock_list),
    str(start_date),
    str(end_date),
    float(risk_free_rate),
    float(total_cash),
    tuple(sorted(f.name for f in uploaded_price_files)),
    backtest_mode,
    float(train_fraction),
    benchmark_symbol,
    rebalance_label,
    float(cost_bps),
    base_currency,
    upload_currency,
    refit_label,
    float(cash_fraction),
    float(max_weight),
    float(min_weight),
    float(shrinkage),
)

if run_btn:
    if start_date >= end_date:
        st.error(
            f"⚠️ ช่วงวันที่ไม่ถูกต้อง — Start Date ({start_date}) ต้องอยู่ก่อน "
            f"End Date ({end_date})"
        )
        st.stop()

    uploaded_frames = []
    file_errors = []
    for uploaded_file in uploaded_price_files:
        try:
            uploaded_file.seek(0)
            uploaded_frames.append(
                custom_data.parse_price_file(uploaded_file, uploaded_file.name)
            )
        except custom_data.PriceDataError as exc:
            file_errors.append(f"{uploaded_file.name}: {exc}")

    if file_errors:
        st.error("อ่านไฟล์ราคาไม่สำเร็จ:\n\n" + "\n".join(f"- {err}" for err in file_errors))
        st.stop()

    if uploaded_frames:
        uploaded_prices = pd.concat(uploaded_frames, axis=1)
        if not uploaded_prices.columns.is_unique:
            duplicates = sorted(set(uploaded_prices.columns[uploaded_prices.columns.duplicated()]))
            st.error(
                "ไฟล์ที่อัปโหลดมีชื่อสินทรัพย์ซ้ำกัน: "
                + ", ".join(duplicates)
                + " — กรุณาเปลี่ยนชื่อคอลัมน์หรือชื่อไฟล์ให้ไม่ซ้ำ"
            )
            st.stop()
    else:
        uploaded_prices = pd.DataFrame()

    if len(stock_list) + len(uploaded_prices.columns) < 2:
        st.error("ต้องมีสินทรัพย์อย่างน้อย 2 ตัวจาก Yahoo/กองทุนไทย/CSV เพื่อคำนวณพอร์ต")
        st.stop()

    yf_symbols, mf_symbols = thai_mf.split_symbols(stock_list)

    if benchmark_symbol and benchmark_symbol in stock_list:
        st.warning(
            f"⚠️ **{benchmark_symbol}** ถูกใช้เป็นทั้งสินทรัพย์ในพอร์ตและ benchmark "
            "— การเปรียบเทียบจะเป็นการเทียบพอร์ตกับส่วนหนึ่งของตัวเอง "
            "Beta จะเข้าใกล้ 1 โดยไม่มีความหมาย เลือก benchmark ตัวอื่นจะได้ผลที่ตีความได้"
        )

    if mf_symbols and (not sec_factsheet_key or not sec_daily_info_key):
        st.error(
            "⚠️ พบสัญลักษณ์กองทุนไทย (MF:) แต่ยังไม่ได้กรอก SEC API Key ให้ครบทั้ง 2 ช่อง "
            "(Fund Factsheet และ Fund Daily Info) ในแถบด้านซ้าย"
        )
        st.stop()

    with st.spinner("Downloading data..."):
        if yf_symbols:
            data_close = download_yf_close(
                tuple(yf_symbols), str(start_date), str(end_date)
            )
        else:
            data_close = pd.DataFrame()

    mf_missing = []
    mf_incomplete = []
    mf_resolved_classes = {}
    if mf_symbols:
        with st.spinner("Downloading Thai mutual fund data..."):
            client = thai_mf.SECFundClient(sec_factsheet_key, sec_daily_info_key)
            fund_navs = {}
            mf_progress = st.progress(0.0, text="กำลังดึงข้อมูลกองทุนไทย...")
            for position, name in enumerate(mf_symbols, start=1):
                display_symbol = f"MF:{name}"
                mf_progress.progress(
                    (position - 1) / len(mf_symbols),
                    text=f"กำลังดึง {position}/{len(mf_symbols)}: {display_symbol}",
                )
                try:
                    proj_id, preferred_class = thai_mf.resolve_fund_id(name, client)
                    if proj_id is None:
                        mf_missing.append(display_symbol)
                        continue
                    nav_series, incomplete, chosen_class = thai_mf.get_nav_history(
                        client, proj_id, preferred_class, start_date, end_date
                    )
                except thai_mf.SECAPIError:
                    mf_missing.append(display_symbol)
                    continue
                if nav_series.empty:
                    mf_missing.append(display_symbol)
                    continue
                if incomplete:
                    mf_incomplete.append(display_symbol)
                mf_resolved_classes[display_symbol] = chosen_class
                fund_navs[display_symbol] = nav_series
            mf_progress.empty()
            data_close = thai_mf.merge_fund_navs(data_close, fund_navs)

    if not uploaded_prices.empty:
        try:
            data_close = custom_data.merge_uploaded_prices(data_close, uploaded_prices)
        except custom_data.CSVPriceDataError as exc:
            st.error(str(exc))
            st.stop()

    if data_close.empty:
        st.error("No data downloaded. Check symbols and date range.")
        st.stop()

    asset_currencies = {
        column: fx.currency_for_symbol(
            column,
            default=upload_currency if column in uploaded_prices.columns else "USD",
        )
        for column in data_close.columns
    }
    needed = fx.required_currencies(asset_currencies, base_currency)
    if needed:
        with st.spinner("Downloading exchange rates..."):
            rates = download_fx_rates(tuple(needed), str(start_date), str(end_date))
        try:
            data_close = fx.convert_prices(
                data_close, asset_currencies, base_currency, rates
            )
        except fx.FXError as exc:
            st.error(f"⚠️ {exc} — ลองเปลี่ยนสกุลเงินฐาน หรือเอาสินทรัพย์สกุลนั้นออก")
            st.stop()
        converted = sorted({
            f"{sym} ({cur})" for sym, cur in asset_currencies.items()
            if cur != base_currency
        })
        if converted:
            st.caption(
                f"💱 แปลงเป็น {base_currency} แล้ว: " + ", ".join(converted)
                + " — ผลตอบแทนและความผันผวนที่แสดงรวมผลของค่าเงินไว้แล้ว"
            )

    benchmark_close = pd.Series(dtype=float)
    if benchmark_symbol:
        with st.spinner(f"Downloading benchmark {benchmark_symbol}..."):
            bench_df = download_yf_close((benchmark_symbol,), str(start_date), str(end_date))
        if bench_df.empty:
            st.warning(
                f"⚠️ โหลด benchmark **{benchmark_symbol}** ไม่สำเร็จ — ข้ามการเปรียบเทียบ"
            )
        else:
            benchmark_close = bench_df.iloc[:, 0].rename(benchmark_symbol)
            bench_currency = fx.currency_for_symbol(benchmark_symbol)
            if bench_currency != base_currency:
                bench_needed = fx.required_currencies(
                    {benchmark_symbol: bench_currency}, base_currency
                )
                bench_rates = download_fx_rates(
                    tuple(bench_needed), str(start_date), str(end_date)
                )
                try:
                    benchmark_close = fx.convert_prices(
                        benchmark_close.to_frame(),
                        {benchmark_symbol: bench_currency},
                        base_currency,
                        bench_rates,
                    ).iloc[:, 0]
                except fx.FXError:
                    st.warning(
                        f"⚠️ แปลงสกุลเงินของ benchmark **{benchmark_symbol}** ไม่ได้ "
                        "— ข้ามการเปรียบเทียบ"
                    )
                    benchmark_close = pd.Series(dtype=float)

    # แจ้งหุ้นที่โหลดสำเร็จ / ไม่สำเร็จ
    loaded = list(data_close.columns)
    missing = [s for s in yf_symbols if s not in loaded] + mf_missing
    if missing:
        st.warning(f"⚠️ ไม่พบข้อมูล: **{', '.join(missing)}** — ตรวจสอบชื่อ symbol อีกครั้ง")
    csv_loaded = [col for col in uploaded_prices.columns if col in loaded]
    if csv_loaded:
        st.info(
            f"📄 รวมข้อมูลจากไฟล์อัปโหลด {len(csv_loaded)} ตัว: **{', '.join(csv_loaded)}** "
            "โดย forward-fill เฉพาะหลังวันแรกที่มีราคา"
        )
    if mf_incomplete:
        st.warning(
            f"⚠️ ข้อมูล NAV อาจไม่ครบทุกวันสำหรับ: **{', '.join(mf_incomplete)}** "
            "(SEC API rate limit ระหว่างดึงข้อมูล ลองกด Calculate ซ้ำเพื่อดึงวันที่เหลือจาก cache)"
        )
    st.success(f"✅ โหลดสำเร็จ {len(loaded)} ตัว: **{', '.join(loaded)}**")

    resolved_notes = [
        f"{sym} → {cls}"
        for sym, cls in mf_resolved_classes.items()
        if sym in loaded and cls != "main" and cls.upper() != sym[3:].upper()
    ]
    if resolved_notes:
        st.caption(
            "ℹ️ กองทุนไทยที่มีหลายชนิดหน่วยลงทุน (share class) เลือกใช้ชนิดนี้: "
            + ", ".join(resolved_notes)
            + " — ถ้าต้องการชนิดอื่น ให้พิมพ์ชื่อชนิดหน่วยลงทุนแบบเต็ม เช่น `MF:K-GOLD-A(D)`"
        )

    # ─── Calculations ───
    with st.spinner("Computing efficient frontier..."):
        if backtest_mode == "Walk-Forward":
            split_date = None
            train_close = data_close
            test_close = data_close
        elif backtest_mode == "Train / Test Split":
            split_date = metrics.split_index(data_close.index, train_fraction)
            train_close = data_close.loc[:split_date]
            test_close = data_close.loc[split_date:]
            if len(test_close) < 20:
                st.error(
                    "⚠️ ช่วง Test สั้นเกินไป — ลดสัดส่วน Train ลง หรือขยายช่วงวันที่"
                )
                st.stop()
        else:
            split_date = None
            train_close = data_close
            test_close = data_close

        weekly = train_close.resample("W-FRI").last()
        ar, ar_observations = metrics.annual_return_estimates(weekly)

        # An expected return built from a handful of overlapping 52-week
        # windows is noise, and the optimiser will happily chase it into
        # a near-100% allocation. Require a real sample, not merely a
        # non-NaN value, and say how short each excluded asset was.
        insufficient = metrics.unreliable_assets(ar_observations)
        if insufficient:
            detail = ", ".join(
                f"**{sym}** ({count} สัปดาห์)" for sym, count in insufficient.items()
            )
            st.warning(
                f"⚠️ ข้อมูลไม่พอสำหรับประมาณผลตอบแทนที่เชื่อถือได้ — ต้องมีอย่างน้อย "
                f"{metrics.MIN_ANNUAL_OBSERVATIONS} สัปดาห์ (ประมาณ 2 ปี) ในช่วงวันที่ที่เลือก: "
                f"{detail} — ไม่รวมในการคำนวณพอร์ต "
                "(ลองขยายช่วงวันที่ หรือปรับ Start Date ให้อยู่หลังวันที่กองทุนจดทะเบียน)"
            )
            drop = list(insufficient)
            data_close = data_close.drop(columns=drop)
            train_close = train_close.drop(columns=drop)
            test_close = test_close.drop(columns=drop)
            weekly = weekly.drop(columns=drop)
            ar = ar.drop(index=drop)

        if len(ar) < 2:
            st.error(
                "เหลือสินทรัพย์ที่มีข้อมูลเพียงพอน้อยกว่า 2 ตัว ไม่สามารถคำนวณพอร์ตได้ "
                "ลองขยายช่วงวันที่ หรือลดกองทุนที่เพิ่งจดทะเบียนออก"
            )
            st.stop()

        try:
            optimizer.validate_bounds(len(ar), max_weight, min_weight)
        except ValueError as exc:
            st.error(f"⚠️ {exc}")
            st.stop()

        sample_cov = weekly.pct_change().cov() * 52
        covr = optimizer.shrink_covariance(sample_cov, shrinkage)

        # Random portfolios
        n_samples = 200_000
        rng = np.random.default_rng(42)
        w = optimizer.sample_weights(
            rng, len(ar), n_samples, max_weight, min_weight=min_weight
        )
        n_samples = len(w)
        rets = w.dot(ar)
        # NumPy's BLAS kernels raise spurious divide/overflow warnings on
        # matmuls this wide -- an identity matrix times ones trips them
        # too. The results are finite; only the FPU flags are wrong.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            stds = np.sqrt((w.T * (covr.values @ w.T)).sum(axis=0))
        sharpes = (rets - risk_free_rate) / stds

        # Plot a subsample: 200k markers is what makes this tab crawl,
        # and the cloud looks identical at a few thousand points.
        plot_n = min(PLOT_SAMPLE, n_samples)
        pick = rng.choice(n_samples, plot_n, replace=False)
        stds, rets, sharpes = stds[pick], rets[pick], sharpes[pick]

        # Max Sharpe weights
        try:
            cleaned = optimizer.optimize_weights(
                ar, covr, "Max Sharpe", risk_free_rate, max_weight, min_weight
            )
        except (ValueError, OptimizationError):
            st.error(
                "⚠️ หาพอร์ต Max Sharpe ไม่ได้ — สินทรัพย์ที่เลือกมีผลตอบแทนคาดหวังใกล้เคียงหรือต่ำกว่า "
                f"Risk-Free Rate ที่ตั้งไว้ ({risk_free_rate:.2%}) เกินไป "
                "ลองลด Risk-Free Rate ลง หรือเพิ่มสินทรัพย์ที่ผลตอบแทนสูงกว่าเข้าไปในพอร์ต"
            )
            st.stop()
        opt_ret, opt_vol, opt_sharpe = optimizer.portfolio_performance(
            ar, covr, cleaned, risk_free_rate
        )
        opt_ret, opt_vol = metrics.blend_performance(
            opt_ret, opt_vol, risk_free_rate, cash_fraction
        )

        # Min Volatility weights
        mv_cleaned = optimizer.optimize_weights(
            ar, covr, "Min Volatility", risk_free_rate, max_weight, min_weight
        )
        mv_ret, mv_vol, mv_sharpe = optimizer.portfolio_performance(
            ar, covr, mv_cleaned, risk_free_rate
        )
        mv_ret, mv_vol = metrics.blend_performance(
            mv_ret, mv_vol, risk_free_rate, cash_fraction
        )

        # Efficient frontier curve, solved directly instead of read back
        # out of a throwaway matplotlib figure.
        ef_x, ef_y = optimizer.frontier_curve(ar, covr, max_weight, min_weight=min_weight)
        # Draw the unconstrained frontier alongside it so the cost of the
        # bounds is visible rather than merely asserted.
        if max_weight < 1.0 or min_weight > 0.0:
            free_x, free_y = optimizer.frontier_curve(ar, covr)
        else:
            free_x, free_y = np.array([]), np.array([])

    # Store in session for tabs
    st.session_state["data_close"] = data_close
    st.session_state["train_close"] = train_close
    st.session_state["test_close"] = test_close
    st.session_state["split_date"] = split_date
    st.session_state["backtest_mode"] = backtest_mode
    st.session_state["benchmark"] = benchmark_close
    st.session_state["benchmark_symbol"] = benchmark_symbol
    st.session_state["rebalance_label"] = rebalance_label
    st.session_state["cost_bps"] = cost_bps
    st.session_state["base_currency"] = base_currency
    st.session_state["refit_freq"] = metrics.REBALANCE_FREQUENCIES[refit_label]
    st.session_state["refit_label"] = refit_label
    st.session_state["cash_fraction"] = cash_fraction
    st.session_state["max_weight"] = max_weight
    st.session_state["shrinkage"] = shrinkage
    st.session_state["cost_bps_used"] = cost_bps
    st.session_state["ar"] = ar
    st.session_state["covr"] = covr
    st.session_state["cleaned"] = cleaned
    st.session_state["opt_perf"] = (opt_ret, opt_vol, opt_sharpe)
    st.session_state["mv_cleaned"] = mv_cleaned
    st.session_state["mv_perf"] = (mv_ret, mv_vol, mv_sharpe)
    st.session_state["random"] = (stds, rets, sharpes)
    st.session_state["ef_curve"] = (ef_x, ef_y)
    st.session_state["ef_curve_free"] = (free_x, free_y)
    st.session_state["min_weight"] = min_weight
    st.session_state["stock_list"] = list(data_close.columns)
    st.session_state["total_cash"] = total_cash
    st.session_state["risk_free_rate"] = risk_free_rate
    st.session_state["calculated"] = True
    st.session_state["input_signature"] = input_signature

    # D5: the URL now carries the whole setup, so it can be bookmarked
    # or shared and will reopen with the same inputs.
    st.query_params.update({
        "symbols": symbols_input, "start": str(start_date), "end": str(end_date),
        "base": base_currency, "cash": str(total_cash), "rf": str(risk_free_rate),
        "mode": backtest_mode, "reb": rebalance_label, "bench": benchmark_symbol,
        "cost": str(cost_bps), "maxw": str(max_weight), "minw": str(min_weight),
        "cashpct": str(cash_fraction),
        "shrink": str(shrinkage),
    })

# ─── Display results ───
if st.session_state.get("calculated"):
    data_close = st.session_state["data_close"]
    ar = st.session_state["ar"]
    covr = st.session_state["covr"]
    cleaned = st.session_state["cleaned"]
    opt_ret, opt_vol, opt_sharpe = st.session_state["opt_perf"]
    mv_cleaned = st.session_state["mv_cleaned"]
    mv_ret, mv_vol, mv_sharpe = st.session_state["mv_perf"]
    stds, rets, sharpes = st.session_state["random"]
    ef_x, ef_y = st.session_state["ef_curve"]
    free_x, free_y = st.session_state["ef_curve_free"]
    min_weight = st.session_state["min_weight"]
    stock_list = st.session_state["stock_list"]
    total_cash = st.session_state["total_cash"]
    risk_free_rate = st.session_state["risk_free_rate"]
    train_close = st.session_state["train_close"]
    test_close = st.session_state["test_close"]
    split_date = st.session_state["split_date"]
    benchmark = st.session_state["benchmark"]
    benchmark_symbol = st.session_state["benchmark_symbol"]
    rebalance_label = st.session_state["rebalance_label"]
    rebalance_freq = metrics.REBALANCE_FREQUENCIES[rebalance_label]
    cost_bps = st.session_state["cost_bps"]
    base_currency = st.session_state["base_currency"]
    refit_freq = st.session_state["refit_freq"]
    refit_label = st.session_state["refit_label"]
    cash_fraction = st.session_state["cash_fraction"]
    max_weight = st.session_state["max_weight"]
    shrinkage = st.session_state["shrinkage"]
    backtest_mode = st.session_state["backtest_mode"]

    if st.session_state.get("input_signature") != input_signature:
        st.warning(
            "⚠️ การตั้งค่าในแถบด้านซ้ายถูกแก้ไขหลังจากคำนวณครั้งล่าสุด — "
            "ผลลัพธ์ด้านล่างยังเป็นของค่าเดิม กด **Calculate** เพื่อคำนวณใหม่"
        )

    # A run's mode and settings change what every number below means,
    # so state them once, above the tabs, instead of only inside one.
    mode_icon = {"Walk-Forward": "🔒", "Train / Test Split": "🔒"}.get(backtest_mode, "⚠️")
    status = [
        f"{mode_icon} **{backtest_mode}**",
        f"💱 {base_currency}",
        f"🔄 Rebalance {rebalance_label}",
    ]
    if cost_bps:
        status.append(f"💸 {cost_bps:.0f} bps")
    if cash_fraction:
        status.append(f"💵 เงินสด {cash_fraction:.0%}")
    if max_weight < 1.0:
        status.append(f"⚖️ สูงสุด {max_weight:.0%}/ตัว")
    if min_weight > 0:
        status.append(f"⚖️ ขั้นต่ำ {min_weight:.0%}/ตัว")
    if benchmark_symbol:
        status.append(f"📊 เทียบ {benchmark_symbol}")
    st.caption(" · ".join(status))

    with st.expander("🔗 บันทึกหรือแชร์พอร์ตนี้"):
        st.caption("ลิงก์นี้เก็บการตั้งค่าทั้งหมดไว้ เปิดแล้วได้ค่าเดิม ส่งต่อให้คนอื่นได้")
        st.code(current_url(), language=None)

    # Computed before the tabs so every tab is a renderer: the weights
    # tab can show backtest diagnostics, and the NAV tab cannot drift
    # from the backtest by recomputing its own version.
    custom_w_available = read_custom_weights(sorted(ar.index))
    if backtest_mode == "Walk-Forward":
        walk_objective = st.radio(
            "วัตถุประสงค์ที่ใช้คำนวณน้ำหนักใหม่ทุกงวด",
            OBJECTIVES, horizontal=True,
        )
        weight_source = walk_objective
        active_w = cleaned if walk_objective == "Max Sharpe" else mv_cleaned
    else:
        sources = OBJECTIVES + ([CUSTOM_SOURCE] if custom_w_available else [])
        weight_source = st.radio(
            "น้ำหนักที่ใช้ backtest", sources, horizontal=True,
            help=(
                "Max Sharpe และ Min Volatility ใช้น้ำหนักที่ optimizer คำนวณแบบเป๊ะๆ "
                "ส่วน Custom ใช้ค่าจาก slider ในแท็บน้ำหนักพอร์ต ซึ่งปัดเศษทีละ 1%"
            ),
        )
        walk_objective = (
            weight_source if weight_source in OBJECTIVES else "Max Sharpe"
        )
        if weight_source == "Max Sharpe":
            active_w = cleaned
        elif weight_source == "Min Volatility":
            active_w = mv_cleaned
        else:
            active_w = custom_w_available
    
    if backtest_mode == "Walk-Forward":
        st.info(
            f"โหมด Walk-Forward คำนวณน้ำหนักใหม่ทุกงวดด้วยวัตถุประสงค์ **{walk_objective}** "
            "— น้ำหนักของแต่ละงวดดูได้จากตารางด้านล่าง"
        )
    else:
        st.info(
            f"ใช้น้ำหนัก **{weight_source}**: "
            + ", ".join(
                f"{k}={v:.1%}" for k, v in sorted(active_w.items()) if v > 0
            )
        )
    
    # C2: a cash sleeve is modelled as a synthetic holding accruing
    # the risk-free rate, so it flows through the ordinary simulator.
    backtest_w = metrics.blend_with_cash(active_w, cash_fraction)
    test_prices = test_close
    train_prices = train_close
    if cash_fraction > 0:
        test_prices = test_close.assign(
            **{metrics.CASH_SYMBOL: metrics.cash_price_series(test_close.index, risk_free_rate)}
        )
        train_prices = train_close.assign(
            **{metrics.CASH_SYMBOL: metrics.cash_price_series(train_close.index, risk_free_rate)}
        )
        st.caption(
            f"💵 ถือเงินสด {cash_fraction:.0%} ที่ได้ผลตอบแทน {risk_free_rate:.2%} ต่อปี "
            f"— สินทรัพย์เสี่ยงที่เหลือ {1 - cash_fraction:.0%} คงสัดส่วนภายในเดิม"
        )
    
    walk_result = None
    if backtest_mode == "Walk-Forward":
        with st.spinner("Running walk-forward..."):
            walk_result = optimizer.walk_forward(
                test_close, risk_free_rate, walk_objective, max_weight,
                shrinkage, refit_freq, cost_bps,
                cash_fraction=cash_fraction, rebalance_freq=rebalance_freq,
                min_weight=min_weight,
            )
        if walk_result.returns.empty:
            st.error(
                "⚠️ ข้อมูลไม่พอสำหรับ Walk-Forward — ต้องมีอย่างน้อย 2 ปี "
                "ก่อนการคำนวณน้ำหนักครั้งแรก ลองขยายช่วงวันที่"
            )
            st.stop()
        weights_in_force = walk_result.weight_history[-1][1]
        result = metrics.simulate_portfolio(
            test_prices, weights_in_force, rebalance_freq, cost_bps
        )
        port_daily = walk_result.returns
        daily_returns = result.assets
    else:
        weights_in_force = backtest_w
        # In split mode the weights were fitted on train_close only,
        # so the headline backtest runs on the untouched test window.
        result = metrics.simulate_portfolio(
            test_prices, backtest_w, rebalance_freq, cost_bps
        )
        port_daily = result.returns
        daily_returns = result.assets
    
    if port_daily.empty:
        st.error(
            "⚠️ ช่วงเวลาของสินทรัพย์ในพอร์ตไม่ทับซ้อนกันเลย จึง backtest ไม่ได้ "
            "— ลองเอาสินทรัพย์ที่เพิ่งเริ่มมีข้อมูลออก หรือขยายช่วงวันที่"
        )
        st.stop()
    
    # The backtest can only begin once every holding actually exists.
    requested_start = pd.Timestamp(test_prices.index[0])
    if result.start is not None and result.start > requested_start:
        firsts = metrics.first_valid_dates(test_prices[result.held])
        limiter = firsts.idxmax()
        st.warning(
            f"⚠️ Backtest เริ่มจริงที่ **{result.start.date()}** ไม่ใช่ "
            f"{requested_start.date()} เพราะ **{limiter}** เพิ่งมีข้อมูลวันแรกตอนนั้น "
            "— พอร์ตจะถือครบทุกตัวได้ก็ต่อเมื่อทุกตัวมีอยู่จริงแล้ว"
        )
    
    cumulative = (1 + port_daily).cumprod()
    stats = metrics.backtest_stats(port_daily, risk_free_rate)
    periods_per_year = stats["periods_per_year"]
    n_years = stats["years"]
    total_ret = stats["total_return"]
    ann_ret = stats["annual_return"]
    ann_vol = stats["annual_volatility"]
    sharpe = stats["sharpe"]
    max_dd = stats["max_drawdown"]
    calmar = stats["calmar"]
    sortino = stats["sortino"]
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    
    if backtest_mode == "Walk-Forward":
        st.success(
            f"🔒 **Walk-Forward** — คำนวณน้ำหนักใหม่{refit_label} "
            f"รวม **{len(walk_result.weight_history)} ครั้ง** โดยแต่ละครั้งใช้เฉพาะข้อมูล "
            "ที่มีอยู่ ณ วันนั้น ทั้งช่วงที่ทดสอบจึงเป็น out-of-sample ทั้งหมด"
        )
    elif split_date is not None:
        st.success(
            f"🔒 **Out-of-sample** — น้ำหนักคำนวณจากข้อมูลถึง **{split_date.date()}** "
            f"แล้วทดสอบบน **{result.start.date()} ถึง {port_daily.index[-1].date()}** "
            "ซึ่งเป็นข้อมูลที่ optimizer ไม่เคยเห็น"
        )
    else:
        st.warning(
            "⚠️ **In-sample** — น้ำหนักถูกหาจากข้อมูลชุดเดียวกับที่ใช้ทดสอบ "
            "ผลลัพธ์จึงสวยเกินจริงเสมอ เปลี่ยนเป็นโหมด Train/Test Split เพื่อดูผลจริง"
        )
    
    # ── Benchmark ──
    # Computed before the summary so the headline metrics can carry
    # a delta against it.
    bench_daily = pd.Series(dtype=float)
    bench_stats = None
    beta = alpha = 0.0
    if not benchmark.empty:
        bench_window = benchmark.reindex(
            benchmark.index.union(port_daily.index)
        ).ffill().reindex(port_daily.index)
        bench_daily = bench_window.pct_change().fillna(0.0)
        bench_stats = metrics.backtest_stats(bench_daily, risk_free_rate)
        beta, alpha = metrics.beta_alpha(
            port_daily, bench_daily, risk_free_rate, periods_per_year
        )
    
    def build_workbook():
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame({
                "Asset": list(weights_in_force),
                "Weight": list(weights_in_force.values()),
                "Source": [
                    f"{weight_source} (งวดล่าสุด)" if walk_result is not None
                    else weight_source
                ] * len(weights_in_force),
            }).to_excel(writer, sheet_name="Weights", index=False)
            pd.DataFrame(
                {"Metric": list(stats), "Value": list(stats.values())}
            ).to_excel(writer, sheet_name="Stats", index=False)
            pd.DataFrame({
                "Date": port_daily.index, "Daily Return": port_daily.values,
                "Cumulative": cumulative.values, "Drawdown": drawdown.values,
            }).to_excel(writer, sheet_name="Backtest", index=False)
            data_close.to_excel(writer, sheet_name="Prices")
            if walk_result is not None:
                pd.DataFrame(
                    [w for _, w in walk_result.weight_history],
                    index=[d.date() for d, _ in walk_result.weight_history],
                ).to_excel(writer, sheet_name="Walk-Forward Weights")
        return buffer.getvalue()

    # Hand the NAV tab the exact series behind these numbers.
    st.session_state["nav_view"] = {
        "returns": port_daily,
        "asset_returns": daily_returns,
        "weights": backtest_w if walk_result is None else weights_in_force,
        "held": result.held,
        "start": result.start,
        "source": weight_source,
        "is_walk_forward": walk_result is not None,
    }

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 เส้นขอบประสิทธิภาพ",
        "⚖️ น้ำหนักพอร์ต",
        "🔍 ทดสอบย้อนหลัง",
        "💰 มูลค่าพอร์ต",
    ])

    # ════════════════════════════════════════
    # Tab 1: Efficient Frontier
    # ════════════════════════════════════════
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=stds, y=rets, mode="markers",
            marker=dict(size=2, color=sharpes, colorscale="Viridis_r",
                        colorbar=dict(title="Sharpe"), opacity=0.6),
            name="Random Portfolios",
        ))
        if len(free_x):
            fig.add_trace(go.Scatter(
                x=free_x, y=free_y, mode="lines",
                line=dict(color="#9E9E9E", width=1.5, dash="dash"),
                name="ไม่มีข้อจำกัดน้ำหนัก",
            ))
        fig.add_trace(go.Scatter(
            x=ef_x, y=ef_y,
            mode="lines" if len(ef_x) > 1 else "markers",
            line=dict(color="red", width=2),
            marker=dict(color="red", size=12),
            name="Efficient Frontier",
        ))
        fig.add_trace(go.Scatter(
            x=[opt_vol], y=[opt_ret], mode="markers",
            marker=dict(size=14, color="gold", symbol="star",
                        line=dict(width=1, color="black")),
            name=f"Max Sharpe (SR={opt_sharpe:.2f})",
        ))
        fig.add_trace(go.Scatter(
            x=[mv_vol], y=[mv_ret], mode="markers",
            marker=dict(size=14, color="limegreen", symbol="diamond",
                        line=dict(width=1, color="black")),
            name=f"Min Volatility (SR={mv_sharpe:.2f})",
        ))
        fig.update_layout(
            title="Efficient Frontier with Random Portfolios",
            xaxis_title="Annual Volatility",
            yaxis_title="Expected Annual Return",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(ef_x) <= 1:
            st.warning(
                f"⚠️ น้ำหนักขั้นต่ำ {min_weight:.0%} × {len(stock_list)} ตัว = 100% พอดี "
                "จึงเหลือพอร์ตที่เป็นไปได้เพียงแบบเดียว คือลงเท่ากันทุกตัว "
                "— เส้น frontier จึงยุบเหลือจุดเดียว ลดขั้นต่ำลงเพื่อให้มีทางเลือก"
            )
        elif len(free_x):
            gap = np.interp(ef_x, free_x, free_y) - ef_y
            st.caption(
                f"เส้นประ = ไม่มีข้อจำกัดน้ำหนัก · ที่ระดับความเสี่ยงเท่ากัน ข้อจำกัดที่ตั้งไว้ "
                f"ทำให้ผลตอบแทนคาดหวังลดลงเฉลี่ย **{gap.mean():.2%}** ต่อปี "
                "(แลกกับพอร์ตที่กระจายตัวกว่าและมักทนทานกว่าเมื่อเจอข้อมูลจริง)"
            )

        st.subheader("ค่าสหสัมพันธ์ระหว่างสินทรัพย์")
        correlation = data_close.resample("W-FRI").last().pct_change().corr()
        fig_corr = px.imshow(
            correlation.values,
            x=correlation.columns.tolist(), y=correlation.index.tolist(),
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            aspect="auto", text_auto=".2f",
        )
        fig_corr.update_layout(height=max(320, len(correlation) * 42))
        st.plotly_chart(fig_corr, use_container_width=True)

        off_diagonal = correlation.where(~np.eye(len(correlation), dtype=bool))
        average_corr = off_diagonal.stack().mean()
        st.caption(
            f"ค่าสหสัมพันธ์เฉลี่ย **{average_corr:.2f}** "
            + (
                "— สูงมาก สินทรัพย์เคลื่อนไหวไปทางเดียวกันเกือบทั้งหมด "
                "การถือหลายตัวจึงกระจายความเสี่ยงได้น้อยกว่าที่คิด"
                if average_corr > 0.7 else
                "— กระจายตัวได้ดี สินทรัพย์ไม่ได้เคลื่อนไหวตามกันทั้งหมด"
                if average_corr < 0.4 else
                "— กระจายตัวปานกลาง"
            )
        )

    # ════════════════════════════════════════
    # Tab 2: Optimal Weights + Custom Sliders
    # ════════════════════════════════════════
    with tab2:
        # Display only. Which weights actually get backtested is chosen
        # on the Backtesting tab, so that choice cannot depend on a bare
        # local defined over here.
        display_strategy = st.radio(
            "Optimization Strategy",
            OBJECTIVES,
            horizontal=True,
        )

        if display_strategy == "Max Sharpe":
            sel_weights = cleaned
            sel_ret, sel_vol, sel_sharpe = opt_ret, opt_vol, opt_sharpe
        else:
            sel_weights = mv_cleaned
            sel_ret, sel_vol, sel_sharpe = mv_ret, mv_vol, mv_sharpe

        st.subheader(f"{display_strategy} Optimal Weights")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Expected Annual Return", f"{sel_ret:.2%}")
        col_m2.metric("Annual Volatility", f"{sel_vol:.2%}")
        col_m3.metric("Sharpe Ratio", f"{sel_sharpe:.2f}")

        weights_df = pd.DataFrame({
            "Stock": list(sel_weights.keys()),
            "Weight": [f"{v:.1%}" for v in sel_weights.values()],
        })
        st.dataframe(weights_df, use_container_width=True, hide_index=True)


        # ── Walk-forward weight history ──
        if walk_result is not None:
            st.subheader("น้ำหนักที่คำนวณใหม่ในแต่ละงวด")
            history = pd.DataFrame(
                [w for _, w in walk_result.weight_history],
                index=[d.date() for d, _ in walk_result.weight_history],
            ).fillna(0.0)
            st.dataframe(
                history.style.format("{:.1%}"), use_container_width=True
            )
            st.caption(
                f"Turnover เฉลี่ยต่อการคำนวณใหม่ 1 ครั้ง: "
                f"**{walk_result.turnover.mean():.0%}** ของมูลค่าพอร์ต"
            )

        # ── Train vs Test ──
        # The gap between the two columns is the overfitting, made visible.
        if split_date is not None:
            train_result = metrics.simulate_portfolio(
                train_prices, backtest_w, rebalance_freq, cost_bps
            )
            if not train_result.returns.empty:
                train_stats = metrics.backtest_stats(train_result.returns, risk_free_rate)
                st.subheader("Train vs Test")
                comparison = pd.DataFrame({
                    "": ["Annual Return", "Annual Volatility", "Sharpe Ratio",
                         "Max Drawdown", "Total Years"],
                    "Train (in-sample)": [
                        f"{train_stats['annual_return']:.2%}",
                        f"{train_stats['annual_volatility']:.2%}",
                        f"{train_stats['sharpe']:.2f}",
                        f"{train_stats['max_drawdown']:.2%}",
                        f"{train_stats['years']:.1f}",
                    ],
                    "Test (out-of-sample)": [
                        f"{ann_ret:.2%}", f"{ann_vol:.2%}", f"{sharpe:.2f}",
                        f"{max_dd:.2%}", f"{n_years:.1f}",
                    ],
                })
                st.dataframe(comparison, use_container_width=True, hide_index=True)
                decay = train_stats["sharpe"] - sharpe
                if decay > 0.5:
                    st.warning(
                        f"⚠️ Sharpe ตกจาก {train_stats['sharpe']:.2f} เหลือ {sharpe:.2f} "
                        f"(−{decay:.2f}) — น้ำหนักชุดนี้ fit กับอดีตมากกว่าที่จะใช้ได้จริง"
                    )

        st.divider()
        st.subheader("น้ำหนักที่กำหนดเอง")
        st.caption(
            "ปรับน้ำหนักเองได้ ระบบจะ normalize ให้รวมเป็น 100% — "
            "ค่าเหล่านี้จะถูกใช้ก็ต่อเมื่อเลือก **Custom** เป็นแหล่งน้ำหนักด้านบนสุดของหน้า"
        )

        assets = sorted(ar.index)
        preset_cols = st.columns(3)
        if preset_cols[0].button("⚖️ เท่ากันทุกตัว", use_container_width=True):
            for sym in assets:
                st.session_state[f"cw_{sym}"] = round(100 / len(assets), 1)
            st.rerun()
        if preset_cols[1].button("↩️ กลับไปใช้ Max Sharpe", use_container_width=True):
            for sym in assets:
                st.session_state[f"cw_{sym}"] = round(cleaned.get(sym, 0.0) * 100, 1)
            st.rerun()
        if preset_cols[2].button("↩️ กลับไปใช้ Min Volatility", use_container_width=True):
            for sym in assets:
                st.session_state[f"cw_{sym}"] = round(mv_cleaned.get(sym, 0.0) * 100, 1)
            st.rerun()

        # A fixed four-column grid of sliders became unreadable past a
        # handful of holdings; number inputs stay one row per asset and
        # accept an exact figure.
        for sym in assets:
            if f"cw_{sym}" not in st.session_state:
                st.session_state[f"cw_{sym}"] = round(cleaned.get(sym, 0.0) * 100, 1)
        rows = st.columns(2)
        for position, sym in enumerate(assets):
            with rows[position % 2]:
                st.number_input(
                    sym, min_value=0.0, max_value=100.0, step=1.0,
                    format="%.1f", key=f"cw_{sym}",
                )

        custom_w_norm = read_custom_weights(assets)
        raw_total = sum(st.session_state[f"cw_{sym}"] for sym in assets)
        if abs(raw_total - 100.0) > 0.05:
            st.caption(
                f"รวมกันได้ {raw_total:.1f}% — จะถูก normalize เป็น 100% ให้อัตโนมัติ"
            )

        custom_ret, custom_vol, custom_sharpe = optimizer.portfolio_performance(
            ar, covr, custom_w_norm, risk_free_rate
        )
        custom_ret, custom_vol = metrics.blend_performance(
            custom_ret, custom_vol, risk_free_rate, cash_fraction
        )
        if cash_fraction > 0:
            st.caption(
                f"ตัวเลขด้านล่างรวมเงินสด {cash_fraction:.0%} ไว้แล้ว "
                "จึงตรงกับผลในแท็บทดสอบย้อนหลัง"
            )
        st.markdown("**ผลลัพธ์คาดหวังของน้ำหนักที่กำหนดเอง (หลัง normalize):**")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("ผลตอบแทนคาดหวังต่อปี", f"{custom_ret:.2%}")
        cc2.metric("ความผันผวนต่อปี", f"{custom_vol:.2%}")
        cc3.metric(
            "Sharpe Ratio", f"{custom_sharpe:.2f}",
            delta=f"{custom_sharpe - opt_sharpe:+.2f} vs Max Sharpe",
        )

        st.dataframe(
            pd.DataFrame({
                "สินทรัพย์": assets,
                "ที่กรอก (%)": [f"{st.session_state[f'cw_{a}']:.1f}" for a in assets],
                "หลัง normalize": [f"{custom_w_norm[a]:.1%}" for a in assets],
                "Max Sharpe": [f"{cleaned.get(a, 0.0):.1%}" for a in assets],
                "Min Volatility": [f"{mv_cleaned.get(a, 0.0):.1%}" for a in assets],
            }),
            use_container_width=True, hide_index=True,
        )

    # ════════════════════════════════════════
    # Tab 3: Backtesting
    # ════════════════════════════════════════
    with tab3:
        # ── Performance Stats ──
        # Four headline figures answer the question people actually ask:
        # what did it make, how badly did it hurt, was that worth the
        # risk, and did it beat just buying the benchmark. Everything
        # else is diagnostic and folds away.
        st.subheader("สรุปผลงาน")
        st.download_button(
            "📥 ดาวน์โหลดผลลัพธ์เป็น Excel",
            data=build_workbook(),
            file_name="portfolio_backtest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "ผลตอบแทนต่อปี", f"{ann_ret:.2%}",
            delta=(f"{ann_ret - bench_stats['annual_return']:+.2%} vs {benchmark_symbol}"
                   if bench_stats else None),
            help="ผลตอบแทนทบต้นเฉลี่ยต่อปี (CAGR) คำนวณจากช่วงเวลาปฏิทินจริง",
        )
        m2.metric(
            "ขาดทุนสูงสุด", f"{max_dd:.2%}",
            delta=(f"{max_dd - bench_stats['max_drawdown']:+.2%} vs {benchmark_symbol}"
                   if bench_stats else None),
            delta_color="normal",
            help="Max Drawdown — ระยะที่พอร์ตตกจากจุดสูงสุดลงมาต่ำสุด ยิ่งใกล้ 0 ยิ่งดี",
        )
        m3.metric(
            "Sharpe Ratio", f"{sharpe:.2f}",
            delta=(f"{sharpe - bench_stats['sharpe']:+.2f} vs {benchmark_symbol}"
                   if bench_stats else None),
            help=(
                "ผลตอบแทนส่วนเกินต่อ 1 หน่วยความผันผวน ยิ่งสูงยิ่งดี "
                "ต่ำกว่า 0.5 ถือว่าน้อย, 1.0 ขึ้นไปถือว่าดี"
            ),
        )
        if bench_stats:
            m4.metric(
                f"ชนะ {benchmark_symbol}", f"{ann_ret - bench_stats['annual_return']:+.2%}",
                help=f"ส่วนต่างผลตอบแทนต่อปีเทียบกับการถือ {benchmark_symbol} เฉยๆ",
            )
        else:
            m4.metric(
                "ความผันผวนต่อปี", f"{ann_vol:.2%}",
                help="ส่วนเบี่ยงเบนมาตรฐานของผลตอบแทน ปรับเป็นรายปี",
            )

        with st.expander("สถิติเพิ่มเติม"):
            e1, e2, e3, e4 = st.columns(4)
            e1.metric(
                "ความผันผวนต่อปี", f"{ann_vol:.2%}",
                help="ส่วนเบี่ยงเบนมาตรฐานของผลตอบแทน ปรับเป็นรายปี",
            )
            e2.metric(
                "ผลตอบแทนสะสม", f"{total_ret:.2%}",
                help="ผลตอบแทนรวมตลอดช่วงที่ทดสอบ ไม่ได้เฉลี่ยต่อปี",
            )
            e3.metric(
                "Calmar Ratio", f"{calmar:.2f}",
                help="ผลตอบแทนต่อปี หารด้วยขาดทุนสูงสุด — วัดว่าคุ้มกับการขาดทุนที่ต้องทนไหม",
            )
            e4.metric(
                "Sortino Ratio", f"{sortino:.2f}",
                help=(
                    "คล้าย Sharpe แต่นับเฉพาะความผันผวนขาลง "
                    "จึงไม่ลงโทษพอร์ตที่ผันผวนขึ้นแรง"
                ),
            )
            e5, e6, e7, e8 = st.columns(4)
            e5.metric(
                "ระยะเวลาที่ทดสอบ", f"{n_years:.1f} ปี",
                help="นับจากปฏิทินจริง ไม่ใช่จำนวนวันทำการหารด้วย 252",
            )
            if bench_stats:
                e6.metric(
                    "Beta", f"{beta:.2f}",
                    help=(
                        f"ความอ่อนไหวต่อ {benchmark_symbol} — 1.0 คือเคลื่อนไหวตามกัน "
                        "มากกว่า 1 คือแกว่งแรงกว่า"
                    ),
                )
                e7.metric(
                    "Alpha (ต่อปี)", f"{alpha:.2%}",
                    help="ผลตอบแทนส่วนเกินหลังปรับความเสี่ยงตาม beta แล้ว — บวกคือเก่งกว่าตลาด",
                )
                e8.metric(
                    f"Sharpe ของ {benchmark_symbol}", f"{bench_stats['sharpe']:.2f}",
                    help="ไว้เทียบกับ Sharpe ของพอร์ต",
                )

        if bench_stats is not None:
            st.subheader(f"เทียบกับ Benchmark: {benchmark_symbol}")
            versus = pd.DataFrame({
                "": ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown"],
                "Portfolio": [
                    f"{ann_ret:.2%}", f"{ann_vol:.2%}", f"{sharpe:.2f}", f"{max_dd:.2%}",
                ],
                benchmark_symbol: [
                    f"{bench_stats['annual_return']:.2%}",
                    f"{bench_stats['annual_volatility']:.2%}",
                    f"{bench_stats['sharpe']:.2f}",
                    f"{bench_stats['max_drawdown']:.2%}",
                ],
            })
            st.dataframe(versus, use_container_width=True, hide_index=True)

            b1, b2, b3 = st.columns(3)
            b1.metric("Beta", f"{beta:.2f}", help="ความอ่อนไหวต่อ benchmark — 1.0 คือเคลื่อนไหวตามกัน")
            b2.metric("Alpha (ต่อปี)", f"{alpha:.2%}", help="ผลตอบแทนส่วนเกินหลังปรับความเสี่ยงตาม beta")
            b3.metric("ชนะ Benchmark", f"{ann_ret - bench_stats['annual_return']:+.2%}")

            if ann_ret <= bench_stats["annual_return"]:
                st.info(
                    f"ℹ️ พอร์ตนี้ให้ผลตอบแทนไม่ชนะการถือ **{benchmark_symbol}** เฉยๆ "
                    "ในช่วงที่ทดสอบ — ลองพิจารณาว่าความซับซ้อนที่เพิ่มขึ้นคุ้มหรือไม่"
                )

        if bench_stats is not None:
            st.subheader(f"เทียบกับ Benchmark: {benchmark_symbol}")
            versus = pd.DataFrame({
                "": ["Annual Return", "Annual Volatility", "Sharpe Ratio", "Max Drawdown"],
                "Portfolio": [
                    f"{ann_ret:.2%}", f"{ann_vol:.2%}", f"{sharpe:.2f}", f"{max_dd:.2%}",
                ],
                benchmark_symbol: [
                    f"{bench_stats['annual_return']:.2%}",
                    f"{bench_stats['annual_volatility']:.2%}",
                    f"{bench_stats['sharpe']:.2f}",
                    f"{bench_stats['max_drawdown']:.2%}",
                ],
            })
            st.dataframe(versus, use_container_width=True, hide_index=True)

            b1, b2, b3 = st.columns(3)
            b1.metric("Beta", f"{beta:.2f}", help="ความอ่อนไหวต่อ benchmark — 1.0 คือเคลื่อนไหวตามกัน")
            b2.metric("Alpha (ต่อปี)", f"{alpha:.2%}", help="ผลตอบแทนส่วนเกินหลังปรับความเสี่ยงตาม beta")
            b3.metric("ชนะ Benchmark", f"{ann_ret - bench_stats['annual_return']:+.2%}")

            if ann_ret <= bench_stats["annual_return"]:
                st.info(
                    f"ℹ️ พอร์ตนี้ให้ผลตอบแทนไม่ชนะการถือ **{benchmark_symbol}** เฉยๆ "
                    "ในช่วงที่ทดสอบ — ลองพิจารณาว่าความซับซ้อนที่เพิ่มขึ้นคุ้มหรือไม่"
                )

        # ── Rebalancing ──
        st.subheader("การ Rebalance")
        r1, r2, r3 = st.columns(3)
        r1.metric("ความถี่", rebalance_label)
        r2.metric("จำนวนครั้ง", f"{len(result.rebalances):,}")
        turnover_series = (
            walk_result.turnover if walk_result is not None else result.turnover
        )
        r3.metric(
            "Turnover รวม", f"{turnover_series.sum():.0%}",
            help="มูลค่าที่ซื้อขายรวมทั้งช่วง คิดเป็นสัดส่วนของมูลค่าพอร์ต",
        )
        if cost_bps > 0:
            gross = metrics.simulate_portfolio(test_prices, backtest_w, rebalance_freq, 0.0)
            gross_stats = metrics.backtest_stats(gross.returns, risk_free_rate)
            st.caption(
                f"ค่าธรรมเนียม {cost_bps:.0f} bps กินผลตอบแทนต่อปีไป "
                f"**{gross_stats['annual_return'] - ann_ret:.2%}** "
                f"(ก่อนหักค่าธรรมเนียม {gross_stats['annual_return']:.2%} → หลังหัก {ann_ret:.2%})"
            )
        if rebalance_freq is None and result.final_weights:
            drift = ", ".join(
                f"{sym} {active_w.get(sym, 0):.0%}→{final:.0%}"
                for sym, final in sorted(result.final_weights.items())
            )
            st.caption(f"สัดส่วนเมื่อจบช่วง (ปล่อยให้ drift): {drift}")

        # ── Cumulative Returns Chart ──
        st.subheader("การเติบโตของพอร์ต")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cumulative.index, y=cumulative.values,
            mode="lines", name="Portfolio",
            line=dict(color="#2196F3"),
        ))
        if not bench_daily.empty:
            fig_cum.add_trace(go.Scatter(
                x=bench_daily.index, y=(1 + bench_daily).cumprod().values,
                mode="lines", name=benchmark_symbol,
                line=dict(color="#9E9E9E", width=1.5, dash="dash"),
            ))
        fig_cum.update_layout(
            yaxis_title="Growth of 1 unit",
            height=400,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Drawdown Chart ──
        st.subheader("การขาดทุนจากจุดสูงสุด")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            mode="lines", fill="tozeroy",
            name="Drawdown", line=dict(color="#F44336"),
        ))
        fig_dd.update_layout(
            yaxis_title="Drawdown",
            yaxis_tickformat=".0%",
            height=350,
        )
        st.plotly_chart(fig_dd, use_container_width=True)

        # ── Monthly Returns Heatmap ──
        monthly = port_daily.resample("ME").apply(lambda x: (1 + x).prod() - 1)
        monthly_pivot = pd.DataFrame({
            "Year": monthly.index.year,
            "Month": monthly.index.month,
            "Return": monthly.values,
        }).pivot(index="Year", columns="Month", values="Return")
        monthly_pivot.columns = metrics.month_labels(monthly_pivot.columns)

        fig_hm = px.imshow(
            monthly_pivot.values,
            x=monthly_pivot.columns.tolist(),
            y=monthly_pivot.index.tolist(),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            text_auto=".1%",
        )
        fig_hm.update_layout(height=max(300, len(monthly_pivot) * 30))
        with st.expander("ผลตอบแทนรายเดือน"):
            st.plotly_chart(fig_hm, use_container_width=True)

        # ── Worst Drawdown Periods ──
        dd_series = drawdown.copy()
        periods = []
        for _ in range(5):
            if dd_series.empty or dd_series.min() == 0:
                break
            valley_idx = dd_series.idxmin()
            valley_val = dd_series.loc[valley_idx]
            # find peak before valley
            peak_idx = cumulative.loc[:valley_idx].idxmax()
            # find recovery after valley
            recovery_mask = cumulative.loc[valley_idx:] >= cumulative.loc[peak_idx]
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
                duration = (recovery_idx - peak_idx).days
            else:
                recovery_idx = "Not recovered"
                duration = (cumulative.index[-1] - peak_idx).days
            periods.append({
                "Drawdown": f"{valley_val:.2%}",
                "Peak": str(peak_idx.date()) if hasattr(peak_idx, 'date') else str(peak_idx),
                "Valley": str(valley_idx.date()) if hasattr(valley_idx, 'date') else str(valley_idx),
                "Recovery": str(recovery_idx.date()) if hasattr(recovery_idx, 'date') else str(recovery_idx),
                "Duration (days)": duration,
            })
            # mask out this drawdown period
            if isinstance(recovery_idx, str):
                dd_series.loc[peak_idx:] = 0
            else:
                dd_series.loc[peak_idx:recovery_idx] = 0

        if periods:
            with st.expander("ช่วงที่ขาดทุนหนักที่สุด"):
                st.dataframe(pd.DataFrame(periods), use_container_width=True, hide_index=True)
                st.caption(
                    "Peak คือจุดสูงสุดก่อนตก · Valley คือจุดต่ำสุด · "
                    "Recovery คือวันที่กลับมาเท่าจุดสูงสุดเดิม"
                )

    # ════════════════════════════════════════
    # Tab 4: NAV Breakdown
    # ════════════════════════════════════════
    with tab4:
        # Render exactly what the Backtesting tab computed. Recomputing
        # here is what let the two tabs drift apart -- most visibly under
        # walk-forward, where this tab had no idea the weights changed
        # every period.
        view = st.session_state.get("nav_view")
        if view is None or view["returns"].empty:
            st.info("เปิดแท็บทดสอบย้อนหลังก่อนหนึ่งครั้ง เพื่อให้คำนวณผลลัพธ์")
            st.stop()

        port_daily = view["returns"]
        daily_returns = view["asset_returns"]
        nav_w = view["weights"]
        nav_total = total_cash * (1 + port_daily).cumprod()

        if view["is_walk_forward"]:
            st.info(
                f"โหมด Walk-Forward — เส้น Portfolio NAV มาจากน้ำหนักที่คำนวณใหม่ทุกงวด "
                f"(วัตถุประสงค์ {view['source']}) ส่วนเส้นรายตัวใช้น้ำหนักของงวดล่าสุด"
            )
        else:
            st.info(f"ใช้น้ำหนัก **{view['source']}** (ตรงกับแท็บทดสอบย้อนหลัง)")

        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(
            x=nav_total.index, y=nav_total.values,
            mode="lines", name="Portfolio NAV",
            line=dict(color="#4CAF50", width=2),
        ))

        # Each sleeve held on its own, for comparison against the
        # rebalanced portfolio line above.
        for sym in sorted(view["held"]):
            if sym not in daily_returns.columns:
                continue
            stock_nav = total_cash * nav_w.get(sym, 0.0) * (1 + daily_returns[sym]).cumprod()
            fig_nav.add_trace(go.Scatter(
                x=stock_nav.index, y=stock_nav.values,
                mode="lines", name=f"{sym} (ถือเดี่ยว)",
                line=dict(width=1),
                opacity=0.7,
            ))

        fig_nav.update_layout(
            title=f"NAV (เริ่มต้น {total_cash:,.0f} {base_currency})",
            yaxis_title=f"NAV ({base_currency})",
            yaxis_tickformat=",.0f",
            height=600,
        )
        st.plotly_chart(fig_nav, use_container_width=True)

        st.caption(
            f"เริ่มนับตั้งแต่ **{view['start'].date()}** ซึ่งเป็นวันแรกที่ทุกตัวในพอร์ตมีข้อมูลครบ "
            f"(ตรงกับแท็บทดสอบย้อนหลัง) — เส้น Portfolio NAV ปรับสัดส่วนกลับ{rebalance_label} "
            "ส่วนเส้นรายตัวคือถือเดี่ยวไม่ปรับสัดส่วน จึงบวกกันแล้วไม่เท่ากับเส้นรวม"
        )

        st.metric("Final NAV", f"{nav_total.iloc[-1]:,.0f} {base_currency}")
        st.metric("Total P&L", f"{nav_total.iloc[-1] - total_cash:,.0f} {base_currency}")

else:
    st.subheader("เครื่องมือจัดพอร์ตและทดสอบย้อนหลัง")
    st.markdown(
        "หาสัดส่วนการลงทุนที่ให้ผลตอบแทนดีที่สุดต่อความเสี่ยงที่รับได้ "
        "แล้ว**ทดสอบกับข้อมูลจริงในอดีตที่ระบบไม่เคยเห็น** "
        "รองรับหุ้นและ ETF ทั่วโลก กองทุนรวมไทยจาก SEC และไฟล์ราคาที่คุณมีเอง"
    )

    st.markdown("#### เริ่มจากตัวอย่าง")
    st.caption("กดเลือกหนึ่งชุด แล้วกด **Calculate** ในแถบด้านซ้าย")
    preset_columns = st.columns(len(PRESETS))
    for column, (name, (symbols, description)) in zip(preset_columns, PRESETS.items()):
        with column:
            if st.button(name, use_container_width=True):
                st.session_state["pending_symbols"] = symbols
                st.rerun()
            st.caption(description)
            st.code(symbols, language=None)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### จะได้อะไรกลับมา")
        st.markdown(
            "- **น้ำหนักที่เหมาะสม** ของแต่ละสินทรัพย์ ทั้งแบบ Max Sharpe และ Min Volatility\n"
            "- **ผลทดสอบย้อนหลัง** พร้อมผลตอบแทน ความผันผวน และช่วงขาดทุนหนักที่สุด\n"
            "- **เทียบกับ benchmark** ว่าชนะการถือ index เฉยๆ หรือไม่\n"
            "- **กราฟมูลค่าพอร์ต** และดาวน์โหลดผลทั้งหมดเป็น Excel"
        )
    with right:
        st.markdown("#### สิ่งที่ควรรู้ก่อนเชื่อตัวเลข")
        st.markdown(
            "- ค่าเริ่มต้นใช้โหมด **Train/Test** คือหาน้ำหนักจากช่วงแรก "
            "แล้ววัดผลบนช่วงหลังที่ไม่เคยเห็น ตัวเลขจึงไม่สวยเท่าโหมด In-sample "
            "แต่เป็นตัวเลขที่เชื่อได้จริง\n"
            "- ผลตอบแทนในอดีตไม่รับประกันอนาคต\n"
            "- ทุกสินทรัพย์ถูกแปลงเป็นสกุลเงินฐานก่อนคำนวณ ผลของค่าเงินจึงถูกนับรวมแล้ว"
        )
