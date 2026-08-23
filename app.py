import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from pypfopt import EfficientFrontier, CLA, plotting
from pypfopt.exceptions import OptimizationError
import matplotlib.pyplot as plt

import custom_data
import metrics
import thai_mf

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# Random portfolios are sampled at n_samples but only PLOT_SAMPLE of
# them are drawn, which is what keeps the frontier tab responsive.
PLOT_SAMPLE = 6_000


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
    symbols_input = st.text_input(
        "Stock Symbols (comma-separated)",
        value="AMZN, META, LLY, SPY, NVDA, GOOGL",
    )
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=pd.Timestamp("2010-01-01"),
            min_value=pd.Timestamp("1990-01-01"),
            max_value=pd.Timestamp.today(),
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=pd.Timestamp.today(),
            min_value=pd.Timestamp("1990-01-01"),
            max_value=pd.Timestamp.today(),
        )

    total_cash = st.number_input("Total Cash (USD)", value=1_000_000, step=100_000)
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, step=0.01, format="%.4f")
    uploaded_price_files = st.file_uploader(
        "Upload CSV/XLSX Price Data",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        help=(
            "รองรับ Date,AAPL,SPY... หรือ Date,Symbol,Close หรือ Date,Close "
            "(กรณี Date,Close จะใช้ชื่อไฟล์หรือชื่อ sheet เป็นชื่อสินทรัพย์)"
        ),
    )
    sec_factsheet_key = st.text_input(
        "SEC Fund Factsheet API Key (สำหรับกองทุนไทย)",
        value="",
        type="password",
        help="จำเป็นเฉพาะเมื่อกรอกกองทุนรวมไทยด้วย prefix MF: สมัครฟรีที่ secopendata.sec.or.th",
    )
    sec_daily_info_key = st.text_input(
        "SEC Fund Daily Info API Key (สำหรับกองทุนไทย)",
        value="",
        type="password",
        help="เป็น API key คนละตัวกับ Fund Factsheet ต้อง subscribe แยกกันที่ secopendata.sec.or.th",
    )
    run_btn = st.button("Calculate", type="primary", use_container_width=True)

    st.divider()
    st.caption("⚠️ **Beta Version — ข้อควรระวัง**")
    st.caption("1. รองรับหุ้น/ETF จาก Yahoo Finance, กองทุนรวมไทยจาก SEC (prefix `MF:`) และไฟล์ราคาที่อัปโหลดเอง (CSV/XLSX)")
    st.caption("2. หุ้นไทยต้องเติม `.BK` หลังชื่อ เช่น `PTT.BK` หุ้น US ใส่ชื่อได้เลย")
    st.caption("3. Custom Weight ไม่ต้องรวมกันเป็น 1.0 ระบบจะปรับสัดส่วน (normalize) ให้อัตโนมัติ")
    st.caption("4. ตัวอย่าง: `AMZN, META, NVDA, SPY, LLY`")
    st.caption("5. ค่า Return, Vol, Sharpe ใน Expected กับ Backtest มีค่าใกล้เคียงกันแต่อาจต่างกันเล็กน้อย เนื่องจากคำนวณคนละวิธี")
    st.caption("6. หุ้นบางตัวอาจโหลดไม่สำเร็จ เพราะเขียนชื่อผิด หรือในปีนั้นยังไม่มีข้อมูล (ตรวจสอบชื่อและปีที่ดึงข้อมูลให้ดี)")
    st.caption("7. กองทุนรวมไทยใส่ prefix `MF:` เช่น `MF:K-CHANGE-A(A)` ข้อมูลมาจาก SEC Open Data และต้องกรอก API Key ทั้ง 2 ช่องด้านบน (Fund Factsheet กับ Fund Daily Info เป็นคนละ key กัน ต้อง subscribe แยกกัน)")
    st.caption("8. CSV/XLSX ต้องเป็นราคาหรือ NAV ไม่ใช่ daily return และต้องมีคอลัมน์วันที่ เช่น `Date`")

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
            for name in mf_symbols:
                display_symbol = f"MF:{name}"
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
        weekly = data_close.resample("W-FRI").last()
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
            weekly = weekly.drop(columns=drop)
            ar = ar.drop(index=drop)

        if len(ar) < 2:
            st.error(
                "เหลือสินทรัพย์ที่มีข้อมูลเพียงพอน้อยกว่า 2 ตัว ไม่สามารถคำนวณพอร์ตได้ "
                "ลองขยายช่วงวันที่ หรือลดกองทุนที่เพิ่งจดทะเบียนออก"
            )
            st.stop()

        covr = weekly.pct_change().cov() * 52

        # Random portfolios
        n_samples = 200_000
        rng = np.random.default_rng(42)
        w = rng.dirichlet([0.5] * len(ar), n_samples)
        rets = w.dot(ar)
        stds = np.sqrt((w.T * (covr.values @ w.T)).sum(axis=0))
        sharpes = (rets - risk_free_rate) / stds

        # Plot a subsample: 200k markers is what makes this tab crawl,
        # and the cloud looks identical at a few thousand points.
        plot_n = min(PLOT_SAMPLE, n_samples)
        pick = rng.choice(n_samples, plot_n, replace=False)
        stds, rets, sharpes = stds[pick], rets[pick], sharpes[pick]

        # Max Sharpe weights
        ef = EfficientFrontier(ar, covr)
        try:
            ef.max_sharpe(risk_free_rate=risk_free_rate)
        except (ValueError, OptimizationError):
            st.error(
                "⚠️ หาพอร์ต Max Sharpe ไม่ได้ — สินทรัพย์ที่เลือกมีผลตอบแทนคาดหวังใกล้เคียงหรือต่ำกว่า "
                f"Risk-Free Rate ที่ตั้งไว้ ({risk_free_rate:.2%}) เกินไป "
                "ลองลด Risk-Free Rate ลง หรือเพิ่มสินทรัพย์ที่ผลตอบแทนสูงกว่าเข้าไปในพอร์ต"
            )
            st.stop()
        cleaned = dict(ef.clean_weights())
        perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        opt_ret, opt_vol, opt_sharpe = perf

        # Min Volatility weights
        ef_mv = EfficientFrontier(ar, covr)
        ef_mv.min_volatility()
        mv_cleaned = dict(ef_mv.clean_weights())
        mv_perf = ef_mv.portfolio_performance(risk_free_rate=risk_free_rate)
        mv_ret, mv_vol, mv_sharpe = mv_perf

        # Efficient frontier curve
        ef2 = EfficientFrontier(ar, covr)
        fig_mpl, ax_mpl = plt.subplots()
        plotting.plot_efficient_frontier(ef2, ax=ax_mpl, show_assets=False)
        # extract the line data
        ef_line = ax_mpl.get_lines()[0]
        ef_x = ef_line.get_xdata()
        ef_y = ef_line.get_ydata()
        plt.close(fig_mpl)

    # Store in session for tabs
    st.session_state["data_close"] = data_close
    st.session_state["ar"] = ar
    st.session_state["covr"] = covr
    st.session_state["cleaned"] = cleaned
    st.session_state["opt_perf"] = (opt_ret, opt_vol, opt_sharpe)
    st.session_state["mv_cleaned"] = mv_cleaned
    st.session_state["mv_perf"] = (mv_ret, mv_vol, mv_sharpe)
    st.session_state["random"] = (stds, rets, sharpes)
    st.session_state["ef_curve"] = (ef_x, ef_y)
    st.session_state["stock_list"] = list(data_close.columns)
    st.session_state["total_cash"] = total_cash
    st.session_state["risk_free_rate"] = risk_free_rate
    st.session_state["calculated"] = True
    st.session_state["input_signature"] = input_signature

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
    stock_list = st.session_state["stock_list"]
    total_cash = st.session_state["total_cash"]
    risk_free_rate = st.session_state["risk_free_rate"]

    if st.session_state.get("input_signature") != input_signature:
        st.warning(
            "⚠️ การตั้งค่าในแถบด้านซ้ายถูกแก้ไขหลังจากคำนวณครั้งล่าสุด — "
            "ผลลัพธ์ด้านล่างยังเป็นของค่าเดิม กด **Calculate** เพื่อคำนวณใหม่"
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Efficient Frontier",
        "Optimal Weights",
        "Backtesting",
        "NAV Breakdown",
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
        fig.add_trace(go.Scatter(
            x=ef_x, y=ef_y, mode="lines",
            line=dict(color="red", width=2),
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

    # ════════════════════════════════════════
    # Tab 2: Optimal Weights + Custom Sliders
    # ════════════════════════════════════════
    with tab2:
        strategy = st.radio(
            "Optimization Strategy",
            ["Max Sharpe", "Min Volatility"],
            horizontal=True,
        )

        if strategy == "Max Sharpe":
            sel_weights = cleaned
            sel_ret, sel_vol, sel_sharpe = opt_ret, opt_vol, opt_sharpe
        else:
            sel_weights = mv_cleaned
            sel_ret, sel_vol, sel_sharpe = mv_ret, mv_vol, mv_sharpe

        st.subheader(f"{strategy} Optimal Weights")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Expected Annual Return", f"{sel_ret:.2%}")
        col_m2.metric("Annual Volatility", f"{sel_vol:.2%}")
        col_m3.metric("Sharpe Ratio", f"{sel_sharpe:.2f}")

        weights_df = pd.DataFrame({
            "Stock": list(sel_weights.keys()),
            "Weight": [f"{v:.1%}" for v in sel_weights.values()],
        })
        st.dataframe(weights_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Custom Weights")
        st.caption("Adjust weights manually. They will be normalized to sum to 1.0.")

        custom_w = {}
        cols = st.columns(min(len(stock_list), 4))
        for i, sym in enumerate(sorted(sel_weights.keys())):
            with cols[i % len(cols)]:
                default = sel_weights.get(sym, 0.0)
                custom_w[sym] = st.slider(
                    sym, 0.0, 1.0, float(round(default, 3)),
                    step=0.01, key=f"w_{sym}_{strategy}",
                )

        total_w = sum(custom_w.values())
        if total_w > 0:
            custom_w_norm = {k: v / total_w for k, v in custom_w.items()}
        else:
            custom_w_norm = {k: 1 / len(custom_w) for k in custom_w}

        # Compute custom portfolio performance
        w_arr = np.array([custom_w_norm[s] for s in ar.index])
        custom_ret = w_arr.dot(ar.values)
        custom_vol = np.sqrt(w_arr @ covr.values @ w_arr)
        custom_sharpe = (custom_ret - risk_free_rate) / custom_vol if custom_vol > 0 else 0

        st.markdown("**Custom Portfolio Performance (normalized):**")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Expected Annual Return", f"{custom_ret:.2%}")
        cc2.metric("Annual Volatility", f"{custom_vol:.2%}")
        cc3.metric("Sharpe Ratio", f"{custom_sharpe:.2f}")

        # Show normalized weights
        norm_df = pd.DataFrame({
            "Stock": list(custom_w_norm.keys()),
            "Raw": [f"{custom_w[k]:.2f}" for k in custom_w_norm],
            "Normalized": [f"{v:.1%}" for v in custom_w_norm.values()],
        })
        st.dataframe(norm_df, use_container_width=True, hide_index=True)

        # Store for backtesting
        st.session_state["active_weights"] = custom_w_norm

    # ════════════════════════════════════════
    # Tab 3: Backtesting
    # ════════════════════════════════════════
    with tab3:
        # Use custom weights if available, else optimal
        active_w = st.session_state.get("active_weights", cleaned)

        st.info(f"Backtesting with weights: {', '.join(f'{k}={v:.1%}' for k, v in active_w.items())}")

        result = metrics.portfolio_daily_returns(data_close, active_w)
        port_daily = result.portfolio
        daily_returns = result.assets

        if port_daily.empty:
            st.error(
                "⚠️ ช่วงเวลาของสินทรัพย์ในพอร์ตไม่ทับซ้อนกันเลย จึง backtest ไม่ได้ "
                "— ลองเอาสินทรัพย์ที่เพิ่งเริ่มมีข้อมูลออก หรือขยายช่วงวันที่"
            )
            st.stop()

        # The backtest can only begin once every holding actually exists.
        requested_start = pd.Timestamp(data_close.index[0])
        if result.start > requested_start:
            firsts = metrics.first_valid_dates(data_close[result.held])
            limiter = firsts.idxmax()
            st.warning(
                f"⚠️ Backtest เริ่มจริงที่ **{result.start.date()}** ไม่ใช่ "
                f"{requested_start.date()} เพราะ **{limiter}** เพิ่งมีข้อมูลวันแรกตอนนั้น "
                "— พอร์ตจะถือครบทุกตัวได้ก็ต่อเมื่อทุกตัวมีอยู่จริงแล้ว"
            )

        cumulative = (1 + port_daily).cumprod()
        # Derive both scalings from the index itself: merging a Thai fund
        # with US equities yields ~300 rows a year, not 252.
        periods_per_year = metrics.periods_per_year(port_daily.index)
        n_years = metrics.years_elapsed(port_daily.index)

        # ── Performance Stats ──
        total_ret = cumulative.iloc[-1] - 1
        ann_ret = metrics.cagr(total_ret, n_years)
        ann_vol = port_daily.std() * np.sqrt(periods_per_year)
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0

        # Drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # Sortino
        downside_std = metrics.downside_deviation(port_daily, periods_per_year)
        sortino = metrics.sortino_ratio(ann_ret, downside_std, risk_free_rate)

        # Display metrics
        st.subheader("Performance Summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Annual Return", f"{ann_ret:.2%}")
        m2.metric("Annual Volatility", f"{ann_vol:.2%}")
        m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
        m4.metric("Max Drawdown", f"{max_dd:.2%}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Cumulative Return", f"{total_ret:.2%}")
        m6.metric("Calmar Ratio", f"{calmar:.2f}")
        m7.metric("Sortino Ratio", f"{sortino:.2f}")
        m8.metric("Total Years", f"{n_years:.1f}")

        # ── Cumulative Returns Chart ──
        st.subheader("Cumulative Returns")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=cumulative.index, y=cumulative.values,
            mode="lines", name="Portfolio",
            line=dict(color="#2196F3"),
        ))
        fig_cum.update_layout(
            yaxis_title="Growth of $1",
            height=400,
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Drawdown Chart ──
        st.subheader("Drawdown")
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
        st.subheader("Monthly Returns")
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
        st.plotly_chart(fig_hm, use_container_width=True)

        # ── Worst Drawdown Periods ──
        st.subheader("Worst Drawdown Periods")
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
            st.dataframe(pd.DataFrame(periods), use_container_width=True, hide_index=True)

    # ════════════════════════════════════════
    # Tab 4: NAV Breakdown
    # ════════════════════════════════════════
    with tab4:
        active_w = st.session_state.get("active_weights", cleaned)

        # Share the backtest's exact series so both tabs start on the
        # same date and report the same growth.
        result = metrics.portfolio_daily_returns(data_close, active_w)
        port_daily = result.portfolio
        daily_returns = result.assets

        if port_daily.empty:
            st.error("⚠️ ช่วงเวลาของสินทรัพย์ในพอร์ตไม่ทับซ้อนกันเลย จึงคำนวณ NAV ไม่ได้")
            st.stop()

        nav_total = total_cash * (1 + port_daily).cumprod()

        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(
            x=nav_total.index, y=nav_total.values,
            mode="lines", name="Portfolio NAV",
            line=dict(color="#4CAF50", width=2),
        ))

        # Each sleeve held on its own, for comparison against the
        # rebalanced portfolio line above.
        for sym in sorted(result.held):
            stock_nav = total_cash * active_w[sym] * (1 + daily_returns[sym]).cumprod()
            fig_nav.add_trace(go.Scatter(
                x=stock_nav.index, y=stock_nav.values,
                mode="lines", name=f"{sym} (ถือเดี่ยว)",
                line=dict(width=1),
                opacity=0.7,
            ))

        fig_nav.update_layout(
            title=f"NAV (Starting ${total_cash:,.0f})",
            yaxis_title="NAV (USD)",
            yaxis_tickformat="$,.0f",
            height=600,
        )
        st.plotly_chart(fig_nav, use_container_width=True)

        st.caption(
            f"เริ่มนับตั้งแต่ **{result.start.date()}** ซึ่งเป็นวันแรกที่ทุกตัวในพอร์ตมีข้อมูลครบ "
            "(ตรงกับแท็บ Backtesting) — เส้น Portfolio NAV ปรับสัดส่วนกลับทุกวันทำการ "
            "ส่วนเส้นรายตัวคือถือเดี่ยวไม่ปรับสัดส่วน จึงบวกกันแล้วไม่เท่ากับเส้นรวม"
        )

        st.metric("Final NAV", f"${nav_total.iloc[-1]:,.0f}")
        st.metric("Total P&L", f"${nav_total.iloc[-1] - total_cash:,.0f}")

else:
    st.info("Enter stock symbols and click **Calculate** to begin.")
