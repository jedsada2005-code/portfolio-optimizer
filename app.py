import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from pypfopt import EfficientFrontier, CLA, plotting
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
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
        start_date = st.date_input("Start Date", value=pd.Timestamp("2010-01-01"))
    with col2:
        end_date = st.date_input("End Date", value=pd.Timestamp("2022-01-01"))

    total_cash = st.number_input("Total Cash (USD)", value=1_000_000, step=100_000)
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, step=0.01, format="%.4f")
    run_btn = st.button("Calculate", type="primary", use_container_width=True)

    st.divider()
    st.caption("⚠️ **Beta Version — ข้อควรระวัง**")
    st.caption("1. รองรับเฉพาะสินทรัพย์ที่มีใน Yahoo Finance เท่านั้น")
    st.caption("2. หุ้นไทยต้องเติม `.BK` หลังชื่อ เช่น `PTT.BK` หุ้น US ใส่ชื่อได้เลย")
    st.caption("3. Custom Weight รวมกันต้องเท่ากับ 1.0 เท่านั้น")
    st.caption("4. ตัวอย่าง: `AMZN, META, NVDA, SPY, LLY`")

# ─── Parse symbols ───
stock_list = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

if run_btn and len(stock_list) >= 2:
    with st.spinner("Downloading data..."):
        df = yf.download(
            tickers=stock_list,
            start=str(start_date),
            end=str(end_date),
            interval="1d",
            auto_adjust=True,
        )
        data_close = df["Close"].ffill()
        data_close.dropna(inplace=True)

    if data_close.empty:
        st.error("No data downloaded. Check symbols and date range.")
        st.stop()

    # ─── Calculations ───
    with st.spinner("Computing efficient frontier..."):
        weekly = data_close.resample("W-FRI").last()
        ar = weekly.pct_change(52).mean()
        covr = weekly.pct_change().cov() * 52

        # Random portfolios
        n_samples = 50_000
        w = np.random.dirichlet([0.5] * len(ar), n_samples)
        rets = w.dot(ar)
        stds = np.sqrt((w.T * (covr.values @ w.T)).sum(axis=0))
        sharpes = rets / stds

        # Max Sharpe weights
        ef = EfficientFrontier(ar, covr)
        raw_weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned = dict(raw_weights)
        perf = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        opt_ret, opt_vol, opt_sharpe = perf

        # Min Volatility weights
        ef_mv = EfficientFrontier(ar, covr)
        mv_weights = ef_mv.min_volatility()
        mv_cleaned = dict(mv_weights)
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
    st.session_state["stock_list"] = stock_list
    st.session_state["total_cash"] = total_cash
    st.session_state["risk_free_rate"] = risk_free_rate
    st.session_state["calculated"] = True

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
        weights_s = pd.Series(active_w)

        st.info(f"Backtesting with weights: {', '.join(f'{k}={v:.1%}' for k, v in active_w.items())}")

        daily_returns = data_close.pct_change().fillna(0)
        port_daily = daily_returns.dot(weights_s.reindex(daily_returns.columns, fill_value=0))

        # Remove leading zeros
        first_valid = port_daily.ne(0).idxmax()
        port_daily = port_daily.loc[first_valid:]

        cumulative = (1 + port_daily).cumprod()
        trading_days = 252

        # ── Performance Stats ──
        total_ret = cumulative.iloc[-1] - 1
        n_years = len(port_daily) / trading_days
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1
        ann_vol = port_daily.std() * np.sqrt(trading_days)
        sharpe = (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0 else 0

        # Drawdown
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # Sortino
        downside = port_daily[port_daily < 0]
        downside_std = downside.std() * np.sqrt(trading_days)
        sortino = (ann_ret - risk_free_rate) / downside_std if downside_std > 0 else 0

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
        monthly_pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ][:len(monthly_pivot.columns)]

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
        weights_s = pd.Series(active_w)

        daily_returns = data_close.pct_change().fillna(0)
        port_daily = daily_returns.dot(weights_s.reindex(daily_returns.columns, fill_value=0))

        nav_total = total_cash * (1 + port_daily).cumprod()

        fig_nav = go.Figure()
        fig_nav.add_trace(go.Scatter(
            x=nav_total.index, y=nav_total.values,
            mode="lines", name="Portfolio NAV",
            line=dict(color="#4CAF50", width=2),
        ))

        # Each stock NAV (rebalanced)
        for sym in sorted(active_w.keys()):
            if active_w[sym] > 0 and sym in daily_returns.columns:
                stock_nav = total_cash * active_w[sym] * (1 + daily_returns[sym]).cumprod()
                fig_nav.add_trace(go.Scatter(
                    x=stock_nav.index, y=stock_nav.values,
                    mode="lines", name=sym,
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

        st.metric("Final NAV", f"${nav_total.iloc[-1]:,.0f}")
        st.metric("Total P&L", f"${nav_total.iloc[-1] - total_cash:,.0f}")

else:
    st.info("Enter stock symbols and click **Calculate** to begin.")
