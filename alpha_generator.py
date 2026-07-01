"""
WorldQuant Brain Alpha Generator
=================================
Generates Brain expression alphas across 6 technical categories.
Each alpha is a self-contained expression ready to paste into the simulator.

Usage:
    python alpha_generator.py                  # print all alphas
    python alpha_generator.py --category momentum
    python alpha_generator.py --category mean_reversion
    python alpha_generator.py --category volume
    python alpha_generator.py --category volatility
    python alpha_generator.py --category trend
    python alpha_generator.py --category combined
    python alpha_generator.py --list            # show alpha names only
"""

import argparse
from dataclasses import dataclass
from typing import List

# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Alpha:
    name: str
    category: str
    description: str
    expression: str          # ready-to-paste Brain expression
    suggested_settings: dict # region / universe / neutralization hints


# ─────────────────────────────────────────────────────────────────────────────
# 1. MOMENTUM ALPHAS
# ─────────────────────────────────────────────────────────────────────────────

MOMENTUM_ALPHAS: List[Alpha] = [

    Alpha(
        name="RSI_14",
        category="momentum",
        description="RSI(14): long when oversold <30, short when overbought >70",
        expression="""\
delta    = close - ts_delay(close, 1);
gain     = max(delta, 0);
loss     = max(-delta, 0);
avg_gain = ts_mean(gain, 14);
avg_loss = ts_mean(loss, 14);
rs       = avg_gain / (avg_loss + 0.0001);
rsi      = 100 - (100 / (1 + rs));
-1 * rank(rsi)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="RSI_signal",
        category="momentum",
        description="Binary RSI signal: +1 entry <30, -1 exit >70",
        expression="""\
delta    = close - ts_delay(close, 1);
gain     = max(delta, 0);
loss     = max(-delta, 0);
avg_gain = ts_mean(gain, 14);
avg_loss = ts_mean(loss, 14);
rs       = avg_gain / (avg_loss + 0.0001);
rsi      = 100 - (100 / (1 + rs));
(rsi < 30) - (rsi > 70)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Market"},
    ),

    Alpha(
        name="MACD_signal",
        category="momentum",
        description="MACD(12,26,9): signal line crossover",
        expression="""\
ema12   = ts_mean(close, 12);
ema26   = ts_mean(close, 26);
macd    = ema12 - ema26;
signal9 = ts_mean(macd, 9);
rank(macd - signal9)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="ROC_20",
        category="momentum",
        description="Rate of Change over 20 days — buy recent winners",
        expression="""\
roc = (close - ts_delay(close, 20)) / (ts_delay(close, 20) + 0.0001);
rank(roc)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Industry"},
    ),

    Alpha(
        name="Price_Momentum_12_1",
        category="momentum",
        description="12-month momentum skipping last 1 month (classic Jegadeesh-Titman)",
        expression="""\
ret_12 = (ts_delay(close, 1) - ts_delay(close, 252)) / (ts_delay(close, 252) + 0.0001);
rank(ret_12)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="Stochastic_RSI",
        category="momentum",
        description="StochRSI: RSI ranked within its own 14-day range",
        expression="""\
delta    = close - ts_delay(close, 1);
gain     = max(delta, 0);
loss     = max(-delta, 0);
avg_gain = ts_mean(gain, 14);
avg_loss = ts_mean(loss, 14);
rs       = avg_gain / (avg_loss + 0.0001);
rsi      = 100 - (100 / (1 + rs));
rsi_min  = ts_min(rsi, 14);
rsi_max  = ts_max(rsi, 14);
stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 0.0001);
-1 * rank(stoch_rsi)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MEAN REVERSION ALPHAS
# ─────────────────────────────────────────────────────────────────────────────

MEAN_REVERSION_ALPHAS: List[Alpha] = [

    Alpha(
        name="Bollinger_zscore",
        category="mean_reversion",
        description="Bollinger Band z-score: fade price deviation from 20-day mean",
        expression="""\
ma20     = ts_mean(close, 20);
std20    = ts_std_dev(close, 20);
bb_zscore = (close - ma20) / (std20 + 0.0001);
-1 * rank(bb_zscore)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="Short_term_reversal_5d",
        category="mean_reversion",
        description="5-day short-term return reversal",
        expression="""\
ret5 = (close - ts_delay(close, 5)) / (ts_delay(close, 5) + 0.0001);
-1 * rank(ret5)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="VWAP_deviation",
        category="mean_reversion",
        description="Fade deviation of close from VWAP",
        expression="""\
dev = (close - vwap) / (vwap + 0.0001);
-1 * rank(dev)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Industry"},
    ),

    Alpha(
        name="Intraday_reversal",
        category="mean_reversion",
        description="Stocks that gap up (open >> prev close) tend to revert",
        expression="""\
gap = (open - ts_delay(close, 1)) / (ts_delay(close, 1) + 0.0001);
-1 * rank(gap)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="Zscore_cross_sectional",
        category="mean_reversion",
        description="Cross-sectional z-score of 1-day returns — strong reversion",
        expression="""\
ret1 = (close - ts_delay(close, 1)) / (ts_delay(close, 1) + 0.0001);
-1 * zscore(ret1)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Market"},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 3. VOLUME ALPHAS
# ─────────────────────────────────────────────────────────────────────────────

VOLUME_ALPHAS: List[Alpha] = [

    Alpha(
        name="Volume_surprise",
        category="volume",
        description="Volume spike vs 20-day average: unusual volume predicts continuation",
        expression="""\
vol_ratio = volume / (adv20 + 1);
rank(ts_mean(vol_ratio, 5))""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="OBV_momentum",
        category="volume",
        description="On-Balance Volume trend over 10 days",
        expression="""\
ret1   = close - ts_delay(close, 1);
obv_d  = volume * sign(ret1);
obv10  = ts_sum(obv_d, 10);
rank(obv10)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Industry"},
    ),

    Alpha(
        name="Price_volume_divergence",
        category="volume",
        description="Price rises on falling volume = weakness (divergence)",
        expression="""\
price_chg = ts_delta(close, 5);
vol_chg   = ts_delta(volume, 5);
corr5     = ts_corr(price_chg, vol_chg, 10);
-1 * rank(corr5)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="VWAP_volume_rank",
        category="volume",
        description="Stocks trading below VWAP on high volume: accumulation signal",
        expression="""\
below_vwap = (close < vwap);
high_vol   = (volume > adv20);
rank(below_vwap * high_vol * (vwap - close) / (vwap + 0.0001))""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Industry"},
    ),

    Alpha(
        name="Volume_zscore_reversal",
        category="volume",
        description="Extreme volume + negative return → reversal long",
        expression="""\
vol_z = zscore(volume);
ret1  = (close - ts_delay(close, 1)) / (ts_delay(close, 1) + 0.0001);
rank(vol_z * (-1 * ret1))""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 4. VOLATILITY ALPHAS
# ─────────────────────────────────────────────────────────────────────────────

VOLATILITY_ALPHAS: List[Alpha] = [

    Alpha(
        name="Low_vol_factor",
        category="volatility",
        description="Low volatility anomaly: less volatile stocks outperform",
        expression="""\
vol20 = ts_std_dev(returns, 20);
-1 * rank(vol20)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="ATR_normalized",
        category="volatility",
        description="Average True Range (14) normalized — avoid high ATR stocks",
        expression="""\
tr    = max(high - low, max(abs(high - ts_delay(close, 1)), abs(low - ts_delay(close, 1))));
atr14 = ts_mean(tr, 14);
-1 * rank(atr14 / (close + 0.0001))""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Industry"},
    ),

    Alpha(
        name="Vol_regime_momentum",
        category="volatility",
        description="In low-vol regime: follow trend; high-vol: fade",
        expression="""\
vol10   = ts_std_dev(returns, 10);
ret5    = ts_delta(close, 5) / (ts_delay(close, 5) + 0.0001);
low_vol = (vol10 < ts_median(vol10, 60));
rank(low_vol * ret5 + (-1 * (1 - low_vol) * ret5))""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="Vol_contraction",
        category="volatility",
        description="Volatility contraction precedes breakout — rank contraction",
        expression="""\
vol5  = ts_std_dev(returns, 5);
vol20 = ts_std_dev(returns, 20);
-1 * rank(vol5 / (vol20 + 0.0001))""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 5. TREND FOLLOWING ALPHAS
# ─────────────────────────────────────────────────────────────────────────────

TREND_ALPHAS: List[Alpha] = [

    Alpha(
        name="MA_crossover_20_60",
        category="trend",
        description="Golden/death cross: 20-day MA vs 60-day MA",
        expression="""\
ma20 = ts_mean(close, 20);
ma60 = ts_mean(close, 60);
rank((ma20 - ma60) / (ma60 + 0.0001))""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="Price_above_MA200",
        category="trend",
        description="Long stocks trading above 200-day MA (trend filter)",
        expression="""\
ma200 = ts_mean(close, 200);
above = (close > ma200);
rank(above * (close - ma200) / (ma200 + 0.0001))""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Industry"},
    ),

    Alpha(
        name="ADX_trend_strength",
        category="trend",
        description="Directional movement: up-moves vs down-moves over 14 days",
        expression="""\
up_move   = high - ts_delay(high, 1);
down_move = ts_delay(low, 1) - low;
plus_dm   = max(up_move * (up_move > down_move) * (up_move > 0), 0);
minus_dm  = max(down_move * (down_move > up_move) * (down_move > 0), 0);
di_diff   = ts_mean(plus_dm, 14) - ts_mean(minus_dm, 14);
rank(di_diff)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="Donchian_breakout",
        category="trend",
        description="Price near 20-day high (breakout strength)",
        expression="""\
high20 = ts_max(high, 20);
low20  = ts_min(low,  20);
position = (close - low20) / (high20 - low20 + 0.0001);
rank(position)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Industry"},
    ),

    Alpha(
        name="EMA_slope",
        category="trend",
        description="Slope of 10-day EMA (approximate via ts_mean): rate of trend",
        expression="""\
ema10_now  = ts_mean(close, 10);
ema10_prev = ts_delay(ts_mean(close, 10), 5);
slope      = (ema10_now - ema10_prev) / (ema10_prev + 0.0001);
rank(slope)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# 6. COMBINED / COMPOSITE ALPHAS
# ─────────────────────────────────────────────────────────────────────────────

COMBINED_ALPHAS: List[Alpha] = [

    Alpha(
        name="RSI_volume_combo",
        category="combined",
        description="RSI mean-reversion + volume confirmation",
        expression="""\
delta    = close - ts_delay(close, 1);
gain     = max(delta, 0);
loss     = max(-delta, 0);
avg_gain = ts_mean(gain, 14);
avg_loss = ts_mean(loss, 14);
rs       = avg_gain / (avg_loss + 0.0001);
rsi      = 100 - (100 / (1 + rs));
vol_rank = rank(volume / (adv20 + 1));
rsi_sig  = -1 * rank(rsi);
rank(rsi_sig + 0.5 * vol_rank)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="Momentum_lowvol",
        category="combined",
        description="12-1 month momentum filtered by low volatility",
        expression="""\
ret12   = (ts_delay(close, 1) - ts_delay(close, 252)) / (ts_delay(close, 252) + 0.0001);
vol20   = ts_std_dev(returns, 20);
mom_adj = ret12 / (vol20 + 0.0001);
rank(mom_adj)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="Trend_reversion_blend",
        category="combined",
        description="Short-term reversion + medium-term trend",
        expression="""\
ret1  = (close - ts_delay(close, 1))  / (ts_delay(close, 1)  + 0.0001);
ret20 = (close - ts_delay(close, 20)) / (ts_delay(close, 20) + 0.0001);
rank(-1 * ret1 + 0.5 * ret20)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="MACD_RSI_filter",
        category="combined",
        description="MACD momentum confirmed by RSI not in extreme zone",
        expression="""\
ema12    = ts_mean(close, 12);
ema26    = ts_mean(close, 26);
macd     = ema12 - ema26;
signal9  = ts_mean(macd, 9);
macd_sig = macd - signal9;
delta    = close - ts_delay(close, 1);
gain     = max(delta, 0);
loss     = max(-delta, 0);
avg_gain = ts_mean(gain, 14);
avg_loss = ts_mean(loss, 14);
rs       = avg_gain / (avg_loss + 0.0001);
rsi      = 100 - (100 / (1 + rs));
not_extreme = (rsi > 30) * (rsi < 70);
rank(macd_sig * not_extreme)""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),

    Alpha(
        name="Multi_factor_score",
        category="combined",
        description="Equal-weight blend: momentum + reversion + volume",
        expression="""\
ret5      = (close - ts_delay(close, 5))  / (ts_delay(close, 5)  + 0.0001);
ret60     = (close - ts_delay(close, 60)) / (ts_delay(close, 60) + 0.0001);
vol_surge = volume / (adv20 + 1);
bb_dev    = (close - ts_mean(close, 20)) / (ts_std_dev(close, 20) + 0.0001);
rank(rank(ret60) - rank(ret5) - rank(bb_dev) + rank(vol_surge))""",
        suggested_settings={"universe": "TOP3000", "neutralization": "Subindustry"},
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Registry & helpers
# ─────────────────────────────────────────────────────────────────────────────

ALL_ALPHAS: List[Alpha] = (
    MOMENTUM_ALPHAS
    + MEAN_REVERSION_ALPHAS
    + VOLUME_ALPHAS
    + VOLATILITY_ALPHAS
    + TREND_ALPHAS
    + COMBINED_ALPHAS
)

CATEGORY_MAP = {
    "momentum":       MOMENTUM_ALPHAS,
    "mean_reversion": MEAN_REVERSION_ALPHAS,
    "volume":         VOLUME_ALPHAS,
    "volatility":     VOLATILITY_ALPHAS,
    "trend":          TREND_ALPHAS,
    "combined":       COMBINED_ALPHAS,
}


def print_alpha(alpha: Alpha, index: int) -> None:
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  [{index}] {alpha.name}  ({alpha.category.upper()})")
    print(f"  {alpha.description}")
    print(f"  Settings hint: {alpha.suggested_settings}")
    print(sep)
    print(alpha.expression)


def list_alphas(alphas: List[Alpha]) -> None:
    for i, a in enumerate(alphas, 1):
        print(f"  {i:>2}. [{a.category:<15}] {a.name:<30} — {a.description[:60]}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="WorldQuant Brain Alpha Generator")
    parser.add_argument(
        "--category",
        choices=list(CATEGORY_MAP.keys()),
        help="Filter by category",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show alpha names only (no expressions)",
    )
    parser.add_argument(
        "--name",
        help="Print a single alpha by name (e.g. RSI_14)",
    )
    args = parser.parse_args()

    # single alpha by name
    if args.name:
        matches = [a for a in ALL_ALPHAS if a.name.lower() == args.name.lower()]
        if not matches:
            print(f"Alpha '{args.name}' not found. Use --list to see names.")
            return
        print_alpha(matches[0], 1)
        return

    target = CATEGORY_MAP[args.category] if args.category else ALL_ALPHAS

    if args.list:
        print(f"\n{'─'*70}")
        print(f"  WorldQuant Brain Alphas  ({len(target)} total)")
        print(f"{'─'*70}")
        list_alphas(target)
        return

    print(f"\n{'═'*70}")
    print(f"  WorldQuant Brain Alpha Generator — {len(target)} alpha(s)")
    print(f"{'═'*70}")
    for i, alpha in enumerate(target, 1):
        print_alpha(alpha, i)

    print(f"\n{'═'*70}")
    print("  HOW TO USE IN WQB:")
    print("    1. Open Brain > New Alpha > Expression Editor")
    print("    2. Set Region = USA, Universe = TOP3000, Delay = 1")
    print("    3. Paste the expression above")
    print("    4. Set Neutralization per 'Settings hint'")
    print("    5. Click Simulate")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
