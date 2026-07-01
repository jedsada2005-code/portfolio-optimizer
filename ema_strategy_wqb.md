# EMA 50/200 Trading Strategy — Complete Theory + WorldQuant Brain Code

---

## PART 1: INDICATOR EXPLANATION

### What is EMA 50?
The **Exponential Moving Average 50 (EMA 50)** is the average closing price over the last 50 trading days,
with more weight given to recent prices than older ones. Because it reacts faster than a Simple Moving Average,
it reflects SHORT-TO-MEDIUM term trend direction.

- Reacts quickly to price changes
- Acts as DYNAMIC SUPPORT in uptrends (price bounces off it)
- Acts as DYNAMIC RESISTANCE in downtrends (price rejects at it)
- ~10 weeks of trading data

### What is EMA 200?
The **Exponential Moving Average 200 (EMA 200)** averages the last 200 trading days (~40 weeks / ~10 months).
It is the GOLD STANDARD line for defining the long-term trend used by institutional traders globally.

- Slow to react — filters out short-term noise
- The line that separates BULL MARKET from BEAR MARKET
- Institutional funds often use it as a benchmark line to add/reduce exposure

### Why Use EMA 50 and EMA 200 Together?
Together they show you:
1. **Trend direction** — Is the market in a bull or bear phase?
2. **Trend strength** — How far apart are the two lines?
3. **Transition signals** — When they cross, the trend may be changing
4. **Entry zones** — Where to buy on pullbacks (EMA 50) in an uptrend

---

### Golden Cross
```
EMA 50 crosses ABOVE EMA 200
```
- **Signal:** Bullish — medium-term momentum has overtaken long-term trend
- **Meaning:** The market is transitioning from bearish to bullish
- **Action:** Look for LONG entries
- **Confirmation:** Price should also be ABOVE both EMAs on the cross day
- **Note:** Can lag — cross often happens AFTER price has already recovered

```
Visual:
        /--- EMA 50 (fast)
       X
      / \--- EMA 200 (slow, flat/rising)
     /
────────────────────────────────── Price
```

### Death Cross
```
EMA 50 crosses BELOW EMA 200
```
- **Signal:** Bearish — medium-term momentum has fallen below long-term trend
- **Meaning:** The market is transitioning from bullish to bearish
- **Action:** Look for SHORT entries or exit longs
- **Confirmation:** Price should also be BELOW both EMAs on the cross day
- **Note:** Can lag — cross often happens AFTER price has already fallen

```
Visual:
────────────────────────────────── Price (declining)
      \ \--- EMA 200 (slow, flat/falling)
       X
        \--- EMA 50 (fast, falling faster)
```

---

## PART 2: MARKET INTERPRETATION

### Case 1: Price ABOVE Both EMA 50 and EMA 200 (AND EMA 50 > EMA 200)
```
Status: STRONG BULL MARKET
```
- Market is in full uptrend
- EMA 50 is the first support level; EMA 200 is deeper support
- Bias: LONG ONLY
- Strategy: Buy pullbacks to EMA 50; add on breakouts to new highs

### Case 2: Price BELOW Both EMA 50 and EMA 200 (AND EMA 50 < EMA 200)
```
Status: STRONG BEAR MARKET
```
- Market is in full downtrend
- EMA 50 is the first resistance; EMA 200 is deeper resistance
- Bias: SHORT ONLY (or cash)
- Strategy: Sell rallies to EMA 50; add on breakdowns to new lows

### Case 3: Price ABOVE EMA 200 but BELOW EMA 50
```
Status: PULLBACK IN BULL MARKET (caution zone)
```
- Long-term trend still bullish (price > EMA 200)
- Short-term momentum weakening (price < EMA 50)
- Bias: Look for LONG entries IF price recovers back above EMA 50
- Risk: If price drops below EMA 200 → trend may be reversing

### Case 4: Price BELOW EMA 200 but ABOVE EMA 50
```
Status: RALLY IN BEAR MARKET (caution zone)
```
- Long-term trend still bearish (price < EMA 200)
- Short-term rally occurring (price > EMA 50)
- Bias: This is often a DEAD CAT BOUNCE — be cautious on longs
- Risk: Watch for EMA 50 to reject price downward again

### Case 5: Sideways / Choppy Market (EMA 50 ≈ EMA 200)
```
Status: NO CLEAR TREND — High false signal risk
```
- Both EMAs are flat and intertwined
- Golden/Death Crosses will be FREQUENT and unreliable
- Strategy: REDUCE position size or AVOID trading until separation appears
- Filter: Only trade crosses when EMA 200 is clearly sloping (not flat)

---

## PART 3: ENTRY RULES

### LONG ENTRY (Buy Signal)
```
PRIMARY TRIGGER:  Golden Cross — EMA 50 crosses above EMA 200
CONFIRMATION:     Price closes ABOVE both EMA 50 and EMA 200
ENTRY METHOD A:   Enter on the close of the Golden Cross candle
ENTRY METHOD B:   Wait for a pullback — Enter when price retouches EMA 50 from above
```

**Long Entry Checklist:**
- [x] EMA 50 > EMA 200 (Golden Cross confirmed OR already in place)
- [x] Price > EMA 50 > EMA 200 (price above both)
- [x] Volume on the breakout > 20-day average volume (confirms conviction)
- [x] Ideally price pulls back to EMA 50 without closing below it
- [x] Enter on the close of the first bullish candle that bounces off EMA 50

### SHORT ENTRY (Sell Signal)
```
PRIMARY TRIGGER:  Death Cross — EMA 50 crosses below EMA 200
CONFIRMATION:     Price closes BELOW both EMA 50 and EMA 200
ENTRY METHOD A:   Enter on the close of the Death Cross candle
ENTRY METHOD B:   Wait for a rally — Enter when price retouches EMA 50 from below
```

**Short Entry Checklist:**
- [x] EMA 50 < EMA 200 (Death Cross confirmed OR already in place)
- [x] Price < EMA 50 < EMA 200 (price below both)
- [x] Volume on the breakdown > 20-day average volume
- [x] Ideally price rallies to EMA 50 without closing above it
- [x] Enter on the close of the first bearish candle that rejects EMA 50

---

## PART 4: EXIT RULES

### EXIT LONG (Close Buy Position)
| Rule | Condition | Action |
|------|-----------|--------|
| Stop Loss | Price closes BELOW EMA 50 | Exit 100% of position |
| Hard Stop | Price closes BELOW EMA 200 | Emergency full exit |
| Signal Reversal | Death Cross appears | Flip from long to short |
| Profit Target | Risk-Reward 1:2 or 1:3 | Take profit at 2x or 3x the initial risk |
| Trailing Stop | Move stop up as EMA 50 rises | Lock in profits dynamically |

### EXIT SHORT (Close Sell Position)
| Rule | Condition | Action |
|------|-----------|--------|
| Stop Loss | Price closes ABOVE EMA 50 | Exit 100% of position |
| Hard Stop | Price closes ABOVE EMA 200 | Emergency full exit |
| Signal Reversal | Golden Cross appears | Flip from short to long |
| Profit Target | Risk-Reward 1:2 or 1:3 | Take profit at 2x or 3x the initial risk |
| Trailing Stop | Move stop down as EMA 50 falls | Lock in profits dynamically |

### Risk-Reward Examples
```
Entry at $100, Stop Loss at $95 → Risk = $5 per share
- 1:2 Target → Take Profit at $110
- 1:3 Target → Take Profit at $115
```

---

## PART 5: WORLDQUANT BRAIN EXPRESSIONS

> **Note:** WorldQuant Brain uses `ts_mean(close, N)` as the SMA proxy for EMA N.
> True EMA requires recursive calculation not directly available in WQB expression syntax.
> `ts_mean(close, 50)` is the industry-standard WQB approximation for EMA 50.
> Settings for all alphas: Region=USA, Universe=TOP3000, Delay=1, Neutralization=Subindustry

---

### Alpha 1: EMA Trend Direction (Core Signal)

**Theory mapped:** Price above/below EMAs + EMA 50 vs EMA 200 relationship

```python
rank(ts_mean(close, 50) - ts_mean(close, 200))
```

**How it works:**
- `ts_mean(close, 50)` → EMA 50 proxy (50-day moving average)
- `ts_mean(close, 200)` → EMA 200 proxy (200-day moving average)
- EMA50 - EMA200 > 0 → Golden Cross territory → positive alpha → LONG
- EMA50 - EMA200 < 0 → Death Cross territory → negative alpha → SHORT
- `rank()` normalizes cross-sectionally so positions scale by relative trend strength
- Settings: Decay=10, Truncation=0.08

---

### Alpha 2: Golden Cross / Death Cross Momentum Change

**Theory mapped:** Detecting the actual cross event and its momentum

```python
rank(
    (ts_mean(close, 50) / ts_mean(close, 200)) -
    ts_delay(ts_mean(close, 50) / ts_mean(close, 200), 20)
)
```

**How it works:**
- `ts_mean(close,50) / ts_mean(close,200)` → EMA ratio (>1 = golden, <1 = death)
- Subtract the same ratio 20 days ago → measures HOW FAST the ratio is changing
- Positive value = EMA50 accelerating above EMA200 → fresh Golden Cross signal
- Negative value = EMA50 accelerating below EMA200 → fresh Death Cross signal
- Captures the MOMENTUM OF THE CROSS, not just its level
- Settings: Decay=5, Truncation=0.08

---

### Alpha 3: Pullback-to-EMA50 Entry in Trending Markets

**Theory mapped:** Entry Method B — buy pullbacks to EMA 50 in confirmed uptrends

```python
rank(
    If(
        ts_mean(close, 50) > ts_mean(close, 200),
        -(close - ts_mean(close, 50)) / ts_mean(close, 50),
        (close - ts_mean(close, 50)) / ts_mean(close, 50)
    )
)
```

**How it works:**
- `If(EMA50 > EMA200, ...)` → checks if we are in UPTREND (Golden Cross zone)
  - In UPTREND: `-(close - EMA50)/EMA50` → score is HIGH when price is just BELOW EMA50 (pullback)
    - Buy stocks that have pulled back to their EMA50 support in uptrends
  - In DOWNTREND: `(close - EMA50)/EMA50` → score is HIGH when price is just ABOVE EMA50 (overextended rally)
    - Short stocks that have bounced up to their EMA50 resistance in downtrends
- Captures the exact "pullback to EMA" entry theory
- Settings: Decay=5, Truncation=0.08

---

### Alpha 4: Price-EMA Alignment Strength

**Theory mapped:** Full bullish alignment (Price > EMA50 > EMA200) vs full bearish alignment

```python
rank(
    (close / ts_mean(close, 50) - 1) +
    (ts_mean(close, 50) / ts_mean(close, 200) - 1)
)
```

**How it works:**
- `close / EMA50 - 1` → how far price is above/below EMA50 (momentum)
- `EMA50 / EMA200 - 1` → how far EMA50 is above/below EMA200 (trend structure)
- Sum of both: highest when PRICE > EMA50 >> EMA200 (full bull alignment)
- Lowest when PRICE < EMA50 << EMA200 (full bear alignment)
- This directly encodes the Case 1 / Case 2 market interpretation
- Settings: Decay=10, Truncation=0.08

---

### Alpha 5: Volume-Confirmed EMA Breakout

**Theory mapped:** Entry rule — volume must exceed 20-day average on breakout

```python
rank(
    (ts_mean(close, 50) - ts_mean(close, 200)) *
    (volume / adv20)
)
```

**How it works:**
- `EMA50 - EMA200` → baseline trend signal (positive = uptrend, negative = downtrend)
- `volume / adv20` → volume ratio (>1 = above-average volume day)
- Multiply: signal is AMPLIFIED on HIGH VOLUME days → confirms genuine breakouts
- A golden cross day with 2x average volume gets double the signal weight
- Quiet, low-volume golden crosses get reduced weight (possible false signal)
- Settings: Decay=5, Truncation=0.08

---

### Alpha 6: EMA Cross + Price Confirmation (Full Entry Rule)

**Theory mapped:** Complete long/short entry with all three conditions: cross, price confirmation, and trend

```python
rank(
    If(
        (ts_mean(close, 50) > ts_mean(close, 200)) &
        (close > ts_mean(close, 50)),
        (close - ts_mean(close, 200)) / ts_mean(close, 200),
        If(
            (ts_mean(close, 50) < ts_mean(close, 200)) &
            (close < ts_mean(close, 50)),
            (close - ts_mean(close, 200)) / ts_mean(close, 200),
            0
        )
    )
)
```

**How it works:**
- Branch 1 — LONG conditions:
  - EMA50 > EMA200 ✓ (Golden Cross confirmed)
  - Price > EMA50 ✓ (price above both EMAs)
  - Output: positive distance from EMA200 → strong longs get higher scores
- Branch 2 — SHORT conditions:
  - EMA50 < EMA200 ✓ (Death Cross confirmed)
  - Price < EMA50 ✓ (price below both EMAs)
  - Output: negative distance from EMA200 → strong shorts get lower (more negative) scores
- Branch 3 — NEUTRAL (choppy market):
  - Returns 0 → no position (avoids sideways false signals, Case 5)
- This is the most faithful encoding of the FULL entry rule checklist
- Settings: Decay=5, Truncation=0.08

---

### Alpha 7: Exit Signal — Trend Reversal Detector

**Theory mapped:** Exit long on Death Cross, exit short on Golden Cross

```python
-rank(
    ts_mean(
        (ts_mean(close, 50) - ts_mean(close, 200)) /
        ts_mean(close, 200),
        5
    ) -
    ts_delay(
        ts_mean(
            (ts_mean(close, 50) - ts_mean(close, 200)) /
            ts_mean(close, 200),
            5
        ),
        10
    )
)
```

**How it works:**
- Computes the 5-day smoothed EMA gap ratio
- Subtracts the same ratio from 10 days ago
- Negative result = EMA gap is SHRINKING (cross is approaching)
- The negation `-rank(...)` means positions are REDUCED before the cross happens
- This acts as an EARLY EXIT SIGNAL — cuts positions as trend starts to fade
- Settings: Decay=3, Truncation=0.08

---

### Alpha 8: Composite EMA Strategy (Production Alpha)

**Theory mapped:** All rules combined — direction + strength + pullback + volume + noise filter

```python
rank(
    0.4 * rank(ts_mean(close, 50) - ts_mean(close, 200)) +
    0.3 * rank(
        If(
            ts_mean(close, 50) > ts_mean(close, 200),
            -(close - ts_mean(close, 50)) / ts_mean(close, 50),
            (close - ts_mean(close, 50)) / ts_mean(close, 50)
        )
    ) +
    0.2 * rank(
        (ts_mean(close, 50) / ts_mean(close, 200)) -
        ts_delay(ts_mean(close, 50) / ts_mean(close, 200), 20)
    ) +
    0.1 * rank(
        (ts_mean(close, 50) - ts_mean(close, 200)) * (volume / adv20)
    )
)
```

**How it works:**
- **40% weight → Alpha 1** (EMA direction — the core trend signal)
- **30% weight → Alpha 3** (pullback entry — identifies the optimal entry zone)
- **20% weight → Alpha 2** (cross momentum — rewards fresh crosses)
- **10% weight → Alpha 5** (volume confirmation — penalizes false low-volume signals)
- `rank()` outer layer normalizes the composite for portfolio construction
- Settings: Decay=10, Truncation=0.08, Neutralization=Subindustry

---

## PART 6: STRATEGY SETTINGS REFERENCE

| Alpha | Strategy | Decay | Neutralization | Best For |
|-------|----------|-------|----------------|----------|
| 1 | EMA Trend Direction | 10 | Subindustry | Trend-following |
| 2 | Cross Momentum | 5 | Subindustry | Cross detection |
| 3 | Pullback to EMA50 | 5 | Subindustry | Entry timing |
| 4 | Price-EMA Alignment | 10 | Subindustry | Trend strength |
| 5 | Volume-Confirmed Breakout | 5 | Subindustry | Signal quality |
| 6 | Full Entry Rule | 5 | Subindustry | Precision entries |
| 7 | Exit / Reversal Signal | 3 | Market | Exits / flips |
| 8 | Composite (Production) | 10 | Subindustry | Full deployment |

---

## PART 7: VISUAL DECISION TREE

```
                      START
                        |
            ┌───────────┴───────────┐
        EMA50 > EMA200           EMA50 < EMA200
       (Golden Cross Zone)      (Death Cross Zone)
            |                       |
    ┌───────┴───────┐       ┌───────┴───────┐
Price > EMA50   Price < EMA50  Price < EMA50  Price > EMA50
    |               |           |               |
STRONG LONG     WAIT /       STRONG SHORT    WAIT /
  SIGNAL        PULLBACK       SIGNAL        PULLBACK
  ENTRY         SETUP          ENTRY         SETUP
    |               |           |               |
Alpha 6=+ve     Alpha 3=+ve  Alpha 6=-ve    Alpha 3=-ve
  (Long)       (Buy dip)      (Short)      (Sell rally)
```

---

## PART 8: RISK MANAGEMENT IN WQB

```python
# Apply truncation to cap max position at 8%
# Use these settings in WQB simulator:

# Decay    = 10   → holds positions ~10 days (matching medium-term EMA signals)
# Truncate = 0.08 → max 8% of portfolio in any single stock
# Delay    = 1    → use previous day's data (no look-ahead bias)
# Universe = TOP3000 → liquid stocks with tight spreads

# For short holding period (reversal exits):
# Decay    = 3    → aggressively exits positions quickly
```
