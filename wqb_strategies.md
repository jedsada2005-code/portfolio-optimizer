# 10 WorldQuant Brain Investment Strategies

**Datasets used:** Price Volume, Company Fundamentals, Analyst Estimates, Fundamental Scores, Sentiment, Volatility, News, Options Analytics, Systematic Risk Metrics

---

## Strategy 1: Classic Price Momentum (12-1 Month)

**Thesis:** Stocks that outperformed over the past 12 months (excluding last month) tend to continue outperforming.

**Region:** USA | **Universe:** TOP3000 | **Decay:** 5 | **Delay:** 1 | **Neutralization:** Subindustry

```python
rank(
    ts_sum(returns, 252) - ts_sum(returns, 21)
)
```

**Explanation:**
- `ts_sum(returns, 252)` — cumulative 12-month return
- Subtract `ts_sum(returns, 21)` to skip the most recent 1-month (reversal noise)
- `rank()` converts to cross-sectional percentile — long top decile, short bottom decile

---

## Strategy 2: EPS Revision Momentum

**Thesis:** Stocks with upward analyst EPS estimate revisions tend to outperform as the market gradually incorporates the new information.

**Region:** USA | **Universe:** TOP3000 | **Decay:** 10 | **Delay:** 1 | **Neutralization:** Subindustry

```python
rank(
    (anl4_afv4_eps_mean - ts_delay(anl4_afv4_eps_mean, 30)) /
    (abs(ts_delay(anl4_afv4_eps_mean, 30)) + 0.01)
)
```

**Explanation:**
- Measures % change in consensus EPS mean over past 30 days
- `+ 0.01` avoids division by zero for near-zero estimates
- Positive score = analysts revised earnings UP → bullish signal
- Uses `anl4_afv4_eps_mean` (annual forward EPS consensus mean)

---

## Strategy 3: Value — Earnings Yield + Book-to-Price

**Thesis:** Stocks with high earnings yield and high book-to-market value are underpriced relative to fundamentals.

**Region:** USA | **Universe:** TOP3000 | **Decay:** 5 | **Delay:** 1 | **Neutralization:** Subindustry

```python
rank(
    rank(anl4_afv4_eps_mean / close) +
    rank(bookvalue_ps / close)
)
```

**Explanation:**
- `anl4_afv4_eps_mean / close` — forward earnings yield (inverse of P/E)
- `bookvalue_ps / close` — book-to-price ratio (inverse of P/B)
- Sum of ranks creates a composite value score
- High score → stock is cheap on both earnings and book metrics

---

## Strategy 4: Quality — Free Cash Flow Yield

**Thesis:** High-FCF companies relative to market cap are better quality businesses generating real cash, not just accounting profits.

**Region:** USA | **Universe:** TOP3000 | **Decay:** 5 | **Delay:** 1 | **Neutralization:** Subindustry

```python
rank(
    anl4_fcf_mean / (cap + 1)
)
```

**Explanation:**
- `anl4_fcf_mean` — analyst consensus for free cash flow
- `cap` — market capitalization
- FCF / Market Cap = free cash flow yield
- High yield = company generates abundant cash relative to valuation

---

## Strategy 5: Composite Fundamental Score (Multi-Factor)

**Thesis:** Combining Growth, Quality, Value, and Momentum scores from a single integrated scoring model identifies the highest-quality all-around stocks.

**Region:** USA | **Universe:** TOP3000 | **Decay:** 5 | **Delay:** 1 | **Neutralization:** Subindustry

```python
rank(
    rank(fscore_growth) +
    rank(fscore_quality) +
    rank(fscore_value) +
    rank(fscore_momentum)
)
```

**Explanation:**
- `fscore_growth` — medium-term growth potential score
- `fscore_quality` — earnings sustainability and certainty rank
- `fscore_value` — under/overpricing relative to valuation standards
- `fscore_momentum` — analyst revision and price momentum rank
- Equal-weight rank sum produces a robust, multi-dimensional score

---

## Strategy 6: News Sentiment Trend

**Thesis:** Stocks with rising positive news sentiment over the past week tend to outperform as market participants react gradually to new information.

**Region:** USA | **Universe:** TOP3000 | **Decay:** 3 | **Delay:** 1 | **Neutralization:** Subindustry

```python
rank(
    ts_mean(nws18_ssc, 5) -
    ts_mean(nws18_ssc, 20)
)
```

**Explanation:**
- `nws18_ssc` — composite news sentiment score (multiple techniques)
- Short-term (5d) minus long-term (20d) moving average of sentiment
- Positive value = sentiment is trending UP recently → bullish signal
- `ts_mean` smooths noisy daily sentiment readings

---

## Strategy 7: Short-term Mean Reversion

**Thesis:** Stocks that drop sharply over 5 days tend to bounce back due to temporary overselling and liquidity-driven price moves.

**Region:** USA | **Universe:** TOP3000 | **Decay:** 3 | **Delay:** 1 | **Neutralization:** Market

```python
-rank(
    ts_sum(returns, 5)
)
```

**Explanation:**
- `ts_sum(returns, 5)` — cumulative 5-day return
- Negate with `-` so recent losers get high score (reversal bet)
- Works well at shorter holding periods (decay = 3)
- Best in liquid, large-cap universes where spreads are low

---

## Strategy 8: Volatility Risk Premium (IV vs HV)

**Thesis:** Stocks where implied volatility (IV) is high relative to realized historical volatility (HV) are pricing in too much uncertainty — short vol premium by going long stocks with low IV/HV ratio.

**Region:** USA | **Universe:** TOP3000 | **Decay:** 5 | **Delay:** 1 | **Neutralization:** Subindustry

```python
-rank(
    implied_volatility_call_30 / (historical_volatility_30 + 0.001)
)
```

**Explanation:**
- `implied_volatility_call_30` — 30-day ATM call implied vol
- `historical_volatility_30` — 30-day close-to-close realized vol
- High IV/HV ratio = options expensive relative to realized vol → negative signal
- Negate so stocks with LOW IV/HV get highest rank (long low-premium stocks)

---

## Strategy 9: Analyst Recommendation Upgrade Signal

**Thesis:** Stocks receiving more buy recommendations than sell recommendations, with rising consensus, tend to outperform due to institutional buying pressure.

**Region:** USA | **Universe:** TOP3000 | **Decay:** 10 | **Delay:** 1 | **Neutralization:** Subindustry

```python
rank(
    (anl4_buy - anl4_sell) /
    (anl4_buy + anl4_hold + anl4_sell + 1)
)
```

**Explanation:**
- `anl4_buy` — count of buy recommendations
- `anl4_sell` — count of sell recommendations
- `anl4_hold` — count of hold recommendations
- Net buy ratio = (buys - sells) / total recommendations
- High positive ratio = strong analyst consensus to go long

---

## Strategy 10: Low Beta Anomaly (Risk-Adjusted Outperformance)

**Thesis:** Low-beta stocks tend to outperform their expected CAPM return, offering better risk-adjusted returns than high-beta stocks.

**Region:** USA | **Universe:** TOP3000 | **Decay:** 5 | **Delay:** 1 | **Neutralization:** Subindustry

```python
rank(
    ts_sum(returns, 63) /
    (beta_last_90_days_spy + 0.1)
) - rank(beta_last_90_days_spy)
```

**Explanation:**
- `ts_sum(returns, 63)` — 3-month cumulative return
- `beta_last_90_days_spy` — 90-day market beta to SPY
- Return / Beta = return per unit of systematic risk
- Subtract beta rank to further penalize high-beta stocks
- Long: high return-per-beta, low beta stocks; Short: low return-per-beta, high beta stocks

---

## Summary Table

| # | Strategy | Key Fields | Factor Type | Holding Period |
|---|----------|-----------|-------------|----------------|
| 1 | Price Momentum (12-1) | `returns` | Momentum | Medium |
| 2 | EPS Revision Momentum | `anl4_afv4_eps_mean` | Analyst/Momentum | Medium |
| 3 | Earnings Yield + B/P Value | `anl4_afv4_eps_mean`, `bookvalue_ps`, `close` | Value | Long |
| 4 | FCF Yield Quality | `anl4_fcf_mean`, `cap` | Quality | Long |
| 5 | Composite Fundamental Score | `fscore_growth/quality/value/momentum` | Multi-factor | Medium |
| 6 | News Sentiment Trend | `nws18_ssc` | Sentiment | Short |
| 7 | Short-term Reversal | `returns` | Mean Reversion | Short |
| 8 | IV/HV Volatility Premium | `implied_volatility_call_30`, `historical_volatility_30` | Volatility | Medium |
| 9 | Analyst Buy Ratio | `anl4_buy`, `anl4_sell`, `anl4_hold` | Analyst/Sentiment | Medium |
| 10 | Low Beta Anomaly | `returns`, `beta_last_90_days_spy` | Risk/Quality | Medium |

---

## WorldQuant Brain Settings Reference

All strategies use these standard settings (adjust per backtest results):

```
Region:          USA
Universe:        TOP3000
Delay:           1         (use delay=1 to avoid look-ahead bias)
Decay:           5-10      (days to linearly decay position weights)
Neutralization:  Subindustry  (remove sector bets, isolate stock selection)
Truncation:      0.08      (max 8% weight per stock)
Pasteurize NaN:  On        (replace missing data with 0)
Unit Handling:   Verify    (ensure consistent units across fields)
```
