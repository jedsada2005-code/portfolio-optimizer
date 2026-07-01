# High Sharpe Strategies (Target: Sharpe > 1.25, Fitness > 1.0)

## WHY CURRENT STRATEGY FAILS

Current: rank((anl4_afv4_eps_mean - ts_delay(anl4_afv4_eps_mean, 30)) / (abs(ts_delay(anl4_afv4_eps_mean, 30)) + 0.01))

Problems:
1. Divides by abs(prev_value) → if EPS near 0 or negative → ratio explodes → noisy signal
2. 30-day window too short → analysts revise quarterly (63 days), not monthly
3. No winsorize → outliers pollute the rank
4. Single signal → 2020 crash wiped it (no diversification)

Fix:
- Normalize by ts_std_dev (cross-time volatility) instead of level
- Use 63-day window (1 quarter)
- Add winsorize(x, 4) to cut extreme outliers
- Combine with uncorrelated signals
