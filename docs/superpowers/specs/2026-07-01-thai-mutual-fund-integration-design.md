# Thai Mutual Fund Data Integration — Design

Status: Approved
Date: 2026-07-01

## Problem

`app.py` (Portfolio Optimizer & Backtesting) sources all price data from
`yfinance`. Yahoo Finance does not carry Thai mutual fund NAV data, so the
app currently cannot include Thai mutual funds in a portfolio alongside
stocks/ETFs.

## Data source

Thailand SEC (ก.ล.ต.) Open API Developer Portal — `api-portal.sec.or.th`.
Two products are relevant:

- **Fund Factsheet API** — fund list per AMC (`GET
  https://api.sec.or.th/FundFactsheet/fund/amc`) and fund detail lookup, used
  to resolve a human-readable fund short name to its internal `proj_id`.
- **Fund Daily Info API** — daily NAV per fund per date (`GET
  https://api.sec.or.th/FundDailyInfo/{proj_id}/dailynav/{nav_date}`),
  returns `last_val` (NAV per unit) among other fields.

Both require a free subscription key, obtained by registering at
`api-portal.sec.or.th` and subscribing to both products. The key is sent as
the `Ocp-Apim-Subscription-Key` header. **The user must register and obtain
this key themselves** — it cannot be provisioned by this change. Note: the
portal migrated to a new version in January 2026; exact paths should be
re-verified against current docs at integration time, but the mechanism
(subscription key + per-fund-per-date NAV lookup, no bulk range endpoint) is
confirmed current.

## Symbol format

Thai mutual funds are entered in the existing comma-separated symbols box,
prefixed with `MF:`, e.g.:

```
AMZN, META, SPY, MF:K-CHANGE-A(A)
```

Parsing splits `stock_list` into `yf_symbols` (no prefix, unchanged
behavior) and `mf_symbols` (prefix stripped).

## New module: `thai_mf.py`

- `SECFundClient(subscription_key: str)` — wraps the two SEC endpoints.
  Raises a clear error type on missing/invalid key or HTTP failure.
- Local disk cache under `.cache/thai_mf/` (added to `.gitignore`):
  - `fund_list.json` — cached name → `proj_id` map, refreshed periodically
    (e.g. re-fetched if the cache file is older than N days).
  - `nav_<proj_id>.csv` — cached `(date, nav)` rows per fund.
- `resolve_fund_id(name: str) -> str | None` — looks up `name` (as typed
  after `MF:`) against the cached fund list. Returns `None` if unmatched;
  the caller reports this the same way an invalid yfinance ticker is
  reported today.
- `get_nav_history(proj_id: str, start: date, end: date) -> pd.Series` —
  reads whatever's already cached for `proj_id`, fetches only the missing
  dates in `[start, end]` from the SEC Daily NAV endpoint (one HTTP call
  per fund per missing date — there is no bulk range endpoint), appends
  results to the CSV cache, and returns the full series indexed by date.
  This makes the first backtest over a wide date range slow (many
  sequential calls); subsequent runs against the same fund/range are fast
  since they hit cache.

### Per-date error handling inside `get_nav_history`

- HTTP 204 / no NAV published for that date (holiday, weekend, fund not
  yet launched) → skip the date silently, same as how the app already
  treats missing weekend prices for stocks.
- HTTP 429 (rate limited) → retry with backoff up to a small fixed number
  of attempts; if still failing, stop fetching further dates for that fund
  and surface a warning that the range may be incomplete, rather than
  failing the whole page.
- Any other persistent HTTP error → treat the fund as failed to load,
  report it in the existing "missing" warning banner.

## Integration into `app.py`

Right after the existing `yf.download(...)` block:

1. Split `stock_list` into `yf_symbols` / `mf_symbols` as described above.
2. If `mf_symbols` is non-empty and no SEC API key has been entered, show
   `st.error(...)` and `st.stop()` before attempting any download (same
   pattern already used for "no data downloaded").
3. For each `mf_symbols` entry: resolve via `thai_mf.resolve_fund_id`;
   unresolved names are added to the same `missing` list already shown in
   the "ไม่พบข้อมูล" warning. Resolved ones are fetched via
   `thai_mf.get_nav_history` over `[start_date, end_date]`.
4. Build a Thai-fund NAV DataFrame (columns = original `MF:`-prefixed
   symbol names, so downstream weight labels stay distinguishable from
   stock tickers) and outer-join it onto the yfinance `data_close`
   DataFrame on the date index, then `ffill().bfill()` — matching the
   existing fill behavior for stocks.
5. Everything downstream (efficient frontier, optimal weights, custom
   weight sliders, backtesting, NAV breakdown) is unchanged — all of it
   already operates purely on the unified `data_close` DataFrame and a
   list of column names, with no assumption that columns come from
   yfinance.

## UI changes

- New sidebar field: `SEC Open API Key` (`st.text_input(..., type="password")`),
  only required when the symbols box contains an `MF:` entry.
- Updated sidebar captions: explain the `MF:` prefix, note that Thai fund
  data comes from the SEC Open API and requires registering for a free key
  at `api-portal.sec.or.th`, and that first-time fetches for a new fund/date
  range are slower due to per-date API calls.
- Missing-symbol warning banner extended to include unresolved `MF:` names
  alongside unresolved yfinance tickers (already a single combined message
  today — just needs the two missing-lists merged before display).

## Dependencies

`requests` added to `requirements.txt` (used directly by `thai_mf.py`;
currently only pulled in transitively via `yfinance`).

## Testing approach

- Unit-testable without a real API key (using mocked HTTP responses):
  `resolve_fund_id` matching logic, NAV cache read/write/merge behavior in
  `get_nav_history`, and the `data_close` merge step in `app.py`.
- End-to-end testing against the live SEC API requires the user's own
  subscription key and happens manually once they've registered.
- Regression check: with only non-`MF:` symbols entered, app behavior must
  be identical to today (no SEC calls attempted, no new UI required beyond
  the always-visible but conditionally-required API key field).

## Out of scope

- Registering the SEC API subscription on the user's behalf.
- Any data source other than the SEC Open API (e.g. Finnomena, Morningstar
  Thailand) — SEC is the official, free, programmatically-accessible source
  and covers the need.
- Real-time/intraday Thai fund NAV (SEC only publishes end-of-day NAV,
  consistent with how the app already treats daily stock closes).
