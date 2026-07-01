# Thai Mutual Fund Data Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users add Thai mutual funds (via `MF:` prefix in the existing symbols box) to a portfolio alongside yfinance stocks/ETFs, sourcing NAV data from Thailand SEC's Open API.

**Architecture:** A new standalone module `thai_mf.py` wraps the two SEC Open API endpoints (fund list lookup, daily NAV), with an on-disk cache (`.cache/thai_mf/`) so repeated backtests don't re-hit the per-date NAV endpoint. `app.py` splits the symbols box into yfinance symbols and `MF:`-prefixed fund names, fetches each side separately, and outer-joins them into the same `data_close` DataFrame the rest of the app already consumes unchanged.

**Tech Stack:** Python, `requests` (new), `pandas`, `pytest` (new, dev/test only), Streamlit (unchanged).

Reference: [docs/superpowers/specs/2026-07-01-thai-mutual-fund-integration-design.md](../specs/2026-07-01-thai-mutual-fund-integration-design.md)

Confirmed SEC Open API shape (from public reference implementations, since SEC's own docs portal migrated in Jan 2026 and may show updated paths):
- `GET https://api.sec.or.th/FundFactsheet/fund/amc` → list of AMCs, each with `unique_id`.
- `GET https://api.sec.or.th/FundFactsheet/fund/amc/{unique_id}` → list of that AMC's funds, each with `proj_id`, `proj_abbr_name`, `proj_name_th`, `proj_name_en`, `fund_status`.
- `GET https://api.sec.or.th/FundDailyInfo/{proj_id}/dailynav/{nav_date}` (nav_date as `YYYY-MM-DD`) → `200` with JSON containing `last_val` (NAV/unit), or `204` if no NAV published that date.
- All requests need header `Ocp-Apim-Subscription-Key: <key>`.

If the user's real subscription key turns up different field names when Task 7's manual smoke test runs, the only place to fix is `resolve_fund_id`'s `candidates` list and `get_daily_nav`'s `.get("last_val")` in `thai_mf.py` — everything else is shape-agnostic.

---

### Task 1: Project scaffolding for tests and dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `.gitignore`
- Create: `tests/test_setup_sanity.py`

- [ ] **Step 1: Add new dependencies**

Edit `requirements.txt` to:

```
streamlit
yfinance
pandas
numpy
plotly
pyportfolioopt
matplotlib
requests
pytest
```

- [ ] **Step 2: Install dependencies**

Run: `pip install -r requirements.txt`
Expected: installs `requests` and `pytest` (others already satisfied).

- [ ] **Step 3: Ignore the local NAV/fund cache directory**

Create `.gitignore`:

```
.cache/
__pycache__/
*.pyc
```

- [ ] **Step 4: Write a sanity test so pytest wiring is confirmed before real tests are added**

Create `tests/test_setup_sanity.py`:

```python
def test_pytest_is_wired_up():
    assert 1 + 1 == 2
```

- [ ] **Step 5: Run it**

Run: `python -m pytest -v` (from repo root — `-m pytest` ensures the repo root is on `sys.path` so later tests can `import thai_mf`)
Expected: `tests/test_setup_sanity.py::test_pytest_is_wired_up PASSED`, 1 passed.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore tests/test_setup_sanity.py
git commit -m "chore: add requests/pytest deps, gitignore local cache dir"
```

---

### Task 2: `SECFundClient.get_daily_nav` — single fund/date NAV lookup

**Files:**
- Create: `thai_mf.py`
- Test: `tests/test_thai_mf.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_thai_mf.py`:

```python
from unittest.mock import MagicMock, patch
import datetime as dt

import thai_mf


def _make_client(session):
    return thai_mf.SECFundClient("dummy-key", session=session)


def test_get_daily_nav_returns_last_val_on_200():
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=200, json=lambda: {"last_val": 12.3456})
    client = _make_client(session)

    result = client.get_daily_nav("proj-123", dt.date(2024, 1, 5))

    assert result == 12.3456
    called_url = session.get.call_args.args[0]
    assert called_url == "https://api.sec.or.th/FundDailyInfo/proj-123/dailynav/2024-01-05"
    assert session.get.call_args.kwargs["headers"] == {"Ocp-Apim-Subscription-Key": "dummy-key"}


def test_get_daily_nav_returns_none_on_204():
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=204)
    client = _make_client(session)

    result = client.get_daily_nav("proj-123", dt.date(2024, 1, 6))

    assert result is None


def test_get_daily_nav_raises_generic_error_on_persistent_failure():
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=500)
    client = _make_client(session)

    try:
        client.get_daily_nav("proj-123", dt.date(2024, 1, 6))
        assert False, "expected SECAPIError"
    except thai_mf.SECAPIError:
        pass


@patch("thai_mf.time.sleep", return_value=None)
def test_get_daily_nav_raises_rate_limit_error_after_max_retries(mock_sleep):
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=429)
    client = _make_client(session)

    try:
        client.get_daily_nav("proj-123", dt.date(2024, 1, 6))
        assert False, "expected SECRateLimitError"
    except thai_mf.SECRateLimitError:
        pass

    assert session.get.call_count == thai_mf.MAX_RETRIES
    assert mock_sleep.call_count == thai_mf.MAX_RETRIES - 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_thai_mf.py -v`
Expected: `ModuleNotFoundError: No module named 'thai_mf'` (or collection error) — `thai_mf.py` doesn't exist yet.

- [ ] **Step 3: Implement `thai_mf.py`**

Create `thai_mf.py`:

```python
import time

import requests

FUND_FACTSHEET_BASE = "https://api.sec.or.th/FundFactsheet"
FUND_DAILY_INFO_BASE = "https://api.sec.or.th/FundDailyInfo"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


class SECAPIError(Exception):
    """Raised when the SEC Open API returns a persistent, non-recoverable error."""


class SECRateLimitError(SECAPIError):
    """Raised when the SEC Open API keeps returning HTTP 429 after retries."""


class SECFundClient:
    def __init__(self, subscription_key, session=None):
        self.subscription_key = subscription_key
        self.session = session or requests.Session()

    def _headers(self):
        return {"Ocp-Apim-Subscription-Key": self.subscription_key}

    def get_daily_nav(self, proj_id, nav_date):
        """Return the NAV per unit for proj_id on nav_date (a date object),
        or None if no NAV was published that day.
        """
        url = f"{FUND_DAILY_INFO_BASE}/{proj_id}/dailynav/{nav_date.isoformat()}"
        attempts = 0
        while True:
            resp = self.session.get(url, headers=self._headers(), timeout=30)
            if resp.status_code == 200:
                return resp.json().get("last_val")
            if resp.status_code == 204:
                return None
            if resp.status_code == 429:
                attempts += 1
                if attempts >= MAX_RETRIES:
                    raise SECRateLimitError(
                        f"Rate limited fetching NAV for {proj_id} on {nav_date}"
                    )
                time.sleep(RETRY_BACKOFF_SECONDS * attempts)
                continue
            raise SECAPIError(
                f"SEC API error {resp.status_code} fetching NAV for {proj_id} on {nav_date}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_thai_mf.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add thai_mf.py tests/test_thai_mf.py
git commit -m "feat: add SECFundClient.get_daily_nav with retry/error handling"
```

---

### Task 3: `SECFundClient.list_funds` — fund list across all AMCs

**Files:**
- Modify: `thai_mf.py`
- Test: `tests/test_thai_mf.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_thai_mf.py`:

```python
def test_list_funds_aggregates_across_amcs():
    session = MagicMock()

    def fake_get(url, headers, timeout):
        if url == "https://api.sec.or.th/FundFactsheet/fund/amc":
            return MagicMock(
                status_code=200,
                json=lambda: [{"unique_id": "amc-1"}, {"unique_id": "amc-2"}],
            )
        if url == "https://api.sec.or.th/FundFactsheet/fund/amc/amc-1":
            return MagicMock(
                status_code=200,
                json=lambda: [{"proj_id": "p1", "proj_abbr_name": "FUND-A"}],
            )
        if url == "https://api.sec.or.th/FundFactsheet/fund/amc/amc-2":
            return MagicMock(
                status_code=200,
                json=lambda: [{"proj_id": "p2", "proj_abbr_name": "FUND-B"}],
            )
        raise AssertionError(f"unexpected url {url}")

    session.get.side_effect = fake_get

    client = _make_client(session)
    funds = client.list_funds()

    assert {f["proj_id"] for f in funds} == {"p1", "p2"}
```

Note: `MagicMock(status_code=200, ...)` objects returned by `fake_get` don't have `raise_for_status` wired to actually raise — add `raise_for_status=lambda: None` isn't needed since `MagicMock()` auto-creates a no-op callable attribute for `raise_for_status` by default.

- [ ] **Step 2: Run tests to verify the new one fails**

Run: `python -m pytest tests/test_thai_mf.py -v`
Expected: `test_list_funds_aggregates_across_amcs` fails with `AttributeError: 'SECFundClient' object has no attribute 'list_funds'`.

- [ ] **Step 3: Implement `list_funds`**

Add to the `SECFundClient` class in `thai_mf.py` (after `get_daily_nav`):

```python
    def list_funds(self):
        """Return a flat list of fund dicts (proj_id, proj_abbr_name,
        proj_name_th, proj_name_en, fund_status, ...) across every AMC.
        """
        resp = self.session.get(
            f"{FUND_FACTSHEET_BASE}/fund/amc", headers=self._headers(), timeout=30
        )
        resp.raise_for_status()
        amcs = resp.json()

        funds = []
        for amc in amcs:
            unique_id = amc.get("unique_id")
            if not unique_id:
                continue
            detail_resp = self.session.get(
                f"{FUND_FACTSHEET_BASE}/fund/amc/{unique_id}",
                headers=self._headers(),
                timeout=30,
            )
            detail_resp.raise_for_status()
            detail = detail_resp.json()
            if isinstance(detail, list):
                funds.extend(detail)
        return funds
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_thai_mf.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add thai_mf.py tests/test_thai_mf.py
git commit -m "feat: add SECFundClient.list_funds aggregating all AMCs"
```

---

### Task 4: Fund-list caching + `resolve_fund_id`

**Files:**
- Modify: `thai_mf.py`
- Test: `tests/test_thai_mf.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_thai_mf.py`:

```python
def test_resolve_fund_id_matches_abbr_name_case_insensitively(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.list_funds = MagicMock(
        return_value=[
            {"proj_id": "p1", "proj_abbr_name": "K-CHANGE-A(A)"},
            {"proj_id": "p2", "proj_abbr_name": "SCBGOLD"},
        ]
    )

    result = thai_mf.resolve_fund_id("k-change-a(a)", client)

    assert result == "p1"
    client.list_funds.assert_called_once()


def test_resolve_fund_id_returns_none_when_unmatched(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.list_funds = MagicMock(return_value=[{"proj_id": "p1", "proj_abbr_name": "SCBGOLD"}])

    result = thai_mf.resolve_fund_id("NOT-A-REAL-FUND", client)

    assert result is None


def test_resolve_fund_id_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.list_funds = MagicMock(return_value=[{"proj_id": "p1", "proj_abbr_name": "SCBGOLD"}])

    thai_mf.resolve_fund_id("SCBGOLD", client)
    thai_mf.resolve_fund_id("SCBGOLD", client)

    client.list_funds.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_thai_mf.py -v`
Expected: `AttributeError: module 'thai_mf' has no attribute 'resolve_fund_id'` (and `CACHE_DIR` missing).

- [ ] **Step 3: Implement fund-list caching and `resolve_fund_id`**

Add to `thai_mf.py` (top-level, after the existing imports/constants — add `import json`, `import time` is already imported, add `from pathlib import Path`):

```python
import json
from pathlib import Path
```

Add near the other constants:

```python
CACHE_DIR = Path(".cache/thai_mf")
FUND_LIST_MAX_AGE_DAYS = 7
```

Add after the `SECFundClient` class:

```python
def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fund_list_cache_path():
    return CACHE_DIR / "fund_list.json"


def _load_fund_list_cache():
    path = _fund_list_cache_path()
    if not path.exists():
        return None
    age_seconds = time.time() - path.stat().st_mtime
    if age_seconds > FUND_LIST_MAX_AGE_DAYS * 86400:
        return None
    with open(path) as f:
        return json.load(f)


def _save_fund_list_cache(funds):
    _ensure_cache_dir()
    with open(_fund_list_cache_path(), "w") as f:
        json.dump(funds, f)


def resolve_fund_id(name, client):
    """Resolve a fund short name (as typed after the MF: prefix) to its
    SEC proj_id. Matches case-insensitively against proj_abbr_name,
    proj_name_th, and proj_name_en. Returns None if no fund matches.
    """
    funds = _load_fund_list_cache()
    if funds is None:
        funds = client.list_funds()
        _save_fund_list_cache(funds)

    target = name.strip().upper()
    for fund in funds:
        candidates = (
            fund.get("proj_abbr_name"),
            fund.get("proj_name_th"),
            fund.get("proj_name_en"),
        )
        for candidate in candidates:
            if candidate and candidate.strip().upper() == target:
                return fund.get("proj_id")
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_thai_mf.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add thai_mf.py tests/test_thai_mf.py
git commit -m "feat: add fund-list caching and resolve_fund_id lookup"
```

---

### Task 5: NAV history caching — `get_nav_history`

**Files:**
- Modify: `thai_mf.py`
- Test: `tests/test_thai_mf.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_thai_mf.py`:

```python
import pandas as pd


def test_get_nav_history_fetches_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.get_daily_nav = MagicMock(
        side_effect=lambda proj_id, d: {
            dt.date(2024, 1, 1): 10.0,
            dt.date(2024, 1, 2): 10.5,
            dt.date(2024, 1, 3): None,  # e.g. weekend, no NAV published
        }[d]
    )

    series, incomplete = thai_mf.get_nav_history(
        client, "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3)
    )

    assert incomplete is False
    assert list(series.values) == [10.0, 10.5]
    assert client.get_daily_nav.call_count == 3
    assert (tmp_path / "nav_p1.csv").exists()


def test_get_nav_history_reuses_cache_for_already_fetched_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.get_daily_nav = MagicMock(return_value=10.0)

    thai_mf.get_nav_history(client, "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    assert client.get_daily_nav.call_count == 1

    thai_mf.get_nav_history(client, "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    assert client.get_daily_nav.call_count == 1  # no new calls, served from cache


def test_get_nav_history_stops_early_and_flags_incomplete_on_rate_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)

    def fake_get_daily_nav(proj_id, d):
        if d == dt.date(2024, 1, 1):
            return 10.0
        raise thai_mf.SECRateLimitError("rate limited")

    client.get_daily_nav = MagicMock(side_effect=fake_get_daily_nav)

    series, incomplete = thai_mf.get_nav_history(
        client, "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3)
    )

    assert incomplete is True
    assert list(series.values) == [10.0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_thai_mf.py -v`
Expected: `AttributeError: module 'thai_mf' has no attribute 'get_nav_history'`.

- [ ] **Step 3: Implement NAV caching and `get_nav_history`**

Add to `thai_mf.py`, after the `resolve_fund_id` function (add `import pandas as pd` alongside the other top-of-file imports):

```python
import pandas as pd
```

```python
def _nav_cache_path(proj_id):
    return CACHE_DIR / f"nav_{proj_id}.csv"


def _load_nav_cache(proj_id):
    path = _nav_cache_path(proj_id)
    if not path.exists():
        return pd.Series(dtype=float, name="nav")
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df["nav"]


def _save_nav_cache(proj_id, series):
    _ensure_cache_dir()
    series.rename("nav").rename_axis("date").to_csv(_nav_cache_path(proj_id))


def get_nav_history(client, proj_id, start_date, end_date):
    """Fetch NAV history for proj_id over [start_date, end_date] inclusive,
    reading/writing the on-disk cache so already-fetched dates aren't
    re-requested. Returns (series, incomplete) where series is indexed by
    date and incomplete is True if fetching stopped early due to sustained
    rate limiting (SEC has no bulk date-range endpoint, so this is one
    HTTP call per missing date).
    """
    cached = _load_nav_cache(proj_id)
    all_dates = pd.date_range(start_date, end_date, freq="D")
    known_dates = set(cached.index)
    missing_dates = [d for d in all_dates if d not in known_dates]

    incomplete = False
    updates = {}
    for ts in missing_dates:
        try:
            val = client.get_daily_nav(proj_id, ts.date())
        except SECRateLimitError:
            incomplete = True
            break
        if val is not None:
            updates[ts] = val

    if updates:
        cached = pd.concat([cached, pd.Series(updates)]).sort_index()
        _save_nav_cache(proj_id, cached)

    result = cached.loc[
        (cached.index >= pd.Timestamp(start_date)) & (cached.index <= pd.Timestamp(end_date))
    ]
    return result.sort_index(), incomplete
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_thai_mf.py -v`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add thai_mf.py tests/test_thai_mf.py
git commit -m "feat: add NAV history fetch with on-disk caching"
```

---

### Task 6: `split_symbols` and `merge_fund_navs` helpers

**Files:**
- Modify: `thai_mf.py`
- Test: `tests/test_thai_mf.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_thai_mf.py`:

```python
def test_split_symbols_separates_mf_prefixed_entries():
    yf_symbols, mf_symbols = thai_mf.split_symbols(["AMZN", "MF:K-CHANGE-A(A)", "SPY", "MF:SCBGOLD"])

    assert yf_symbols == ["AMZN", "SPY"]
    assert mf_symbols == ["K-CHANGE-A(A)", "SCBGOLD"]


def test_split_symbols_with_no_mf_entries():
    yf_symbols, mf_symbols = thai_mf.split_symbols(["AMZN", "SPY"])

    assert yf_symbols == ["AMZN", "SPY"]
    assert mf_symbols == []


def test_merge_fund_navs_outer_joins_and_fills():
    data_close = pd.DataFrame(
        {"AMZN": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    fund_navs = {
        "MF:SCBGOLD": pd.Series(
            [50.0, 51.0],
            index=pd.to_datetime(["2024-01-01", "2024-01-03"]),
        )
    }

    merged = thai_mf.merge_fund_navs(data_close, fund_navs)

    assert list(merged.columns) == ["AMZN", "MF:SCBGOLD"]
    assert merged.loc["2024-01-02", "MF:SCBGOLD"] == 50.0  # forward-filled
    assert merged.loc["2024-01-03", "MF:SCBGOLD"] == 51.0


def test_merge_fund_navs_returns_data_close_unchanged_when_no_funds():
    data_close = pd.DataFrame(
        {"AMZN": [100.0]}, index=pd.to_datetime(["2024-01-01"])
    )

    merged = thai_mf.merge_fund_navs(data_close, {})

    pd.testing.assert_frame_equal(merged, data_close)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_thai_mf.py -v`
Expected: `AttributeError: module 'thai_mf' has no attribute 'split_symbols'`.

- [ ] **Step 3: Implement `split_symbols` and `merge_fund_navs`**

Add to the end of `thai_mf.py`:

```python
MF_PREFIX = "MF:"


def split_symbols(stock_list):
    """Split a parsed symbol list into (yf_symbols, mf_symbols). mf_symbols
    have the MF: prefix stripped so callers can pass them straight to
    resolve_fund_id.
    """
    yf_symbols = []
    mf_symbols = []
    for symbol in stock_list:
        if symbol.startswith(MF_PREFIX):
            mf_symbols.append(symbol[len(MF_PREFIX):].strip())
        else:
            yf_symbols.append(symbol)
    return yf_symbols, mf_symbols


def merge_fund_navs(data_close, fund_navs):
    """Outer-join Thai fund NAV series (dict of display_symbol -> pd.Series)
    onto the yfinance data_close DataFrame, forward/backward filling to
    match the existing stock-data fill behavior. No-op if fund_navs is empty.
    """
    if not fund_navs:
        return data_close
    mf_df = pd.DataFrame(fund_navs)
    merged = data_close.join(mf_df, how="outer") if not data_close.empty else mf_df
    return merged.sort_index().ffill().bfill()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_thai_mf.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add thai_mf.py tests/test_thai_mf.py
git commit -m "feat: add split_symbols and merge_fund_navs helpers"
```

---

### Task 7: Wire Thai mutual funds into `app.py`

**Files:**
- Modify: `app.py:1` (imports)
- Modify: `app.py:14-38` (sidebar)
- Modify: `app.py:40-65` (symbol parsing, download, missing-symbol reporting)

- [ ] **Step 1: Add the import**

In `app.py`, change line 1-9:

```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from pypfopt import EfficientFrontier, CLA, plotting
import matplotlib.pyplot as plt
import io
```

to:

```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from pypfopt import EfficientFrontier, CLA, plotting
import matplotlib.pyplot as plt
import io

import thai_mf
```

- [ ] **Step 2: Add the SEC API key field and update captions in the sidebar**

Change (`app.py:27-38`):

```python
    total_cash = st.number_input("Total Cash (USD)", value=1_000_000, step=100_000)
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, step=0.01, format="%.4f")
    run_btn = st.button("Calculate", type="primary", use_container_width=True)

    st.divider()
    st.caption("⚠️ **Beta Version — ข้อควรระวัง**")
    st.caption("1. รองรับเฉพาะสินทรัพย์ที่มีใน Yahoo Finance เท่านั้น")
    st.caption("2. หุ้นไทยต้องเติม `.BK` หลังชื่อ เช่น `PTT.BK` หุ้น US ใส่ชื่อได้เลย")
    st.caption("3. Custom Weight รวมกันต้องเท่ากับ 1.0 เท่านั้น")
    st.caption("4. ตัวอย่าง: `AMZN, META, NVDA, SPY, LLY`")
    st.caption("5. ค่า Return, Vol, Sharpe ใน Expected กับ Backtest มีค่าใกล้เคียงกันแต่อาจต่างกันเล็กน้อย เนื่องจากคำนวณคนละวิธี")
    st.caption("6. หุ้นบางตัวอาจโหลดไม่สำเร็จ เพราะเขียนชื่อผิด หรือในปีนั้นยังไม่มีข้อมูล (ตรวจสอบชื่อและปีที่ดึงข้อมูลให้ดี)")
```

to:

```python
    total_cash = st.number_input("Total Cash (USD)", value=1_000_000, step=100_000)
    risk_free_rate = st.number_input("Risk-Free Rate", value=0.02, step=0.01, format="%.4f")
    sec_api_key = st.text_input(
        "SEC Open API Key (สำหรับกองทุนไทย)",
        value="",
        type="password",
        help="จำเป็นเฉพาะเมื่อกรอกกองทุนรวมไทยด้วย prefix MF: สมัครฟรีที่ api-portal.sec.or.th",
    )
    run_btn = st.button("Calculate", type="primary", use_container_width=True)

    st.divider()
    st.caption("⚠️ **Beta Version — ข้อควรระวัง**")
    st.caption("1. รองรับเฉพาะสินทรัพย์ที่มีใน Yahoo Finance เท่านั้น")
    st.caption("2. หุ้นไทยต้องเติม `.BK` หลังชื่อ เช่น `PTT.BK` หุ้น US ใส่ชื่อได้เลย")
    st.caption("3. Custom Weight รวมกันต้องเท่ากับ 1.0 เท่านั้น")
    st.caption("4. ตัวอย่าง: `AMZN, META, NVDA, SPY, LLY`")
    st.caption("5. ค่า Return, Vol, Sharpe ใน Expected กับ Backtest มีค่าใกล้เคียงกันแต่อาจต่างกันเล็กน้อย เนื่องจากคำนวณคนละวิธี")
    st.caption("6. หุ้นบางตัวอาจโหลดไม่สำเร็จ เพราะเขียนชื่อผิด หรือในปีนั้นยังไม่มีข้อมูล (ตรวจสอบชื่อและปีที่ดึงข้อมูลให้ดี)")
    st.caption("7. กองทุนรวมไทยใส่ prefix `MF:` เช่น `MF:K-CHANGE-A(A)` ข้อมูลมาจาก SEC Open API และต้องกรอก API Key ด้านบน (ครั้งแรกที่ดึงข้อมูลกองทุนใหม่จะช้า เพราะ SEC ไม่มี endpoint ดึงทีละช่วงวันที่)")
```

- [ ] **Step 3: Split symbols, fetch Thai funds, and merge into `data_close`**

Change (`app.py:40-65`):

```python
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
        data_close = df["Close"].ffill().bfill()
        data_close.dropna(how="all", inplace=True)
        data_close.dropna(axis=1, how="all", inplace=True)

    if data_close.empty:
        st.error("No data downloaded. Check symbols and date range.")
        st.stop()

    # แจ้งหุ้นที่โหลดสำเร็จ / ไม่สำเร็จ
    loaded = list(data_close.columns)
    missing = [s for s in stock_list if s not in loaded]
    if missing:
        st.warning(f"⚠️ ไม่พบข้อมูล: **{', '.join(missing)}** — ตรวจสอบชื่อ symbol อีกครั้ง")
    st.success(f"✅ โหลดสำเร็จ {len(loaded)} ตัว: **{', '.join(loaded)}**")
```

to:

```python
# ─── Parse symbols ───
stock_list = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

if run_btn and len(stock_list) >= 2:
    yf_symbols, mf_symbols = thai_mf.split_symbols(stock_list)

    if mf_symbols and not sec_api_key:
        st.error("⚠️ พบสัญลักษณ์กองทุนไทย (MF:) แต่ยังไม่ได้กรอก SEC Open API Key ในแถบด้านซ้าย")
        st.stop()

    with st.spinner("Downloading data..."):
        if yf_symbols:
            df = yf.download(
                tickers=yf_symbols,
                start=str(start_date),
                end=str(end_date),
                interval="1d",
                auto_adjust=True,
            )
            data_close = df["Close"].ffill().bfill()
            data_close.dropna(how="all", inplace=True)
            data_close.dropna(axis=1, how="all", inplace=True)
        else:
            data_close = pd.DataFrame()

    mf_missing = []
    mf_incomplete = []
    if mf_symbols:
        with st.spinner("Downloading Thai mutual fund data..."):
            client = thai_mf.SECFundClient(sec_api_key)
            fund_navs = {}
            for name in mf_symbols:
                display_symbol = f"MF:{name}"
                try:
                    proj_id = thai_mf.resolve_fund_id(name, client)
                    if proj_id is None:
                        mf_missing.append(display_symbol)
                        continue
                    nav_series, incomplete = thai_mf.get_nav_history(
                        client, proj_id, start_date, end_date
                    )
                except thai_mf.SECAPIError:
                    mf_missing.append(display_symbol)
                    continue
                if nav_series.empty:
                    mf_missing.append(display_symbol)
                    continue
                if incomplete:
                    mf_incomplete.append(display_symbol)
                fund_navs[display_symbol] = nav_series
            data_close = thai_mf.merge_fund_navs(data_close, fund_navs)

    if data_close.empty:
        st.error("No data downloaded. Check symbols and date range.")
        st.stop()

    # แจ้งหุ้นที่โหลดสำเร็จ / ไม่สำเร็จ
    loaded = list(data_close.columns)
    missing = [s for s in yf_symbols if s not in loaded] + mf_missing
    if missing:
        st.warning(f"⚠️ ไม่พบข้อมูล: **{', '.join(missing)}** — ตรวจสอบชื่อ symbol อีกครั้ง")
    if mf_incomplete:
        st.warning(
            f"⚠️ ข้อมูล NAV อาจไม่ครบทุกวันสำหรับ: **{', '.join(mf_incomplete)}** "
            "(SEC API rate limit ระหว่างดึงข้อมูล ลองกด Calculate ซ้ำเพื่อดึงวันที่เหลือจาก cache)"
        )
    st.success(f"✅ โหลดสำเร็จ {len(loaded)} ตัว: **{', '.join(loaded)}**")
```

- [ ] **Step 4: Regression check — existing yfinance-only flow still works**

Run: `streamlit run app.py`

In the browser: leave the symbols box at its default (`AMZN, META, LLY, SPY, NVDA, GOOGL`), leave the new "SEC Open API Key" field empty, click **Calculate**.

Expected: identical behavior to before this change — data downloads, success/warning banners show, all four tabs render. No error about the missing API key (none of the symbols use the `MF:` prefix, so the key isn't required).

- [ ] **Step 5: Manual smoke test with a real Thai fund (requires the user's own SEC subscription key)**

This step needs a real key from `api-portal.sec.or.th` and can't be done by an agent. Hand off to the user:

In the browser: enter `AMZN, MF:<a real proj_abbr_name from SEC>` in the symbols box, paste a real SEC Open API key into the new field, click **Calculate**.

Expected: either the fund resolves and its NAV series merges into `data_close` (all tabs work as normal, the fund shows up as a `MF:`-prefixed column), or — if `proj_abbr_name`/`last_val` turn out not to match the live API's actual field names — a clear `ไม่พบข้อมูล` / error rather than a crash. If field names are wrong, fix them in `resolve_fund_id`'s `candidates` list and `get_daily_nav`'s `.get("last_val")` in `thai_mf.py` (see the note at the top of this plan), then re-run this step.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: integrate Thai mutual fund data (MF: prefix) into portfolio app"
```

---

### Task 8: Final full-suite check

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `python -m pytest -v`
Expected: all tests from Tasks 1–6 pass (16 total: 1 sanity + 15 in `test_thai_mf.py`), 0 failures.

- [ ] **Step 2: Confirm nothing else references the old single-source assumption**

Run: `grep -n "stock_list" app.py`
Expected: `stock_list` is only used to build `yf_symbols`/`mf_symbols` via `thai_mf.split_symbols` and in the sidebar caption examples — no remaining direct `yf.download(tickers=stock_list, ...)` call (that would re-introduce Thai fund names into the yfinance request and break it).

No commit needed for this task — it's a verification pass over work already committed in Tasks 1–7.
