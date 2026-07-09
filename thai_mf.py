import json
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

FUND_PROFILES_URL = "https://api.sec.or.th/v2/fund/general-info/profiles"
FUND_NAV_URL = "https://api.sec.or.th/v2/fund/daily-info/nav"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
PAGE_SIZE = 100

CACHE_DIR = Path(".cache/thai_mf")
FUND_LIST_MAX_AGE_DAYS = 7


class SECAPIError(Exception):
    """Raised when the SEC Open API returns a persistent, non-recoverable error."""


class SECRateLimitError(SECAPIError):
    """Raised when the SEC Open API keeps returning HTTP 429 after retries."""


class SECFundClient:
    """SEC Open Data issues a separate subscription key per API product —
    the Fund Factsheet API (fund lookup) and the Fund Daily Info API (NAV
    history) are subscribed to and keyed independently, even though both
    sit under the same api.sec.or.th host.
    """

    def __init__(self, factsheet_key, daily_info_key, session=None):
        self.factsheet_key = factsheet_key
        self.daily_info_key = daily_info_key
        self.session = session or requests.Session()

    @staticmethod
    def _headers(key):
        return {"Ocp-Apim-Subscription-Key": key}

    def _get(self, url, key):
        """GET url, normalizing network-level failures (connection errors,
        timeouts, etc.) into SECAPIError so callers only ever need to catch
        that one exception type for HTTP-calling methods.
        """
        try:
            return self.session.get(url, headers=self._headers(key), timeout=30)
        except requests.exceptions.RequestException as exc:
            raise SECAPIError(f"Network error calling SEC API at {url}: {exc}") from exc

    def _raise_for_status(self, resp, url):
        """Like resp.raise_for_status(), but re-raises as SECAPIError so
        callers only ever need to catch that one exception type.
        """
        try:
            resp.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise SECAPIError(
                f"SEC API error {resp.status_code} calling {url}: {exc}"
            ) from exc

    def _parse_json_object(self, resp, url):
        """Parse resp as JSON and require the top-level value to be an
        object (dict) with an "items" list, matching every paginated SEC
        endpoint's documented shape. Raises SECAPIError (instead of a raw
        AttributeError/TypeError further down the call chain) if the API
        ever returns something else — e.g. a bare string or list — and
        logs the raw payload to stderr so it shows up in deployment logs
        for diagnosis.
        """
        try:
            payload = resp.json()
        except ValueError as exc:
            print(
                f"[thai_mf] Non-JSON response from {url}: {resp.text!r}",
                file=sys.stderr,
            )
            raise SECAPIError(f"Non-JSON response from {url}: {exc}") from exc
        if not isinstance(payload, dict):
            print(
                f"[thai_mf] Unexpected response shape from {url}: {payload!r}",
                file=sys.stderr,
            )
            raise SECAPIError(f"Unexpected response shape from {url}: {payload!r}")
        return payload

    def _get_with_retry(self, url, key):
        """GET url, retrying on HTTP 429 with backoff. Returns (resp, incomplete)
        where incomplete is True if retries were exhausted while still rate
        limited (resp is the last, still-429 response in that case).
        """
        attempts = 0
        while True:
            resp = self._get(url, key)
            if resp.status_code != 429:
                return resp, False
            attempts += 1
            if attempts >= MAX_RETRIES:
                return resp, True
            time.sleep(RETRY_BACKOFF_SECONDS * attempts)

    def list_funds(self):
        """Return a flat list of fund dicts (proj_id, proj_abbr_name,
        proj_name_th, proj_name_en, fund_status, ...) by paginating through
        the SEC fund general-info/profiles endpoint. Uses the Fund
        Factsheet API subscription key.
        """
        funds = []
        next_cursor = None
        while True:
            params = {"page_size": PAGE_SIZE}
            if next_cursor:
                params["next_cursor"] = next_cursor
            url = f"{FUND_PROFILES_URL}?{urlencode(params)}"
            resp = self._get(url, self.factsheet_key)
            self._raise_for_status(resp, url)
            payload = self._parse_json_object(resp, url)
            funds.extend(payload.get("items", []))
            next_cursor = payload.get("next_cursor")
            if not next_cursor:
                break
        return funds

    @staticmethod
    def _select_class_items(items, preferred_class):
        """From NAV rows that may span several share classes of one
        proj_id, return only the rows of a single class.

        A single proj_id can cover several share classes (accumulation,
        dividend, currency-hedged, ...) whose NAVs differ by a large
        factor; blending them into one series manufactures enormous
        phantom day-over-day returns that corrupt any backtest, so exactly
        one class must be kept.

        preferred_class is only a hint (it may be None when the user typed
        a fund-level name, and even when set it may not appear in the NAV
        feed because the NAV and profiles feeds use different labels —
        e.g. K-CASH is "K-CASH-A" in profiles but "main" in NAV).
        Selection order:
          1. preferred_class, if set and present in the NAV rows;
          2. otherwise "main", if present;
          3. otherwise, among the fund's classes, prefer an accumulation
             class over a dividend-paying one (a dividend class's NAV
             drops on each payout and understates total return, which
             would undercount a backtest), then the class with the most
             rows (longest history), ties broken alphabetically.
        Whatever the path, exactly one class is returned — never a mix.
        """
        by_class = {}
        for item in items:
            by_class.setdefault(item.get("fund_class_name"), []).append(item)
        if not by_class:
            return []
        if preferred_class is not None and preferred_class in by_class:
            chosen = preferred_class
        elif "main" in by_class:
            chosen = "main"
        else:
            def rank(cls):
                label = (cls or "").strip().upper()
                is_dividend = label.endswith("(D)") or label.endswith("-D")
                return (is_dividend, -len(by_class[cls]), str(cls))

            chosen = sorted(by_class, key=rank)[0]
        return by_class[chosen]

    def get_nav_range(self, proj_id, fund_class_name, start_date, end_date):
        """Fetch daily NAV records for a single share class of proj_id
        between start_date and end_date inclusive, paginating via
        next_cursor until exhausted. Uses the Fund Daily Info API key.

        All classes are fetched (the endpoint's own fund_class_name filter
        is NOT used — it returns an empty body for funds whose NAV class
        labels differ from the profiles labels, e.g. K-CASH), then exactly
        one class is selected via _select_class_items so a mix of classes
        can never leak into one price series.

        Returns (items, incomplete) — the single-class item dicts, and
        whether pagination was cut short by sustained rate limiting (in
        which case items holds whatever class-selected rows were fetched
        before that point).
        """
        raw_items = []
        incomplete = False
        next_cursor = None
        while True:
            params = {
                "proj_id": proj_id,
                "start_nav_date": start_date.isoformat(),
                "end_nav_date": end_date.isoformat(),
                "page_size": PAGE_SIZE,
            }
            if next_cursor:
                params["next_cursor"] = next_cursor
            url = f"{FUND_NAV_URL}?{urlencode(params)}"

            resp, hit_rate_limit = self._get_with_retry(url, self.daily_info_key)
            if hit_rate_limit:
                incomplete = True
                break

            self._raise_for_status(resp, url)
            payload = self._parse_json_object(resp, url)
            raw_items.extend(payload.get("items", []))
            next_cursor = payload.get("next_cursor")
            if not next_cursor:
                break
        return self._select_class_items(raw_items, fund_class_name), incomplete


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
    """Resolve a fund name (as typed after the MF: prefix) to a
    (proj_id, preferred_class) pair. Returns (None, None) if no fund
    matches.

    Matching is case-insensitive and tries, in order:
      1. An exact share-class name (fund_class_name) — lets the user pin
         an exact class, e.g. "K-GOLD-A(D)". preferred_class is that class.
      2. The fund-level abbreviation / Thai / English name
         (proj_abbr_name / proj_name_th / proj_name_en). This is
         ambiguous across the fund's share classes, so preferred_class is
         returned as None, meaning "let the NAV layer pick the primary
         class" (see SECFundClient._select_class_items) — the profiles
         feed's own class labels are unreliable for this (they can be
         typo'd, and differ from the NAV feed's labels), whereas the
         class with the most NAV history is a robust primary-class proxy.
    """
    funds = _load_fund_list_cache()
    if funds is None:
        funds = client.list_funds()
        _save_fund_list_cache(funds)

    target = name.strip().upper()

    # 1. Exact share-class match -> pin that class.
    for fund in funds:
        cls = fund.get("fund_class_name")
        if cls and cls.strip().upper() == target:
            return fund.get("proj_id"), cls

    # 2. Fund-level name match -> defer class choice to the NAV layer.
    for fund in funds:
        if any(
            (fund.get(field) or "").strip().upper() == target
            for field in ("proj_abbr_name", "proj_name_th", "proj_name_en")
        ):
            return fund.get("proj_id"), None

    return None, None


def _cache_key(proj_id, fund_class_name):
    """Build a filesystem-safe cache key for one share class of a fund.
    NAV is cached per (proj_id, fund_class_name) because different classes
    of the same proj_id have different NAV series. fund_class_name is None
    when the class is auto-selected from the NAV data (fund-level match).
    """
    label = fund_class_name if fund_class_name is not None else "auto"
    slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    return f"{proj_id}__{slug}"


def _nav_cache_path(proj_id, fund_class_name):
    return CACHE_DIR / f"nav_{_cache_key(proj_id, fund_class_name)}.csv"


def _nav_meta_path(proj_id, fund_class_name):
    return CACHE_DIR / f"nav_{_cache_key(proj_id, fund_class_name)}.meta.json"


def _load_nav_cache(proj_id, fund_class_name):
    path = _nav_cache_path(proj_id, fund_class_name)
    if not path.exists():
        return pd.Series(dtype=float, name="nav")
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df["nav"]


def _save_nav_cache(proj_id, fund_class_name, series):
    _ensure_cache_dir()
    series.rename("nav").rename_axis("date").to_csv(_nav_cache_path(proj_id, fund_class_name))


def _load_nav_meta(proj_id, fund_class_name):
    """Return (fetched_start, fetched_end, chosen_class) for the date
    range already fully fetched for this class key, or None if never
    fetched. chosen_class is the actual NAV class stored (which may differ
    from the requested fund_class_name).
    """
    path = _nav_meta_path(proj_id, fund_class_name)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return (
        date.fromisoformat(data["start_date"]),
        date.fromisoformat(data["end_date"]),
        data.get("chosen_class"),
    )


def _save_nav_meta(proj_id, fund_class_name, start_date, end_date, chosen_class):
    _ensure_cache_dir()
    with open(_nav_meta_path(proj_id, fund_class_name), "w") as f:
        json.dump(
            {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "chosen_class": chosen_class,
            },
            f,
        )


def get_nav_history(client, proj_id, fund_class_name, start_date, end_date):
    """Fetch NAV history for one share class of proj_id over
    [start_date, end_date] inclusive, reading/writing the on-disk cache so
    a range that's already been fully fetched isn't re-requested.

    fund_class_name is the preferred class (may be None for a fund-level
    match, in which case the primary class is auto-selected from the NAV
    data). Returns (series, incomplete, chosen_class):
      - series indexed by date (a single class, never a mix);
      - incomplete True if fetching was interrupted by sustained rate
        limiting (whatever was fetched is still cached and returned);
      - chosen_class the actual NAV class label used.
    """
    fetched = _load_nav_meta(proj_id, fund_class_name)
    if fetched and fetched[0] <= start_date and fetched[1] >= end_date:
        cached = _load_nav_cache(proj_id, fund_class_name)
        result = cached.loc[
            (cached.index >= pd.Timestamp(start_date)) & (cached.index <= pd.Timestamp(end_date))
        ]
        return result.sort_index(), False, fetched[2]

    items, incomplete = client.get_nav_range(proj_id, fund_class_name, start_date, end_date)
    chosen_class = items[0].get("fund_class_name") if items else fund_class_name

    cached = _load_nav_cache(proj_id, fund_class_name)
    updates = {
        pd.Timestamp(item["nav_date"]): item["last_val"]
        for item in items
        if item.get("last_val") is not None
    }
    if updates:
        cached = pd.concat([cached, pd.Series(updates)])
        cached = cached[~cached.index.duplicated(keep="last")].sort_index()
        _save_nav_cache(proj_id, fund_class_name, cached)

    if not incomplete:
        _save_nav_meta(proj_id, fund_class_name, start_date, end_date, chosen_class)

    result = cached.loc[
        (cached.index >= pd.Timestamp(start_date)) & (cached.index <= pd.Timestamp(end_date))
    ]
    return result.sort_index(), incomplete, chosen_class


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
    onto the yfinance data_close DataFrame, forward-filling gaps (e.g.
    non-trading days) within each asset's own history. No-op if fund_navs
    is empty.

    Deliberately does NOT back-fill: a fund's dates before its own first
    published NAV are left as NaN rather than papered over with its
    earliest known price. Back-filling a flat price across a fund's entire
    pre-inception period makes its computed variance/covariance collapse
    toward zero, which can produce a near-singular covariance matrix and
    break the portfolio optimizer for any fund significantly younger than
    the other assets in the selected date range.
    """
    if not fund_navs:
        return data_close
    mf_df = pd.DataFrame(fund_navs)
    merged = data_close.join(mf_df, how="outer") if not data_close.empty else mf_df
    return merged.sort_index().ffill()
