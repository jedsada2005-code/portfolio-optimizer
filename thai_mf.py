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

    def get_nav_range(self, proj_id, start_date, end_date):
        """Fetch all daily NAV records for proj_id between start_date and
        end_date (inclusive), paginating via next_cursor until exhausted.
        Uses the Fund Daily Info API subscription key. Returns
        (items, incomplete) — the item dicts collected so far, and whether
        pagination was cut short by sustained rate limiting (in which case
        items holds whatever was fetched before that point).
        """
        items = []
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

            resp, incomplete = self._get_with_retry(url, self.daily_info_key)
            if incomplete:
                return items, True

            self._raise_for_status(resp, url)
            payload = self._parse_json_object(resp, url)
            items.extend(payload.get("items", []))
            next_cursor = payload.get("next_cursor")
            if not next_cursor:
                break
        return items, False


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


def _nav_cache_path(proj_id):
    return CACHE_DIR / f"nav_{proj_id}.csv"


def _nav_meta_path(proj_id):
    return CACHE_DIR / f"nav_{proj_id}.meta.json"


def _load_nav_cache(proj_id):
    path = _nav_cache_path(proj_id)
    if not path.exists():
        return pd.Series(dtype=float, name="nav")
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    return df["nav"]


def _save_nav_cache(proj_id, series):
    _ensure_cache_dir()
    series.rename("nav").rename_axis("date").to_csv(_nav_cache_path(proj_id))


def _load_nav_meta(proj_id):
    """Return (fetched_start, fetched_end) as date objects for the date
    range already fully fetched for proj_id, or None if never fetched.
    """
    path = _nav_meta_path(proj_id)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return date.fromisoformat(data["start_date"]), date.fromisoformat(data["end_date"])


def _save_nav_meta(proj_id, start_date, end_date):
    _ensure_cache_dir()
    with open(_nav_meta_path(proj_id), "w") as f:
        json.dump({"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}, f)


def get_nav_history(client, proj_id, start_date, end_date):
    """Fetch NAV history for proj_id over [start_date, end_date] inclusive,
    reading/writing the on-disk cache so a range that's already been fully
    fetched isn't re-requested. Returns (series, incomplete) where series
    is indexed by date and incomplete is True if fetching was interrupted
    by sustained rate limiting (whatever was fetched is still cached and
    returned; incomplete signals the caller that some dates in range may
    be missing).
    """
    fetched_range = _load_nav_meta(proj_id)
    if fetched_range and fetched_range[0] <= start_date and fetched_range[1] >= end_date:
        cached = _load_nav_cache(proj_id)
        result = cached.loc[
            (cached.index >= pd.Timestamp(start_date)) & (cached.index <= pd.Timestamp(end_date))
        ]
        return result.sort_index(), False

    items, incomplete = client.get_nav_range(proj_id, start_date, end_date)

    cached = _load_nav_cache(proj_id)
    updates = {
        pd.Timestamp(item["nav_date"]): item["last_val"]
        for item in items
        if item.get("last_val") is not None
    }
    if updates:
        cached = pd.concat([cached, pd.Series(updates)])
        cached = cached[~cached.index.duplicated(keep="last")].sort_index()
        _save_nav_cache(proj_id, cached)

    if not incomplete:
        _save_nav_meta(proj_id, start_date, end_date)

    result = cached.loc[
        (cached.index >= pd.Timestamp(start_date)) & (cached.index <= pd.Timestamp(end_date))
    ]
    return result.sort_index(), incomplete


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
