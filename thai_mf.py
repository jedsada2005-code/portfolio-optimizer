import json
import time
from pathlib import Path

import pandas as pd
import requests

FUND_FACTSHEET_BASE = "https://api.sec.or.th/FundFactsheet"
FUND_DAILY_INFO_BASE = "https://api.sec.or.th/FundDailyInfo"

MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2

CACHE_DIR = Path(".cache/thai_mf")
FUND_LIST_MAX_AGE_DAYS = 7


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

    def _get(self, url):
        """GET url, normalizing network-level failures (connection errors,
        timeouts, etc.) into SECAPIError so callers only ever need to catch
        that one exception type for HTTP-calling methods.
        """
        try:
            return self.session.get(url, headers=self._headers(), timeout=30)
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

    def get_daily_nav(self, proj_id, nav_date):
        """Return the NAV per unit for proj_id on nav_date (a date object),
        or None if no NAV was published that day.
        """
        url = f"{FUND_DAILY_INFO_BASE}/{proj_id}/dailynav/{nav_date.isoformat()}"
        attempts = 0
        while True:
            resp = self._get(url)
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

    def list_funds(self):
        """Return a flat list of fund dicts (proj_id, proj_abbr_name,
        proj_name_th, proj_name_en, fund_status, ...) across every AMC.
        """
        amc_list_url = f"{FUND_FACTSHEET_BASE}/fund/amc"
        resp = self._get(amc_list_url)
        self._raise_for_status(resp, amc_list_url)
        amcs = resp.json()

        funds = []
        for amc in amcs:
            unique_id = amc.get("unique_id")
            if not unique_id:
                continue
            detail_url = f"{FUND_FACTSHEET_BASE}/fund/amc/{unique_id}"
            detail_resp = self._get(detail_url)
            self._raise_for_status(detail_resp, detail_url)
            detail = detail_resp.json()
            if isinstance(detail, list):
                funds.extend(detail)
        return funds


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
