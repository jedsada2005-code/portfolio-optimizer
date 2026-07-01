import json
import time
from pathlib import Path

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
