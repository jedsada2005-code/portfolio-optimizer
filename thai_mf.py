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
