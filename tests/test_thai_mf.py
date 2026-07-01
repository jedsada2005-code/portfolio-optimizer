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
