from unittest.mock import MagicMock, patch
import datetime as dt
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests

import thai_mf


def _make_client(session):
    return thai_mf.SECFundClient(
        "dummy-factsheet-key", "dummy-daily-info-key", session=session
    )


def _query(url):
    return parse_qs(urlparse(url).query)


def test_list_funds_paginates_via_next_cursor():
    session = MagicMock()

    def fake_get(url, headers, timeout):
        q = _query(url)
        if "next_cursor" not in q:
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "message": "success",
                    "next_cursor": "page-2",
                    "items": [{"proj_id": "p1", "proj_abbr_name": "FUND-A"}],
                },
            )
        if q["next_cursor"] == ["page-2"]:
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "message": "success",
                    "next_cursor": "",
                    "items": [{"proj_id": "p2", "proj_abbr_name": "FUND-B"}],
                },
            )
        raise AssertionError(f"unexpected url {url}")

    session.get.side_effect = fake_get
    client = _make_client(session)

    funds = client.list_funds()

    assert {f["proj_id"] for f in funds} == {"p1", "p2"}
    assert session.get.call_count == 2
    first_url = session.get.call_args_list[0].args[0]
    assert first_url.startswith("https://api.sec.or.th/v2/fund/general-info/profiles?")


def test_list_funds_sends_factsheet_key_not_daily_info_key():
    session = MagicMock()
    session.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"message": "success", "next_cursor": "", "items": []},
    )
    client = _make_client(session)

    client.list_funds()

    sent_headers = session.get.call_args.kwargs["headers"]
    assert sent_headers == {"Ocp-Apim-Subscription-Key": "dummy-factsheet-key"}


def test_get_nav_range_sends_daily_info_key_not_factsheet_key():
    session = MagicMock()
    session.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {"message": "success", "next_cursor": "", "items": []},
    )
    client = _make_client(session)

    client.get_nav_range("p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3))

    sent_headers = session.get.call_args.kwargs["headers"]
    assert sent_headers == {"Ocp-Apim-Subscription-Key": "dummy-daily-info-key"}


def test_list_funds_raises_sec_api_error_on_http_error():
    session = MagicMock()
    error_resp = MagicMock(status_code=401)
    error_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Client Error")
    session.get.return_value = error_resp
    client = _make_client(session)

    try:
        client.list_funds()
        assert False, "expected SECAPIError"
    except thai_mf.SECAPIError:
        pass


def test_list_funds_raises_sec_api_error_on_connection_error():
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("connection refused")
    client = _make_client(session)

    try:
        client.list_funds()
        assert False, "expected SECAPIError"
    except thai_mf.SECAPIError:
        pass


def test_list_funds_raises_sec_api_error_when_response_body_is_not_an_object():
    # Reproduces a live-API failure mode: a 200 response whose JSON body
    # decodes to a plain string instead of the documented {"items": [...]}
    # object, which used to crash with a raw AttributeError deep in the
    # payload.get(...) call.
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=200, json=lambda: "unexpected string body")
    client = _make_client(session)

    try:
        client.list_funds()
        assert False, "expected SECAPIError"
    except thai_mf.SECAPIError:
        pass


def test_get_nav_range_paginates_and_returns_items():
    session = MagicMock()

    def fake_get(url, headers, timeout):
        q = _query(url)
        assert q["proj_id"] == ["p1"]
        assert q["start_nav_date"] == ["2024-01-01"]
        assert q["end_nav_date"] == ["2024-01-03"]
        if "next_cursor" not in q:
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "message": "success",
                    "next_cursor": "page-2",
                    "items": [
                        {"proj_id": "p1", "nav_date": "2024-01-01", "last_val": 10.0}
                    ],
                },
            )
        if q["next_cursor"] == ["page-2"]:
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "message": "success",
                    "next_cursor": "",
                    "items": [
                        {"proj_id": "p1", "nav_date": "2024-01-02", "last_val": 10.5}
                    ],
                },
            )
        raise AssertionError(f"unexpected url {url}")

    session.get.side_effect = fake_get
    client = _make_client(session)

    items, incomplete = client.get_nav_range("p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3))

    assert incomplete is False
    assert [item["nav_date"] for item in items] == ["2024-01-01", "2024-01-02"]


def test_get_nav_range_raises_sec_api_error_on_http_error():
    session = MagicMock()
    error_resp = MagicMock(status_code=500)
    error_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
    session.get.return_value = error_resp
    client = _make_client(session)

    try:
        client.get_nav_range("p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3))
        assert False, "expected SECAPIError"
    except thai_mf.SECAPIError:
        pass


def test_get_nav_range_raises_sec_api_error_on_connection_error():
    session = MagicMock()
    session.get.side_effect = requests.exceptions.Timeout("timed out")
    client = _make_client(session)

    try:
        client.get_nav_range("p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3))
        assert False, "expected SECAPIError"
    except thai_mf.SECAPIError:
        pass


def test_get_nav_range_raises_sec_api_error_when_response_body_is_not_an_object():
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=200, json=lambda: "unexpected string body")
    client = _make_client(session)

    try:
        client.get_nav_range("p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3))
        assert False, "expected SECAPIError"
    except thai_mf.SECAPIError:
        pass


@patch("thai_mf.time.sleep", return_value=None)
def test_get_nav_range_returns_incomplete_on_sustained_rate_limit(mock_sleep):
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=429)
    client = _make_client(session)

    items, incomplete = client.get_nav_range("p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3))

    assert incomplete is True
    assert items == []
    assert session.get.call_count == thai_mf.MAX_RETRIES
    assert mock_sleep.call_count == thai_mf.MAX_RETRIES - 1


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


def test_get_nav_history_fetches_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.get_nav_range = MagicMock(
        return_value=(
            [
                {"proj_id": "p1", "nav_date": "2024-01-01", "last_val": 10.0},
                {"proj_id": "p1", "nav_date": "2024-01-02", "last_val": 10.5},
            ],
            False,
        )
    )

    series, incomplete = thai_mf.get_nav_history(
        client, "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3)
    )

    assert incomplete is False
    assert list(series.values) == [10.0, 10.5]
    client.get_nav_range.assert_called_once_with(
        "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3)
    )
    assert (tmp_path / "nav_p1.csv").exists()
    assert (tmp_path / "nav_p1.meta.json").exists()


def test_get_nav_history_reuses_cache_for_already_fetched_range(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.get_nav_range = MagicMock(
        return_value=([{"proj_id": "p1", "nav_date": "2024-01-01", "last_val": 10.0}], False)
    )

    thai_mf.get_nav_history(client, "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    assert client.get_nav_range.call_count == 1

    thai_mf.get_nav_history(client, "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    assert client.get_nav_range.call_count == 1  # no new call, served from cache


def test_get_nav_history_refetches_when_requested_range_extends_beyond_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.get_nav_range = MagicMock(
        return_value=([{"proj_id": "p1", "nav_date": "2024-01-01", "last_val": 10.0}], False)
    )

    thai_mf.get_nav_history(client, "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    assert client.get_nav_range.call_count == 1

    thai_mf.get_nav_history(client, "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert client.get_nav_range.call_count == 2  # wider range not covered by cache


def test_get_nav_history_flags_incomplete_and_keeps_partial_results_on_rate_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.get_nav_range = MagicMock(
        return_value=([{"proj_id": "p1", "nav_date": "2024-01-01", "last_val": 10.0}], True)
    )

    series, incomplete = thai_mf.get_nav_history(
        client, "p1", dt.date(2024, 1, 1), dt.date(2024, 1, 3)
    )

    assert incomplete is True
    assert list(series.values) == [10.0]
    # an incomplete fetch must not mark the range as fully cached
    assert not (tmp_path / "nav_p1.meta.json").exists()


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


def test_merge_fund_navs_does_not_backfill_before_funds_first_nav_date():
    # A fund that started trading well after the rest of the portfolio's
    # history (e.g. a newly registered Thai fund) must stay NaN for dates
    # before its own first NAV, not get a fake flat price backfilled in —
    # that would collapse its variance to ~0 for that whole span and can
    # make the portfolio optimizer's covariance matrix singular.
    data_close = pd.DataFrame(
        {"AMZN": [100.0, 101.0, 102.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    )
    fund_navs = {
        "MF:NEWFUND": pd.Series(
            [50.0],
            index=pd.to_datetime(["2024-01-03"]),
        )
    }

    merged = thai_mf.merge_fund_navs(data_close, fund_navs)

    assert pd.isna(merged.loc["2024-01-01", "MF:NEWFUND"])
    assert pd.isna(merged.loc["2024-01-02", "MF:NEWFUND"])
    assert merged.loc["2024-01-03", "MF:NEWFUND"] == 50.0


def test_merge_fund_navs_returns_data_close_unchanged_when_no_funds():
    data_close = pd.DataFrame(
        {"AMZN": [100.0]}, index=pd.to_datetime(["2024-01-01"])
    )

    merged = thai_mf.merge_fund_navs(data_close, {})

    pd.testing.assert_frame_equal(merged, data_close)
