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

    client.get_nav_range("p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 3))

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
        # the endpoint's own class filter is intentionally NOT sent (it
        # returns empty for some funds); selection happens client-side
        assert "fund_class_name" not in q
        assert q["start_nav_date"] == ["2024-01-01"]
        assert q["end_nav_date"] == ["2024-01-03"]
        if "next_cursor" not in q:
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "message": "success",
                    "next_cursor": "page-2",
                    "items": [
                        {"proj_id": "p1", "fund_class_name": "A(A)",
                         "nav_date": "2024-01-01", "last_val": 10.0}
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
                        {"proj_id": "p1", "fund_class_name": "A(A)",
                         "nav_date": "2024-01-02", "last_val": 10.5}
                    ],
                },
            )
        raise AssertionError(f"unexpected url {url}")

    session.get.side_effect = fake_get
    client = _make_client(session)

    items, incomplete = client.get_nav_range("p1", "A(A)", dt.date(2024, 1, 1), dt.date(2024, 1, 3))

    assert incomplete is False
    assert [item["nav_date"] for item in items] == ["2024-01-01", "2024-01-02"]


def test_get_nav_range_filters_out_other_share_classes():
    # Regression: a single proj_id spans multiple share classes whose NAVs
    # differ hugely (e.g. K-GOLD-A(A) ~13.8 vs K-GOLD-C(A) ~63.5). The
    # endpoint may return rows for several classes; only the requested
    # class must survive, otherwise the NAV series jumps between classes
    # and manufactures enormous fake day-over-day returns.
    session = MagicMock()
    session.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "message": "success",
            "next_cursor": "",
            "items": [
                {"proj_id": "p1", "fund_class_name": "K-GOLD-A(A)",
                 "nav_date": "2024-04-29", "last_val": 13.80},
                {"proj_id": "p1", "fund_class_name": "K-GOLD-C(A)",
                 "nav_date": "2024-04-29", "last_val": 63.51},
                {"proj_id": "p1", "fund_class_name": "K-GOLD-A(D)",
                 "nav_date": "2024-04-29", "last_val": 11.68},
            ],
        },
    )
    client = _make_client(session)

    items, incomplete = client.get_nav_range(
        "p1", "K-GOLD-A(A)", dt.date(2024, 4, 29), dt.date(2024, 4, 29)
    )

    assert incomplete is False
    assert len(items) == 1
    assert items[0]["fund_class_name"] == "K-GOLD-A(A)"
    assert items[0]["last_val"] == 13.80


def test_get_nav_range_falls_back_to_main_when_preferred_class_absent():
    # The NAV feed labels some funds' class differently from the profiles
    # feed (e.g. K-CASH is "K-CASH-A" in profiles but "main" in NAV). When
    # the preferred class isn't present, fall back to "main" rather than
    # returning nothing.
    session = MagicMock()
    session.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "message": "success",
            "next_cursor": "",
            "items": [
                {"proj_id": "p1", "fund_class_name": "main",
                 "nav_date": "2024-01-01", "last_val": 13.60},
            ],
        },
    )
    client = _make_client(session)

    items, incomplete = client.get_nav_range(
        "p1", "K-CASH-A", dt.date(2024, 1, 1), dt.date(2024, 1, 1)
    )

    assert [i["fund_class_name"] for i in items] == ["main"]


def test_get_nav_range_falls_back_to_largest_class_when_no_match_and_no_main():
    session = MagicMock()
    session.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "message": "success",
            "next_cursor": "",
            "items": [
                {"proj_id": "p1", "fund_class_name": "X-B", "nav_date": "2024-01-01", "last_val": 5.0},
                {"proj_id": "p1", "fund_class_name": "X-A", "nav_date": "2024-01-01", "last_val": 9.0},
                {"proj_id": "p1", "fund_class_name": "X-A", "nav_date": "2024-01-02", "last_val": 9.1},
            ],
        },
    )
    client = _make_client(session)

    items, incomplete = client.get_nav_range(
        "p1", "SOMETHING-ELSE", dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )

    # X-A has the most rows -> selected; never a mix of X-A and X-B
    assert {i["fund_class_name"] for i in items} == {"X-A"}
    assert len(items) == 2


def test_get_nav_range_prefers_accumulation_over_dividend_class():
    # A dividend class ("-D"/"(D)") often has MORE history, but its NAV
    # drops on payouts and understates total return, so an accumulation
    # class must win even with fewer rows.
    session = MagicMock()
    session.get.return_value = MagicMock(
        status_code=200,
        json=lambda: {
            "message": "success",
            "next_cursor": "",
            "items": [
                {"proj_id": "p1", "fund_class_name": "K-GOLD-A(D)",
                 "nav_date": "2024-01-01", "last_val": 11.5},
                {"proj_id": "p1", "fund_class_name": "K-GOLD-A(D)",
                 "nav_date": "2024-01-02", "last_val": 11.6},
                {"proj_id": "p1", "fund_class_name": "K-GOLD-A(D)",
                 "nav_date": "2024-01-03", "last_val": 11.7},
                {"proj_id": "p1", "fund_class_name": "K-GOLD-A(A)",
                 "nav_date": "2024-01-03", "last_val": 13.8},
            ],
        },
    )
    client = _make_client(session)

    items, incomplete = client.get_nav_range(
        "p1", None, dt.date(2024, 1, 1), dt.date(2024, 1, 3)
    )

    assert {i["fund_class_name"] for i in items} == {"K-GOLD-A(A)"}


def test_get_nav_range_raises_sec_api_error_on_http_error():
    session = MagicMock()
    error_resp = MagicMock(status_code=500)
    error_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
    session.get.return_value = error_resp
    client = _make_client(session)

    try:
        client.get_nav_range("p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 3))
        assert False, "expected SECAPIError"
    except thai_mf.SECAPIError:
        pass


def test_get_nav_range_raises_sec_api_error_on_connection_error():
    session = MagicMock()
    session.get.side_effect = requests.exceptions.Timeout("timed out")
    client = _make_client(session)

    try:
        client.get_nav_range("p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 3))
        assert False, "expected SECAPIError"
    except thai_mf.SECAPIError:
        pass


def test_get_nav_range_raises_sec_api_error_when_response_body_is_not_an_object():
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=200, json=lambda: "unexpected string body")
    client = _make_client(session)

    try:
        client.get_nav_range("p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 3))
        assert False, "expected SECAPIError"
    except thai_mf.SECAPIError:
        pass


@patch("thai_mf.time.sleep", return_value=None)
def test_get_nav_range_returns_incomplete_on_sustained_rate_limit(mock_sleep):
    session = MagicMock()
    session.get.return_value = MagicMock(status_code=429)
    client = _make_client(session)

    items, incomplete = client.get_nav_range("p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 3))

    assert incomplete is True
    assert items == []
    assert session.get.call_count == thai_mf.MAX_RETRIES
    assert mock_sleep.call_count == thai_mf.MAX_RETRIES - 1


def test_resolve_fund_id_fund_level_match_defers_class_choice(tmp_path, monkeypatch):
    # Typing the fund-level abbreviation is ambiguous across share classes,
    # so resolve returns preferred_class=None and lets the NAV layer pick
    # the primary class (the profiles class label is unreliable here).
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.list_funds = MagicMock(
        return_value=[
            {"proj_id": "p1", "proj_abbr_name": "K-GOLD", "fund_class_name": "K-GOLD-A(A)"},
            {"proj_id": "p1", "proj_abbr_name": "K-GOLD", "fund_class_name": "K-GOLD-A(D)"},
            {"proj_id": "p2", "proj_abbr_name": "SCBGOLD", "fund_class_name": "main"},
        ]
    )

    proj_id, preferred_class = thai_mf.resolve_fund_id("k-gold", client)

    assert proj_id == "p1"
    assert preferred_class is None
    client.list_funds.assert_called_once()


def test_resolve_fund_id_exact_class_match_pins_that_class(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.list_funds = MagicMock(
        return_value=[
            {"proj_id": "p1", "proj_abbr_name": "K-GOLD", "fund_class_name": "K-GOLD-A(A)"},
            {"proj_id": "p1", "proj_abbr_name": "K-GOLD", "fund_class_name": "K-GOLD-A(D)"},
            {"proj_id": "p1", "proj_abbr_name": "K-GOLD", "fund_class_name": "K-GOLD-C(A)"},
        ]
    )

    proj_id, preferred_class = thai_mf.resolve_fund_id("k-gold-a(d)", client)

    assert proj_id == "p1"
    assert preferred_class == "K-GOLD-A(D)"


def test_resolve_fund_id_returns_none_tuple_when_unmatched(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.list_funds = MagicMock(
        return_value=[{"proj_id": "p1", "proj_abbr_name": "SCBGOLD", "fund_class_name": "main"}]
    )

    proj_id, fund_class_name = thai_mf.resolve_fund_id("NOT-A-REAL-FUND", client)

    assert proj_id is None
    assert fund_class_name is None


def test_resolve_fund_id_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.list_funds = MagicMock(
        return_value=[{"proj_id": "p1", "proj_abbr_name": "SCBGOLD", "fund_class_name": "main"}]
    )

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
                {"proj_id": "p1", "fund_class_name": "main",
                 "nav_date": "2024-01-01", "last_val": 10.0},
                {"proj_id": "p1", "fund_class_name": "main",
                 "nav_date": "2024-01-02", "last_val": 10.5},
            ],
            False,
        )
    )

    series, incomplete, chosen_class = thai_mf.get_nav_history(
        client, "p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 3)
    )

    assert incomplete is False
    assert chosen_class == "main"
    assert list(series.values) == [10.0, 10.5]
    client.get_nav_range.assert_called_once_with(
        "p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 3)
    )
    assert (tmp_path / "nav_p1__main.csv").exists()
    assert (tmp_path / "nav_p1__main.meta.json").exists()


def test_get_nav_history_reuses_cache_for_already_fetched_range(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.get_nav_range = MagicMock(
        return_value=([{"proj_id": "p1", "fund_class_name": "main", "nav_date": "2024-01-01", "last_val": 10.0}], False)
    )

    thai_mf.get_nav_history(client, "p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    assert client.get_nav_range.call_count == 1

    thai_mf.get_nav_history(client, "p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    assert client.get_nav_range.call_count == 1  # no new call, served from cache


def test_get_nav_history_refetches_when_requested_range_extends_beyond_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.get_nav_range = MagicMock(
        return_value=([{"proj_id": "p1", "fund_class_name": "main", "nav_date": "2024-01-01", "last_val": 10.0}], False)
    )

    thai_mf.get_nav_history(client, "p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 1))
    assert client.get_nav_range.call_count == 1

    thai_mf.get_nav_history(client, "p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 5))
    assert client.get_nav_range.call_count == 2  # wider range not covered by cache


def test_get_nav_history_flags_incomplete_and_keeps_partial_results_on_rate_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(thai_mf, "CACHE_DIR", tmp_path)
    session = MagicMock()
    client = _make_client(session)
    client.get_nav_range = MagicMock(
        return_value=([{"proj_id": "p1", "fund_class_name": "main", "nav_date": "2024-01-01", "last_val": 10.0}], True)
    )

    series, incomplete, chosen_class = thai_mf.get_nav_history(
        client, "p1", "main", dt.date(2024, 1, 1), dt.date(2024, 1, 3)
    )

    assert incomplete is True
    assert list(series.values) == [10.0]
    # an incomplete fetch must not mark the range as fully cached
    assert not (tmp_path / "nav_p1__main.meta.json").exists()


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
