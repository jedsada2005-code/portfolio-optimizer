from unittest.mock import MagicMock, patch
import datetime as dt

import pandas as pd

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
