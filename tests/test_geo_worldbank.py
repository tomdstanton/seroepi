"""Unit tests for WorldBankClient, resolve_iso3, and persistent disk caching."""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from seroepi import WorldBankClient
from seroepi.client import CUSTOM_ALIASES, resolve_iso3


def test_resolve_iso3_standard_and_aliases():
    client = WorldBankClient()

    # Standard country names
    assert resolve_iso3("United States") == "USA"
    assert resolve_iso3("Australia") == "AUS"
    assert resolve_iso3("France") == "FRA"
    assert resolve_iso3("Germany") == "DEU"

    # Static method delegation check
    assert client.resolve_iso3("United States") == "USA"
    assert client.resolve_iso3("Australia") == "AUS"

    # ISO codes
    assert resolve_iso3("USA") == "USA"
    assert resolve_iso3("GBR") == "GBR"
    assert resolve_iso3("US") == "USA"
    assert resolve_iso3("DE") == "DEU"

    # Custom aliases & edge cases
    assert resolve_iso3("UK") == "GBR"
    assert resolve_iso3("Great Britain") == "GBR"
    assert resolve_iso3("England") == "GBR"
    assert resolve_iso3("Russia") == "RUS"
    assert resolve_iso3("Ivory Coast") == "CIV"
    assert resolve_iso3("Cote d'Ivoire") == "CIV"
    assert resolve_iso3("Côte d'Ivoire") == "CIV"
    assert resolve_iso3("Turkey") == "TUR"
    assert resolve_iso3("Türkiye") == "TUR"
    assert resolve_iso3("Kosovo") == "XKX"
    assert resolve_iso3("XKX") == "XKX"
    assert resolve_iso3("xkx") == "XKX"
    assert resolve_iso3("Eswatini") == "SWZ"
    assert resolve_iso3("South Korea") == "KOR"

    # Invalid inputs
    assert resolve_iso3("NonExistentCountryName123") is None
    assert resolve_iso3("") is None
    assert resolve_iso3(None) is None


def test_worldbank_client_disk_caching(tmp_path):
    cache_dir = tmp_path / "cache"
    client = WorldBankClient(cache_dir=cache_dir)

    mock_payload = [
        {"page": 1, "pages": 1, "per_page": 1000, "total": 2},
        [
            {
                "indicator": {"id": "SP.DYN.LE00.IN", "value": "Life expectancy"},
                "country": {"id": "US", "value": "United States"},
                "countryiso3code": "USA",
                "date": "2022",
                "value": 77.43,
            },
            {
                "indicator": {"id": "SP.DYN.LE00.IN", "value": "Life expectancy"},
                "country": {"id": "GB", "value": "United Kingdom"},
                "countryiso3code": "GBR",
                "date": "2022",
                "value": 81.0,
            },
        ],
    ]

    # Initial call (mocked network request)
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_payload
        mock_response.raise_for_status = lambda: None
        mock_get.return_value = mock_response

        data1 = client.fetch_indicator("SP.DYN.LE00.IN", ["USA", "GBR"])
        assert "USA" in data1
        assert data1["USA"][2022] == pytest.approx(77.43)
        assert mock_get.call_count == 1

    # Ensure cache directory was created and populated
    assert cache_dir.exists()
    cache_files = list(cache_dir.rglob("*"))
    assert len(cache_files) > 0

    # Second call with network mocked to fail: must succeed via joblib disk cache
    with patch("requests.get", side_effect=RuntimeError("Network call should not occur!")):
        data2 = client.fetch_indicator("SP.DYN.LE00.IN", ["USA", "GBR"])
        assert data2 == data1


def test_worldbank_fetch_indicator_mocked(tmp_path):
    client = WorldBankClient(cache_dir=tmp_path / "cache_mocked")
    mock_payload = [
        {"page": 1, "pages": 1, "per_page": 1000, "total": 2},
        [
            {
                "indicator": {"id": "SP.DYN.LE00.IN", "value": "Life expectancy"},
                "country": {"id": "US", "value": "United States"},
                "countryiso3code": "USA",
                "date": "2022",
                "value": 77.43,
            },
            {
                "indicator": {"id": "SP.DYN.LE00.IN", "value": "Life expectancy"},
                "country": {"id": "GB", "value": "United Kingdom"},
                "countryiso3code": "GBR",
                "date": "2022",
                "value": 81.0,
            },
        ],
    ]

    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_payload
        mock_response.raise_for_status = lambda: None
        mock_get.return_value = mock_response

        res = client.fetch_indicator("SP.DYN.LE00.IN", ["USA", "UK"])
        assert "USA" in res
        assert res["USA"][2022] == pytest.approx(77.43)
        assert "GBR" in res
        assert res["GBR"][2022] == pytest.approx(81.0)


def test_worldbank_fetch_indicator_invalid_inputs():
    client = WorldBankClient()

    # Empty ISO list
    assert client.fetch_indicator("SP.DYN.LE00.IN", []) == {}

    # Unresolvable countries
    assert client.fetch_indicator("SP.DYN.LE00.IN", ["NonExistent123"]) == {}


def test_worldbank_pagination(tmp_path):
    cache_dir = tmp_path / "cache_page"
    client = WorldBankClient(cache_dir=cache_dir)

    page1_payload = [
        {"page": 1, "pages": 2, "per_page": 1000, "total": 2},
        [
            {
                "indicator": {"id": "SP.DYN.LE00.IN"},
                "countryiso3code": "USA",
                "date": "2021",
                "value": 76.33,
            }
        ],
    ]
    page2_payload = [
        {"page": 2, "pages": 2, "per_page": 1000, "total": 2},
        [
            {
                "indicator": {"id": "SP.DYN.LE00.IN"},
                "countryiso3code": "USA",
                "date": "2022",
                "value": 77.43,
            }
        ],
    ]

    with patch("requests.get") as mock_get:
        resp1 = MagicMock()
        resp1.json.return_value = page1_payload
        resp1.raise_for_status = lambda: None

        resp2 = MagicMock()
        resp2.json.return_value = page2_payload
        resp2.raise_for_status = lambda: None

        mock_get.side_effect = [resp1, resp2]

        res = client.fetch_indicator("SP.DYN.LE00.IN", ["USA"])
        assert "USA" in res
        assert res["USA"][2021] == pytest.approx(76.33)
        assert res["USA"][2022] == pytest.approx(77.43)
        assert mock_get.call_count == 2


def test_worldbank_malformed_country_records(tmp_path):
    """Test handling of records where 'country' field is a string, int, or None."""
    client = WorldBankClient(cache_dir=tmp_path / "cache_malformed")
    payload = [
        {"page": 1, "pages": 1, "per_page": 1000, "total": 4},
        [
            # Valid record with country dict
            {
                "country": {"id": "US", "value": "United States"},
                "countryiso3code": "USA",
                "date": "2022",
                "value": 77.5,
            },
            # country field is a string instead of dict
            {
                "country": "United Kingdom",
                "countryiso3code": None,
                "date": "2021",
                "value": 81.0,
            },
            # country field is an int instead of dict
            {
                "country": 12345,
                "countryiso3code": None,
                "date": "2021",
                "value": 50.0,
            },
            # country field is None
            {
                "country": None,
                "countryiso3code": None,
                "date": "2021",
                "value": 60.0,
            },
            # Valid record with country dict fallback (countryiso3code missing)
            {
                "country": {"id": "GBR"},
                "date": "2020",
                "value": 80.5,
            },
        ],
    ]

    with patch("requests.get") as mock_get:
        resp = MagicMock()
        resp.json.return_value = payload
        resp.raise_for_status = lambda: None
        mock_get.return_value = resp

        res = client.fetch_indicator("SP.DYN.LE00.IN", ["USA", "GBR"])
        assert "USA" in res
        assert res["USA"][2022] == pytest.approx(77.5)
        assert "GBR" in res
        assert res["GBR"][2020] == pytest.approx(80.5)


def test_worldbank_readonly_cache_fallback(tmp_path):
    """Test graceful fallback to location=None when cache_dir is read-only (chmod 0o444)."""
    import os
    import stat

    ro_dir = tmp_path / "readonly_cache"
    ro_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(ro_dir, stat.S_IRUSR | stat.S_IXUSR)  # Read & execute only (0o500/0o444-like)

    try:
        client_ro = WorldBankClient(cache_dir=ro_dir)
        payload = [
            {"page": 1, "pages": 1, "per_page": 1000, "total": 1},
            [{"countryiso3code": "USA", "date": "2022", "value": 77.5}],
        ]

        with patch("requests.get") as mock_get:
            resp = MagicMock()
            resp.json.return_value = payload
            resp.raise_for_status = lambda: None
            mock_get.return_value = resp

            res = client_ro.fetch_indicator("SP.DYN.LE00.IN", ["USA"])
            assert "USA" in res
            assert res["USA"][2022] == pytest.approx(77.5)
    finally:
        os.chmod(ro_dir, stat.S_IRWXU)


def test_worldbank_pagination_empty_records_early_exit(tmp_path):
    """Test that pagination loop exits early when page_records is empty despite total pages > 1."""
    client = WorldBankClient(cache_dir=tmp_path / "cache_empty_records")
    payload_page1 = [
        {"page": 1, "pages": 10, "per_page": 1000, "total": 0},
        [],
    ]
    with patch("requests.get") as mock_get:
        resp = MagicMock()
        resp.json.return_value = payload_page1
        resp.raise_for_status = lambda: None
        mock_get.return_value = resp

        res = client.fetch_indicator("SP.DYN.LE00.IN", ["USA"])
        assert res == {}
        assert mock_get.call_count == 1


def test_geomixin_with_metadata_without_year(tmp_path):
    """Test with_metadata without year_col appends Float64 column with latest year data and preserves immutability."""
    import polars as pl
    from seroepi import SeroEpiDataset

    df = pl.DataFrame({
        "country": ["USA", "United Kingdom", "France"],
        "sample_id": ["S1", "S2", "S3"],
    })
    dataset = SeroEpiDataset(data=df, name="TestDataset")

    mock_payload = {
        "USA": {2020: 77.0, 2022: 77.5},
        "GBR": {2020: 80.5, 2022: 81.0},
        "FRA": {2020: 82.0, 2022: 82.5},
    }

    with patch.object(WorldBankClient, "fetch_indicator", return_value=mock_payload):
        res = dataset.geo.with_metadata("SP.DYN.LE00.IN", country_col="country")

        # Dataset immutability check
        assert "SP.DYN.LE00.IN" not in dataset.columns
        assert isinstance(res, SeroEpiDataset)
        assert "SP.DYN.LE00.IN" in res.columns
        assert res.schema["SP.DYN.LE00.IN"] == pl.Float64

        # Latest available year values check
        res_df = res.to_polars()
        usa_val = res_df.filter(pl.col("country") == "USA")["SP.DYN.LE00.IN"][0]
        gbr_val = res_df.filter(pl.col("country") == "United Kingdom")["SP.DYN.LE00.IN"][0]
        fra_val = res_df.filter(pl.col("country") == "France")["SP.DYN.LE00.IN"][0]

        assert usa_val == pytest.approx(77.5)
        assert gbr_val == pytest.approx(81.0)
        assert fra_val == pytest.approx(82.5)


def test_geomixin_with_metadata_with_year_and_fallback(tmp_path):
    """Test with_metadata with year_col handles exact year matching and temporal fallback logic."""
    import polars as pl
    from seroepi import SeroEpiDataset

    df = pl.DataFrame({
        "country": ["USA", "USA", "USA", "GBR", "USA"],
        "year": [2020, 2021, 1950, 2020, None],
    })
    dataset = SeroEpiDataset(data=df)

    mock_payload = {
        "USA": {2020: 77.0, 2022: 77.5},
        "GBR": {2020: 81.0},
    }

    with patch.object(WorldBankClient, "fetch_indicator", return_value=mock_payload):
        res = dataset.geo.with_metadata("SP.DYN.LE00.IN", country_col="country", year_col="year")

        assert isinstance(res, SeroEpiDataset)
        assert "SP.DYN.LE00.IN" in res.columns
        assert res.schema["SP.DYN.LE00.IN"] == pl.Float64

        vals = res.to_polars()["SP.DYN.LE00.IN"].to_list()
        # Row 0: USA 2020 -> exact match -> 77.0
        assert vals[0] == pytest.approx(77.0)
        # Row 1: USA 2021 -> fallback to latest year <= 2021 (2020) -> 77.0
        assert vals[1] == pytest.approx(77.0)
        # Row 2: USA 1950 -> no year <= 1950, fallback to max available year (2022) -> 77.5
        assert vals[2] == pytest.approx(77.5)
        # Row 3: GBR 2020 -> exact match -> 81.0
        assert vals[3] == pytest.approx(81.0)
        # Row 4: USA None -> null year fallback to max available year (2022) -> 77.5
        assert vals[4] == pytest.approx(77.5)


def test_geomixin_with_metadata_edge_cases(tmp_path):
    """Test edge cases: unknown countries, null entries, missing column error, column overwrite, re-exports."""
    import polars as pl
    from seroepi import GeoMixin, SeroEpiDataset, WorldBankClient

    # Re-export check
    assert WorldBankClient is not None
    assert GeoMixin is not None

    df = pl.DataFrame({
        "country": ["USA", "Atlantis", None],
        "year": [2020, 2020, 2020],
    })
    dataset = SeroEpiDataset(data=df)

    mock_payload = {
        "USA": {2020: 77.0},
    }

    with patch.object(WorldBankClient, "fetch_indicator", return_value=mock_payload):
        # Edge case: Unknown country and null country
        res = dataset.geo.with_metadata("SP.DYN.LE00.IN", country_col="country", year_col="year")
        vals = res.to_polars()["SP.DYN.LE00.IN"].to_list()
        assert vals[0] == pytest.approx(77.0)
        assert vals[1] is None
        assert vals[2] is None

        # Edge case: Column overwrite
        updated_mock_payload = {
            "USA": {2020: 78.0},
        }
        with patch.object(WorldBankClient, "fetch_indicator", return_value=updated_mock_payload):
            overwrite_res = res.geo.with_metadata("SP.DYN.LE00.IN", country_col="country", year_col="year")
            new_vals = overwrite_res.to_polars()["SP.DYN.LE00.IN"].to_list()
            assert overwrite_res.columns.count("SP.DYN.LE00.IN") == 1
            assert new_vals[0] == pytest.approx(78.0)

    # Edge case: Missing country column error
    with pytest.raises(ValueError, match="Country column 'invalid_country' not found"):
        dataset.geo.with_metadata("SP.DYN.LE00.IN", country_col="invalid_country")

    # Edge case: Missing year column error
    with pytest.raises(ValueError, match="Year column 'invalid_year' not found"):
        dataset.geo.with_metadata("SP.DYN.LE00.IN", country_col="country", year_col="invalid_year")

    # Edge case: StrEnum column resolution
    from enum import StrEnum

    class Cols(StrEnum):
        COUNTRY = "country"
        YEAR = "year"

    with patch.object(WorldBankClient, "fetch_indicator", return_value=mock_payload):
        res_strenum = dataset.geo.with_metadata("SP.DYN.LE00.IN", country_col=Cols.COUNTRY, year_col=Cols.YEAR)
        assert "SP.DYN.LE00.IN" in res_strenum.columns




