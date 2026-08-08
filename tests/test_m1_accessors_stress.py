"""
Empirical stress tests for M1 edge-case fixes in src/seroepi/accessors.py:
1. GenoAccessor.has_gene with regex special characters
2. EpiAccessor._get_spatiotemporal_arrays with null date values, NaT, mixed dates, and non-date columns.
"""

import datetime

import numpy as np
import polars as pl

from seroepi.dataset import SeroEpiDataset

# ==============================================================================
# 1. GenoAccessor.has_gene Stress Tests (Regex Special Characters & Edge Cases)
# ==============================================================================


def test_has_gene_regex_special_characters():
    """Verify has_gene treats regex special characters literally across multiple patterns."""
    df = pl.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(10)],
            "geno_genes": [
                "bla(KPC-2), TEM-1",  # 0: parenthetical
                "blaKPC-2",  # 1: no parens
                "gene[1]_v2",  # 2: square brackets
                "gene1_v2",  # 3: no brackets
                "a+b_variant",  # 4: plus sign
                "ab_variant",  # 5: no plus sign
                "c*d_locus",  # 6: asterisk
                "e?f_allele",  # 7: question mark
                "g.h_marker",  # 8: dot
                "^i$_exact",  # 9: caret and dollar
            ],
        }
    )

    # 1. Parentheses: 'bla(KPC-2)' should match row 0, but NOT row 1
    res_parens = df.geno.has_gene("geno_genes", "bla(KPC-2)")
    assert res_parens.to_list() == [True, False, False, False, False, False, False, False, False, False]

    # 2. Square brackets: 'gene[1]' should match row 2, but NOT row 3
    res_brackets = df.geno.has_gene("geno_genes", "gene[1]")
    assert res_brackets.to_list() == [False, False, True, False, False, False, False, False, False, False]

    # 3. Plus sign: 'a+b' should match row 4, but NOT row 5
    res_plus = df.geno.has_gene("geno_genes", "a+b")
    assert res_plus.to_list() == [False, False, False, False, True, False, False, False, False, False]

    # 4. Asterisk: 'c*d' should match row 6
    res_star = df.geno.has_gene("geno_genes", "c*d")
    assert res_star.to_list() == [False, False, False, False, False, False, True, False, False, False]

    # 5. Question mark: 'e?f' should match row 7
    res_qmark = df.geno.has_gene("geno_genes", "e?f")
    assert res_qmark.to_list() == [False, False, False, False, False, False, False, True, False, False]

    # 6. Dot: 'g.h' should match row 8 (literal dot, not wildcard)
    res_dot = df.geno.has_gene("geno_genes", "g.h")
    assert res_dot.to_list() == [False, False, False, False, False, False, False, False, True, False]

    # Verify regex wildcard '.' does not match 'gah' or 'g_h'
    df_dot_check = pl.DataFrame({"sample_id": ["S1", "S2"], "geno_genes": ["g.h", "gah"]})
    assert df_dot_check.geno.has_gene("geno_genes", "g.h").to_list() == [True, False]

    # 7. Caret and Dollar: '^i$' should match row 9 literally
    res_anchors = df.geno.has_gene("geno_genes", "^i$")
    assert res_anchors.to_list() == [False, False, False, False, False, False, False, False, False, True]


def test_has_gene_more_regex_symbols():
    """Verify has_gene with pipe, curly braces, and backslashes."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "geno_genes": [
                "(a|b)_gene",
                "a_gene",
                "gene_{2,3}",
                "gene_\\d+",
            ],
        }
    )

    # Pipe: '(a|b)' should match row 0, but NOT row 1
    res_pipe = df.geno.has_gene("geno_genes", "(a|b)")
    assert res_pipe.to_list() == [True, False, False, False]

    # Curly braces: '{2,3}' should match row 2
    res_braces = df.geno.has_gene("geno_genes", "{2,3}")
    assert res_braces.to_list() == [False, False, True, False]

    # Backslash and regex digits: '\\d+' should match row 3 literally
    res_slash = df.geno.has_gene("geno_genes", "\\d+")
    assert res_slash.to_list() == [False, False, False, True]


def test_has_gene_nulls_and_non_string_types():
    """Verify has_gene behavior on columns with nulls and non-string data types."""
    # Null values in Utf8 column
    df_nulls = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "geno_genes": ["bla(KPC)", None, "CTX-M"],
        }
    )
    res_nulls = df_nulls.geno.has_gene("geno_genes", "bla(KPC)")
    # Polars str.contains on null produces null or False depending on fill, check exact result:
    # In Polars, null.str.contains(...) produces null.
    assert res_nulls[0] is True
    assert res_nulls[1] is None
    assert res_nulls[2] is False

    # Boolean column
    df_bool = pl.DataFrame({"sample_id": ["S1", "S2"], "amr_bla": [True, False]})
    assert df_bool.geno.has_gene("amr_bla", "true").to_list() == [True, False]

    # Numeric column
    df_num = pl.DataFrame({"sample_id": ["S1", "S2"], "vir_count": [10, 20]})
    assert df_num.geno.has_gene("vir_count", "10").to_list() == [True, False]


# ==============================================================================
# 2. _get_spatiotemporal_arrays Stress Tests (Nulls, NaT, Mixed & Non-date)
# ==============================================================================


def test_spatiotemporal_arrays_pl_date_nulls():
    """Verify _get_spatiotemporal_arrays with pl.Date column containing nulls."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "latitude": [1.0, 2.0, 3.0],
            "longitude": [100.0, 101.0, 102.0],
            "temporal_Collection_Date": pl.Series(
                [datetime.date(2023, 1, 1), None, datetime.date(2023, 6, 15)],
                dtype=pl.Date,
            ),
        }
    )
    ds = SeroEpiDataset(data=df)
    coords, raw_dates, valid_mask = ds.epi._get_spatiotemporal_arrays("temporal_Collection_Date")

    assert len(coords) == 3
    assert len(raw_dates) == 3
    assert len(valid_mask) == 3

    # Check raw_dates float representation (days since epoch or nan)
    assert not np.isnan(raw_dates[0])
    assert np.isnan(raw_dates[1])
    assert not np.isnan(raw_dates[2])

    assert valid_mask.tolist() == [True, False, True]


def test_spatiotemporal_arrays_pl_datetime_nulls_and_nat():
    """Verify _get_spatiotemporal_arrays with pl.Datetime column containing nulls and NaT equivalents."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "latitude": [1.0, 2.0, 3.0, 4.0],
            "longitude": [100.0, 101.0, 102.0, 103.0],
            "temporal_Collection_Date": pl.Series(
                [datetime.datetime(2023, 1, 1, 12, 0), None, datetime.datetime(2023, 12, 31, 23, 59), None],
                dtype=pl.Datetime,
            ),
        }
    )
    ds = SeroEpiDataset(data=df)
    coords, raw_dates, valid_mask = ds.epi._get_spatiotemporal_arrays("temporal_Collection_Date")

    assert not np.isnan(raw_dates[0])
    assert np.isnan(raw_dates[1])
    assert not np.isnan(raw_dates[2])
    assert np.isnan(raw_dates[3])

    assert valid_mask.tolist() == [True, False, True, False]


def test_spatiotemporal_arrays_mixed_spatial_and_temporal_nulls():
    """Verify valid_mask accurately identifies rows where EITHER coords OR date are missing/NaN."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4", "S5"],
            "latitude": [1.0, None, 3.0, None, 5.0],
            "longitude": [100.0, 101.0, None, None, 104.0],
            "temporal_Collection_Date": pl.Series(
                [datetime.date(2023, 1, 1), datetime.date(2023, 1, 2), datetime.date(2023, 1, 3), None, None],
                dtype=pl.Date,
            ),
        }
    )
    ds = SeroEpiDataset(data=df)
    coords, raw_dates, valid_mask = ds.epi._get_spatiotemporal_arrays("temporal_Collection_Date")

    # S1: Lat=1.0, Lon=100.0, Date=2023-01-01 -> Valid (True)
    # S2: Lat=None, Lon=101.0, Date=2023-01-02 -> Invalid (False)
    # S3: Lat=3.0, Lon=None, Date=2023-01-03 -> Invalid (False)
    # S4: Lat=None, Lon=None, Date=None -> Invalid (False)
    # S5: Lat=5.0, Lon=104.0, Date=None -> Invalid (False)

    assert valid_mask.tolist() == [True, False, False, False, False]


def test_spatiotemporal_arrays_non_date_column():
    """Verify _get_spatiotemporal_arrays behavior on non-Date/non-Datetime columns and string date coercions."""
    # Case A: Parseable string date in dataset -> clean_and_coerce auto-converts Utf8 date to pl.Datetime
    df_str_date = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [100.0, 101.0],
            "temporal_Collection_Date": ["2023-01-01", "2023-01-02"],
        }
    )
    ds_str = SeroEpiDataset(data=df_str_date)
    assert ds_str.data.schema["temporal_Collection_Date"] == pl.Datetime
    coords_a, raw_dates_a, valid_mask_a = ds_str.epi._get_spatiotemporal_arrays("temporal_Collection_Date")
    assert not np.any(np.isnan(raw_dates_a))
    assert valid_mask_a.tolist() == [True, True]

    # Case B: Non-date column (e.g. Int64 or unparseable string) passed directly to dataframe epi accessor
    df_non_date = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [100.0, 101.0],
            "temporal_unparseable": ["not_a_date_1", "not_a_date_2"],
        }
    )
    coords_b, raw_dates_b, valid_mask_b = df_non_date.epi._get_spatiotemporal_arrays("temporal_unparseable")

    # Non pl.Date/pl.Datetime column returns all NaN raw_dates and all False valid_mask
    assert np.all(np.isnan(raw_dates_b))
    assert valid_mask_b.tolist() == [False, False]


def test_spatiotemporal_arrays_all_null_dates():
    """Verify _get_spatiotemporal_arrays when all date entries are null."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [100.0, 101.0],
            "temporal_Collection_Date": pl.Series([None, None], dtype=pl.Date),
        }
    )
    ds = SeroEpiDataset(data=df)
    coords, raw_dates, valid_mask = ds.epi._get_spatiotemporal_arrays("temporal_Collection_Date")

    assert np.all(np.isnan(raw_dates))
    assert valid_mask.tolist() == [False, False]


def test_transmission_clusters_integration_with_null_dates():
    """Verify transmission_clusters correctly outputs null cluster labels for rows with null dates."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "latitude": [1.0, 1.0001, 1.0002, 1.0003],
            "longitude": [100.0, 100.0001, 100.0002, 100.0003],
            "temporal_Collection_Date": pl.Series(
                [datetime.datetime(2023, 1, 1), None, datetime.datetime(2023, 1, 2), datetime.datetime(2023, 1, 2)],
                dtype=pl.Datetime,
            ),
            "geno_ST": ["ST1", "ST1", "ST1", "ST1"],
        }
    )
    ds = SeroEpiDataset(data=df)
    clusters = ds.epi.transmission_clusters(clone_col="geno_ST")

    assert isinstance(clusters, pl.Series)
    assert clusters.dtype == pl.Categorical
    cluster_list = clusters.to_list()
    # S2 had null date, so its cluster label must be None
    assert cluster_list[1] is None
    # S1, S3, S4 are valid and close, so they should be assigned cluster labels
    assert cluster_list[0] is not None
    assert cluster_list[2] is not None
    assert cluster_list[3] is not None
