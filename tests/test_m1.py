"""
Unit tests for Milestone M1 core data model, accessors, and I/O.
"""

import datetime

import numpy as np
import polars as pl
import pytest
from patito.exceptions import DataFrameValidationError

from seroepi.constants import TemporalResolution
from seroepi.dataset import SeroEpiDataset
from seroepi.domains import EpiAccessor
from seroepi.io import PathogenwatchKleborateParser


def test_cross_join_refactor():
    """Verify that multi-strata zero-padding works without .cross_join() AttributeError."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.23, 1.24],
            "longitude": [103.81, 103.82],
            "spatial_Continent": ["Asia", "Asia"],
            "spatial_Country": ["Singapore", "Malaysia"],
            "geno_ST": ["ST11", "ST258"],
            "amr_blaKPC": [True, False],
            "temporal_Collection_Date": [datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 2)],
        }
    )
    ds = SeroEpiDataset(data=df)

    prev = ds.epi.aggregate_prevalence(
        stratify_by=["spatial_Continent", "spatial_Country"], trait_col="amr_blaKPC", pad_zeros=True
    )
    assert len(prev) == 2

    div = ds.epi.aggregate_diversity(stratify_by=["spatial_Country", "geno_ST"], trait_col="amr_blaKPC", pad_zeros=True)
    assert len(div) == 8

    inc = ds.epi.aggregate_incidence(stratify_by=["spatial_Country", "geno_ST"], trait_col="amr_blaKPC", pad_zeros=True)
    assert len(inc) >= 2


def test_resolve_freq_duration_strings():
    """Verify that _resolve_freq handles Polars duration strings correctly."""
    assert EpiAccessor._resolve_freq("1w") == "1w"
    assert EpiAccessor._resolve_freq("1mo") == "1mo"
    assert EpiAccessor._resolve_freq("1d") == "1d"
    assert EpiAccessor._resolve_freq("week") == "1w"
    assert EpiAccessor._resolve_freq(TemporalResolution.WEEK) == "1w"

    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "temporal_Collection_Date": [datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 15)],
        }
    )
    ds = SeroEpiDataset(data=df)
    curve = ds.epi.epidemic_curve(freq="1w")
    assert len(curve) > 0


def test_transmission_network_parameter_types():
    """Verify that transmission_network passes sample_ids as Pandas Series with .values."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.23, 1.24],
            "longitude": [103.81, 103.82],
            "temporal_Collection_Date": [datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 2)],
            "geno_ST": ["ST258", "ST258"],
        }
    )
    ds = SeroEpiDataset(data=df)
    net = ds.epi.transmission_network(clone_col="geno_ST")
    assert len(net.index) == 2
    assert list(net.index) == ["S1", "S2"]


def test_geno_has_all_empty_traits():
    """Verify that GenoAccessor.has_all([]) returns True boolean Series for empty traits."""
    df = pl.DataFrame({"sample_id": ["S1", "S2"], "vir_geneA": [True, False]})
    res = df.geno.has_all([])
    assert isinstance(res, pl.Series)
    assert res.to_list() == [True, True]


def test_temporal_string_parsing_and_categorical_resolution():
    """Verify clean_and_coerce parses string dates and handles Patito validation with resolution columns."""
    records = [{"Genome Name": "Isolate_01", "Collection Date": "2023-05-15", "ST": "ST258"}]
    df = PathogenwatchKleborateParser.from_records(records)
    assert df.get_column("temporal_Collection_Date")[0] == datetime.datetime(2023, 5, 15, 0, 0)

    genotype_df = pl.DataFrame({"Genome Name": ["Isolate_1"], "Latitude": [1.23], "Longitude": [103.8], "ST": ["ST11"]})
    meta_df = pl.DataFrame({"sample_id": ["Isolate_1"], "isolation_date": ["2023-05-15"]})
    parsed = PathogenwatchKleborateParser.parse(
        genotype_df, meta_df=meta_df, meta_kwargs={"id_col": "sample_id", "date_col": "isolation_date"}
    )
    assert "temporal_res_isolation_date" in parsed.columns


def test_post_init_validation_error_propagation():
    """Verify that SeroEpiDataset.__post_init__ propagates Patito validation errors."""
    invalid_df = pl.DataFrame({"sample_id": ["S1", "S1"], "latitude": [999.0, 20.0], "longitude": [100.0, 110.0]})
    with pytest.raises(DataFrameValidationError):
        SeroEpiDataset(data=invalid_df)


def test_transmission_clusters():
    """Verify that transmission_clusters executes without float to categorical casting errors."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 1.001],
            "longitude": [100.0, 100.001],
            "temporal_Collection_Date": [datetime.datetime(2023, 1, 1), datetime.datetime(2023, 1, 5)],
            "geno_ST": ["ST1", "ST1"],
        }
    )
    ds = SeroEpiDataset(data=df)
    clusters = ds.epi.transmission_clusters(clone_col="geno_ST")
    assert isinstance(clusters, pl.Series)
    assert clusters.dtype == pl.Categorical
    assert len(clusters) == 2


def test_geno_has_gene():
    """Verify that GenoAccessor.has_gene works on string, boolean, and numeric columns without exception."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "amr_bla": [True, False],
            "vir_count": [10, 20],
            "geno_genes": ["blaKPC-2, TEM-1", "SHV-11"],
        }
    )
    res_bool = df.geno.has_gene("amr_bla", "true")
    assert res_bool.to_list() == [True, False]

    res_num = df.geno.has_gene("vir_count", "10")
    assert res_num.to_list() == [True, False]

    res_str = df.geno.has_gene("geno_genes", "blaKPC")
    assert res_str.to_list() == [True, False]


def test_geno_has_gene_regex_special_chars():
    """Verify that GenoAccessor.has_gene matches gene names with regex special characters literally."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "geno_genes": ["bla(KPC-2), TEM-1", "blaKPC-2", "CTX-M-15"],
        }
    )
    # 'bla(KPC-2)' contains special regex characters '(' and ')'
    res = df.geno.has_gene("geno_genes", "bla(KPC-2)")
    assert res.to_list() == [True, False, False]


def test_spatiotemporal_arrays_null_dates():
    """Verify that null date values produce np.nan in _get_spatiotemporal_arrays and nulls in transmission_clusters."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "latitude": [1.0, 1.001, 1.002],
            "longitude": [100.0, 100.001, 100.002],
            "temporal_Collection_Date": [datetime.datetime(2023, 1, 1), None, datetime.datetime(2023, 1, 3)],
            "geno_ST": ["ST1", "ST1", "ST1"],
        }
    )
    ds = SeroEpiDataset(data=df)
    coords, raw_dates, valid_mask = ds.epi._get_spatiotemporal_arrays("temporal_Collection_Date")
    assert np.isnan(raw_dates[1])
    assert not valid_mask[1]

    clusters = ds.epi.transmission_clusters(clone_col="geno_ST")
    assert clusters.to_list()[1] is None


def test_select_does_not_reinject_lat_lon():
    """Verify calling ds.select('sample_id') returns a dataset containing ONLY 'sample_id'."""
    df = pl.DataFrame({"sample_id": ["S1"], "latitude": [1.0], "longitude": [2.0], "geno_ST": ["ST1"]})
    ds = SeroEpiDataset(data=df)
    ds_sel = ds.select("sample_id")
    assert ds_sel.columns == ["sample_id"]


def test_with_columns_preserves_custom_columns():
    """Verify calling ds.with_columns(custom_calc=...) preserves non-prefixed custom columns."""
    df = pl.DataFrame({"sample_id": ["S1"], "latitude": [1.0], "longitude": [2.0], "geno_ST": ["ST1"]})
    ds = SeroEpiDataset(data=df)
    ds_with = ds.with_columns(custom_metric=pl.lit(42))
    assert "custom_metric" in ds_with.columns
    assert ds_with["custom_metric"][0] == 42


def test_metadata_copy_isolation():
    """Verify mutating derived dataset's metadata does not mutate parent dataset's metadata."""
    df = pl.DataFrame({"sample_id": ["S1"], "latitude": [1.0], "longitude": [2.0]})
    initial_meta = {"study": "alpha"}
    ds = SeroEpiDataset(data=df, metadata=initial_meta)
    ds2 = ds.filter(pl.col("sample_id") == "S1")
    ds2.metadata["study"] = "mutated"
    assert ds.metadata["study"] == "alpha"
    assert ds2.metadata["study"] == "mutated"


def test_accessors_pandas_free_import():
    """Verify seroepi.domains imports cleanly without pandas in module namespace."""
    import importlib

    import seroepi.domains

    importlib.reload(seroepi.domains)
    assert not hasattr(seroepi.domains, "pd")


def test_dataset_equals_dataframe_type():
    """Verify that ds1.data.equals(ds2.data) returns True without TypeError caused by dynamic subclasses."""
    df = pl.DataFrame({"sample_id": ["S1"], "latitude": [1.0], "longitude": [2.0]})
    ds1 = SeroEpiDataset(data=df)
    ds2 = SeroEpiDataset(data=df)
    assert isinstance(ds1.data, pl.DataFrame)
    assert isinstance(ds2.data, pl.DataFrame)
    assert ds1.data.equals(ds2.data) is True


def test_dataset_pickling():
    """Verify that SeroEpiDataset can be pickled and unpickled without PicklingError."""
    import pickle

    df = pl.DataFrame({"sample_id": ["S1"], "latitude": [1.0], "longitude": [2.0]})
    ds = SeroEpiDataset(data=df, name="PickleTest", metadata={"k": "v"})
    serialized = pickle.dumps(ds)
    restored = pickle.loads(serialized)
    assert restored.name == "PickleTest"
    assert restored.metadata == {"k": "v"}
    assert restored.data.equals(ds.data)


def test_patito_complex_dtypes_validation():
    """Verify DataFrames containing pl.List, pl.Null, pl.Struct, pl.Object columns validate cleanly."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1"],
            "genes": [["blaKPC", "TEM1"]],
            "null_col": [None],
            "struct_col": [{"a": 1}],
            "obj_col": [object()],
        },
        schema_overrides={"obj_col": pl.Object},
    )
    ds = SeroEpiDataset(data=df)
    assert "genes" in ds.columns
    assert "null_col" in ds.columns
    assert "struct_col" in ds.columns
    assert "obj_col" in ds.columns
    assert ds["genes"][0].to_list() == ["blaKPC", "TEM1"]
