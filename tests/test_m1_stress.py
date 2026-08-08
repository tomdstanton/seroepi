"""
Empirical stress test suite for Milestone M1:
- SeroEpiDataset immutability & slotted frozen dataclass compliance
- Patito SampleModel validation & domain coercions
- Dictionary serialization & deserialization roundtrips
- Edge cases, column projection behavior, and metadata reference leaks
"""

import datetime
from dataclasses import FrozenInstanceError, is_dataclass

import polars as pl
import pytest
from patito.exceptions import DataFrameValidationError

from seroepi.dataset import SeroEpiDataset
from seroepi.io import (
    PathogenwatchKleborateParser,
    SampleModel,
    _flatten_dict,
)

# --- 1. SeroEpiDataset Immutability & Dataclass Compliance ---


def test_dataclass_slots_and_frozen_attributes():
    """Verify SeroEpiDataset is slotted, frozen, and enforces immutability on attributes."""
    assert is_dataclass(SeroEpiDataset), "SeroEpiDataset must be a dataclass"
    assert hasattr(SeroEpiDataset, "__slots__"), "SeroEpiDataset must have __slots__ defined"
    assert set(SeroEpiDataset.__slots__) == {"data", "name", "metadata"}, (
        f"Unexpected __slots__: {SeroEpiDataset.__slots__}"
    )

    df = pl.DataFrame({"sample_id": ["S1", "S2"], "latitude": [1.0, 2.0], "longitude": [10.0, 20.0]})
    ds = SeroEpiDataset(data=df, name="TestDataset", metadata={"env": "test"})

    # Check __dict__ absence
    assert not hasattr(ds, "__dict__"), "Slotted dataclasses must not have __dict__"

    # Attempt attribute mutation -> raises FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        ds.data = pl.DataFrame({"sample_id": ["S3"]})

    with pytest.raises(FrozenInstanceError):
        ds.name = "NewName"

    with pytest.raises(FrozenInstanceError):
        ds.metadata = {}

    # Attempt adding new attribute -> raises TypeError or AttributeError under slotted frozen dataclasses
    with pytest.raises((AttributeError, TypeError)):
        ds.new_attribute = 123

    # Attempt deleting attribute -> raises FrozenInstanceError or AttributeError
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        del ds.name


def test_metadata_reference_leak_immutability():
    """Verify metadata dictionary is copy-isolated on init and transformation."""
    df = pl.DataFrame({"sample_id": ["S1"], "latitude": [1.0], "longitude": [2.0]})
    initial_meta = {"study": "alpha"}
    ds = SeroEpiDataset(data=df, metadata=initial_meta)

    # 1. Mutate original dictionary passed by caller
    initial_meta["study"] = "mutated_external"
    assert ds.metadata["study"] == "alpha", "Metadata dictionary reference must be copy-isolated"

    # 2. Mutate metadata in derived dataset
    ds2 = ds.filter(pl.col("sample_id") == "S1")
    ds2.metadata["study"] = "mutated_in_ds2"
    assert ds.metadata["study"] == "alpha", "Metadata reference must not leak across transformations"
    assert ds2.metadata["study"] == "mutated_in_ds2"


def test_transformation_methods_and_column_projection_bugs():
    """Verify transformation methods return new instances and project/preserve columns correctly."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "latitude": [10.0, 20.0, 30.0],
            "longitude": [50.0, 60.0, 70.0],
            "geno_ST": ["ST1", "ST2", "ST1"],
        }
    )
    ds_orig = SeroEpiDataset(data=df, name="OriginalDS", metadata={"version": 1})

    # 1. filter returns new instance and leaves original untouched
    ds_filtered = ds_orig.filter(pl.col("geno_ST") == "ST1")
    assert isinstance(ds_filtered, SeroEpiDataset)
    assert ds_filtered is not ds_orig
    assert len(ds_orig) == 3
    assert len(ds_filtered) == 2
    assert ds_orig.data.shape == (3, 4)

    # 2. select behavior check: latitude & longitude not re-injected
    ds_selected = ds_orig.select("sample_id", "geno_ST")
    assert isinstance(ds_selected, SeroEpiDataset)
    assert ds_selected.columns == ["sample_id", "geno_ST"]

    # 3. with_columns behavior check: custom non-prefixed column preserved
    ds_with_custom = ds_orig.with_columns(unprefixed_calc=pl.lit(100))
    assert "unprefixed_calc" in ds_with_custom.columns
    assert ds_with_custom["unprefixed_calc"][0] == 100

    # 4. head / tail
    ds_head = ds_orig.head(1)
    ds_tail = ds_orig.tail(1)
    assert len(ds_head) == 1
    assert len(ds_tail) == 1
    assert len(ds_orig) == 3

    # 5. validate
    ds_val = ds_orig.validate()
    assert isinstance(ds_val, SeroEpiDataset)
    assert ds_val is not ds_orig

    # 6. pipe
    ds_piped = ds_orig.pipe(lambda df_: df_.filter(pl.col("sample_id") == "S1"))
    assert isinstance(ds_piped, SeroEpiDataset)
    assert len(ds_piped) == 1


def test_post_init_coercions_and_type_checks():
    """Verify SeroEpiDataset post_init checks invalid types and handles metadata defaults."""
    # Non-DataFrame data
    with pytest.raises(TypeError, match="expected pl.DataFrame"):
        SeroEpiDataset(data="not_a_dataframe")  # type: ignore

    with pytest.raises(TypeError, match="expected pl.DataFrame"):
        SeroEpiDataset(data=[1, 2, 3])  # type: ignore

    # Non-string name coercion
    df = pl.DataFrame({"sample_id": ["S1"]})
    ds = SeroEpiDataset(data=df, name=12345)  # type: ignore
    assert ds.name == "12345"

    # Default metadata
    ds_default_meta = SeroEpiDataset(data=df)
    assert ds_default_meta.metadata == {}
    assert ds_default_meta.name == "SeroEpi Dataset"


# --- 2. Patito SampleModel Validation & Coercion ---


def test_unified_isolate_model_preserves_all_columns():
    """Verify clean_and_coerce preserves non-prefixed custom columns."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1"],
            "latitude": [1.23],
            "longitude": [103.8],
            "qc_score": [99],
            "geno_ST": ["ST11"],
            "amr_gene": [True],
            "vir_score": [5],
            "pheno_type": ["K1"],
            "temporal_date": [datetime.datetime(2023, 1, 1)],
            "spatial_country": ["SG"],
            "meta_notes": ["some note"],
            "unwanted_random_column": ["preserved"],
            "another_bad_col": [1234],
        }
    )

    cleaned = SampleModel.clean_and_coerce(df)
    assert "unwanted_random_column" in cleaned.columns
    assert "another_bad_col" in cleaned.columns
    assert "qc_score" in cleaned.columns
    assert "geno_ST" in cleaned.columns
    assert "meta_notes" in cleaned.columns
    assert "sample_id" in cleaned.columns


def test_unified_isolate_model_coordinate_bounds():
    """Verify Patito validates latitude [-90, 90] and longitude [-180, 180]."""
    # Valid boundary values
    valid_df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [-90.0, 90.0],
            "longitude": [-180.0, 180.0],
        }
    )
    validated = SampleModel.validate(valid_df)
    assert len(validated) == 2

    # Invalid latitude > 90
    invalid_lat_df = pl.DataFrame(
        {
            "sample_id": ["S1"],
            "latitude": [90.001],
            "longitude": [0.0],
        }
    )
    with pytest.raises(DataFrameValidationError):
        SeroEpiDataset(data=invalid_lat_df)

    # Invalid latitude < -90
    invalid_lat_neg = pl.DataFrame(
        {
            "sample_id": ["S1"],
            "latitude": [-90.1],
            "longitude": [0.0],
        }
    )
    with pytest.raises(DataFrameValidationError):
        SeroEpiDataset(data=invalid_lat_neg)

    # Invalid longitude > 180
    invalid_lon_df = pl.DataFrame(
        {
            "sample_id": ["S1"],
            "latitude": [0.0],
            "longitude": [180.01],
        }
    )
    with pytest.raises(DataFrameValidationError):
        SeroEpiDataset(data=invalid_lon_df)

    # Invalid longitude < -180
    invalid_lon_neg = pl.DataFrame(
        {
            "sample_id": ["S1"],
            "latitude": [0.0],
            "longitude": [-180.5],
        }
    )
    with pytest.raises(DataFrameValidationError):
        SeroEpiDataset(data=invalid_lon_neg)


def test_unified_isolate_model_sample_id_uniqueness():
    """Verify Patito enforces sample_id uniqueness."""
    dup_df = pl.DataFrame(
        {
            "sample_id": ["ISO_001", "ISO_001"],
            "latitude": [10.0, 10.0],
            "longitude": [20.0, 20.0],
        }
    )
    with pytest.raises(DataFrameValidationError):
        SampleModel.validate(dup_df)


def test_resolution_enum_validation():
    """Verify invalid temporal_res_* or spatial_res_* values raise ValueError in clean_and_coerce."""
    invalid_temporal_res_df = pl.DataFrame(
        {
            "sample_id": ["S1"],
            "temporal_res_collection": ["invalid_time_unit"],
        }
    )
    with pytest.raises(ValueError, match="Invalid TemporalResolution values"):
        SampleModel.clean_and_coerce(invalid_temporal_res_df)

    invalid_spatial_res_df = pl.DataFrame(
        {
            "sample_id": ["S1"],
            "spatial_res_location": ["mars_orbit"],
        }
    )
    with pytest.raises(ValueError, match="Invalid SpatialResolution values"):
        SampleModel.clean_and_coerce(invalid_spatial_res_df)


# --- 3. Dictionary Serialization & Serialization Roundtrips ---


def test_dataset_dictionary_serialization():
    """Verify to_dict() and to_dicts() structure and behavior."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [10.0, 20.0],
            "geno_ST": ["ST11", "ST258"],
        }
    )
    ds = SeroEpiDataset(data=df, name="SerialTest", metadata={"study": "A"})

    # to_dicts
    records = ds.to_dicts()
    assert isinstance(records, list)
    assert len(records) == 2
    assert records[0] == {"sample_id": "S1", "latitude": 1.0, "longitude": 10.0, "geno_ST": "ST11"}

    # to_dict
    ser = ds.to_dict()
    assert isinstance(ser, dict)
    assert set(ser.keys()) == {"dataset_name", "records", "metadata"}
    assert ser["dataset_name"] == "SerialTest"
    assert ser["records"] == records
    assert ser["metadata"] == {"study": "A"}

    # to_polars
    polars_df = ds.to_polars()
    assert isinstance(polars_df, pl.DataFrame)
    assert polars_df.equals(ds.data)


def test_genotype_parser_flatten_and_from_records():
    """Verify dictionary flattening and parser from_records roundtrip."""
    nested_dict = {"id": "S1", "nested": {"sub1": 10, "sub2": {"deeper": "val"}}}
    flat = _flatten_dict(nested_dict)
    assert flat == {"id": "S1", "nested/sub1": 10, "nested/sub2/deeper": "val"}

    records = [
        {"Genome Name": "Isolate_A", "ST": "ST11", "Latitude": 1.23, "Longitude": 103.8},
        {"Genome Name": "Isolate_B", "ST": "ST258", "Latitude": 2.34, "Longitude": 104.9},
    ]
    parsed_df = PathogenwatchKleborateParser.from_records(records)
    ds = SeroEpiDataset(data=parsed_df, name="FromRecordsDS")

    assert len(ds) == 2
    assert "sample_id" in ds.columns
    assert "geno_ST" in ds.columns
    serialized = ds.to_dict()
    assert serialized["dataset_name"] == "FromRecordsDS"
    assert len(serialized["records"]) == 2


# --- 4. Edge Cases & Stress Harness ---


def test_empty_dataset_handling():
    """Verify SeroEpiDataset handles 0-row DataFrames correctly."""
    empty_df = pl.DataFrame(
        {
            "sample_id": pl.Series([], dtype=pl.Utf8),
            "latitude": pl.Series([], dtype=pl.Float64),
            "longitude": pl.Series([], dtype=pl.Float64),
        }
    )
    ds = SeroEpiDataset(data=empty_df, name="EmptyDS")

    assert len(ds) == 0
    assert ds.shape == (0, 3)
    assert ds.columns == ["sample_id", "latitude", "longitude"]
    assert ds.to_dicts() == []
    assert ds.to_dict()["records"] == []

    filtered = ds.filter(pl.col("sample_id") == "nonexistent")
    assert len(filtered) == 0


def test_large_dataset_stress():
    """Stress test dataset operations with 10,000 rows."""
    n_rows = 10_000
    df = pl.DataFrame(
        {
            "sample_id": [f"ISO_{i:05d}" for i in range(n_rows)],
            "latitude": [1.0 + (i % 80) * 0.1 for i in range(n_rows)],
            "longitude": [100.0 + (i % 50) * 0.1 for i in range(n_rows)],
            "geno_ST": [f"ST{i % 100}" for i in range(n_rows)],
            "amr_gene": [i % 2 == 0 for i in range(n_rows)],
        }
    )

    ds = SeroEpiDataset(data=df, name="LargeDataset")
    assert len(ds) == n_rows

    filtered = ds.filter(pl.col("geno_ST") == "ST5")
    assert len(filtered) == 100

    serialized = ds.to_dict()
    assert len(serialized["records"]) == n_rows


def test_dataset_accessor_forwarding():
    """Verify accessor properties forwarding to Polars DataFrame accessors."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [10.0, 20.0],
            "geno_ST": ["ST11", "ST258"],
            "amr_gene": [True, False],
            "qc_warning": ["pass", "pass"],
        }
    )
    ds = SeroEpiDataset(data=df)

    assert ds.geo is not None
    assert ds.epi is not None
    assert ds.geno is not None
    assert ds.qc is not None

    # Verify len, getitem, shape, columns, schema delegation
    assert len(ds) == 2
    assert ds["sample_id"].to_list() == ["S1", "S2"]
    assert ds.shape == (2, 6)
    assert set(ds.columns) == {"sample_id", "latitude", "longitude", "geno_ST", "amr_gene", "qc_warning"}
    assert isinstance(ds.schema, pl.Schema)
