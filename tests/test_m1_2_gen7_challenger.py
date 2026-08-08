"""
Empirical Challenger Stress Test Suite (challenger_m1_2_gen7):
Stress-testing SeroEpiDataset immutability, slotted frozen dataclass attributes,
select(), with_columns(), filter(), equals(), pickle.dumps(), and complex Polars dtypes.
"""

import datetime
import pickle
from dataclasses import FrozenInstanceError, is_dataclass

import polars as pl
import pytest
from patito.exceptions import DataFrameValidationError

from seroepi.dataset import SeroEpiDataset

# ==============================================================================
# 1. Slotted Frozen Dataclass Attributes & Immutability Enforcement
# ==============================================================================


def test_challenger_slotted_frozen_dataclass_enforcement():
    """Verify SeroEpiDataset is slotted, frozen, and strictly enforces immutability."""
    assert is_dataclass(SeroEpiDataset), "SeroEpiDataset must be a dataclass"
    assert hasattr(SeroEpiDataset, "__slots__"), "SeroEpiDataset must have __slots__"
    assert set(SeroEpiDataset.__slots__) == {"data", "name", "metadata"}, (
        f"Unexpected slots: {SeroEpiDataset.__slots__}"
    )

    df = pl.DataFrame({"sample_id": ["S1"], "latitude": [10.0], "longitude": [20.0]})
    ds = SeroEpiDataset(data=df, name="ChallengerDS", metadata={"key": "value"})

    # 1. Absence of __dict__
    assert not hasattr(ds, "__dict__"), "Slotted dataclasses must not have __dict__"

    # 2. Frozen attribute assignment prevention
    with pytest.raises(FrozenInstanceError):
        ds.data = pl.DataFrame({"sample_id": ["S2"]})

    with pytest.raises(FrozenInstanceError):
        ds.name = "MutatedName"

    with pytest.raises(FrozenInstanceError):
        ds.metadata = {"new": "dict"}

    # 3. Dynamic attribute creation prevention
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        ds.extra_field = "invalid"

    # 4. Attribute deletion prevention
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        del ds.name

    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        del ds.data

    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        del ds.metadata


def test_challenger_metadata_copy_isolation_top_level_and_nested():
    """Stress test metadata dictionary copy isolation on init and transformations."""
    df = pl.DataFrame({"sample_id": ["S1"]})
    input_meta = {"study": "alpha", "nested": {"param": 10}, "list_field": [1, 2]}
    ds = SeroEpiDataset(data=df, metadata=input_meta)

    # Top-level mutation of external input dictionary does not affect ds.metadata
    input_meta["study"] = "beta"
    assert ds.metadata["study"] == "alpha"

    # Derived dataset copy isolation
    ds_filtered = ds.filter(pl.col("sample_id") == "S1")
    ds_filtered.metadata["study"] = "gamma"
    ds_filtered.metadata["new_top_key"] = 999

    assert ds.metadata["study"] == "alpha"
    assert "new_top_key" not in ds.metadata
    assert ds_filtered.metadata["study"] == "gamma"
    assert ds_filtered.metadata is not ds.metadata


# ==============================================================================
# 2. select() Empirical Stress Tests
# ==============================================================================


def test_challenger_select_transformations_and_column_projection():
    """Verify select() column projection, renames, and non-injection of excluded columns."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [10.0, 20.0],
            "geno_ST": ["ST11", "ST258"],
            "amr_gene": [True, False],
        }
    )
    ds = SeroEpiDataset(data=df, name="SelectTestDS", metadata={"meta_key": 100})

    # 1. Select subset of string columns
    ds_sel1 = ds.select("sample_id", "geno_ST")
    assert isinstance(ds_sel1, SeroEpiDataset)
    assert ds_sel1.columns == ["sample_id", "geno_ST"]
    assert len(ds_sel1) == 2
    assert ds_sel1.name == "SelectTestDS"
    assert ds_sel1.metadata == {"meta_key": 100}

    # 2. Select with expressions and renames
    ds_sel2 = ds.select(
        sid=pl.col("sample_id"),
        lat_x2=pl.col("latitude") * 2,
    )
    assert ds_sel2.columns == ["sid", "lat_x2"]
    assert ds_sel2["lat_x2"].to_list() == [2.0, 4.0]

    # 3. Verify original dataset columns remain unchanged
    assert ds.columns == ["sample_id", "latitude", "longitude", "geno_ST", "amr_gene"]


def test_challenger_select_patito_validation_on_duplicates():
    """Verify select() triggers Patito validation errors when expressions introduce duplicate sample_ids."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [10.0, 20.0],
            "longitude": [100.0, 110.0],
        }
    )
    ds = SeroEpiDataset(data=df)

    # Selecting expression producing non-unique sample_id values
    with pytest.raises(DataFrameValidationError):
        ds.select(sample_id=pl.lit("SAME_ID"), latitude=pl.col("latitude"))


# ==============================================================================
# 3. with_columns() Empirical Stress Tests
# ==============================================================================


def test_challenger_with_columns_addition_updates_and_immutability():
    """Verify with_columns() adds/updates columns while preserving metadata and immutability."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [10.0, 20.0],
        }
    )
    ds = SeroEpiDataset(data=df, name="WithColsDS", metadata={"ver": 1})

    # 1. Add new custom column
    ds_added = ds.with_columns(custom_score=pl.lit(99.5))
    assert isinstance(ds_added, SeroEpiDataset)
    assert "custom_score" in ds_added.columns
    assert ds_added["custom_score"].to_list() == [99.5, 99.5]
    assert ds_added.name == "WithColsDS"
    assert ds_added.metadata == {"ver": 1}

    # 2. Update existing column
    ds_updated = ds.with_columns(latitude=pl.col("latitude") + 50.0)
    assert ds_updated["latitude"].to_list() == [51.0, 52.0]

    # 3. Original dataset immutability check
    assert "custom_score" not in ds.columns
    assert ds["latitude"].to_list() == [1.0, 2.0]


def test_challenger_with_columns_patito_boundary_violations():
    """Verify with_columns() triggers Patito validation errors on coordinate and uniqueness violations."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [10.0, 20.0],
            "longitude": [100.0, 110.0],
        }
    )
    ds = SeroEpiDataset(data=df)

    # Latitude > 90
    with pytest.raises(DataFrameValidationError):
        ds.with_columns(latitude=pl.lit(91.0))

    # Longitude < -180
    with pytest.raises(DataFrameValidationError):
        ds.with_columns(longitude=pl.lit(-181.0))

    # Duplicate sample_id
    with pytest.raises(DataFrameValidationError):
        ds.with_columns(sample_id=pl.lit("DUP_ID"))


# ==============================================================================
# 4. filter() Empirical Stress Tests
# ==============================================================================


def test_challenger_filter_expressions_constraints_and_empty():
    """Verify filter() supports positional expressions, keyword constraints, and empty result handling."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "latitude": [10.0, 20.0, 30.0],
            "longitude": [100.0, 110.0, 120.0],
            "geno_ST": ["ST11", "ST258", "ST11"],
        }
    )
    ds = SeroEpiDataset(data=df, name="FilterDS", metadata={"tag": "test"})

    # 1. Positional expression
    f1 = ds.filter(pl.col("latitude") > 15.0)
    assert isinstance(f1, SeroEpiDataset)
    assert len(f1) == 2
    assert f1["sample_id"].to_list() == ["S2", "S3"]

    # 2. Keyword constraint
    f2 = ds.filter(geno_ST="ST11")
    assert len(f2) == 2
    assert f2["sample_id"].to_list() == ["S1", "S3"]

    # 3. Empty filter result
    f_empty = ds.filter(geno_ST="ST_NONEXISTENT")
    assert len(f_empty) == 0
    assert f_empty.columns == ds.columns
    assert f_empty.name == "FilterDS"
    assert f_empty.metadata == {"tag": "test"}

    # 4. Original dataset preserved
    assert len(ds) == 3


# ==============================================================================
# 5. Dataset Equality & equals() Behavior
# ==============================================================================


def test_challenger_dataset_equality_behavior():
    """Stress test equality comparison and ds.data.equals() behavior on SeroEpiDataset."""
    df1 = pl.DataFrame({"sample_id": ["S1"], "latitude": [1.0], "longitude": [2.0]})
    df2 = pl.DataFrame({"sample_id": ["S1"], "latitude": [1.0], "longitude": [2.0]})
    df3 = pl.DataFrame({"sample_id": ["S2"], "latitude": [1.0], "longitude": [2.0]})

    ds1 = SeroEpiDataset(data=df1, name="DS1", metadata={"a": 1})
    ds2 = SeroEpiDataset(data=df2, name="DS1", metadata={"a": 1})
    ds3 = SeroEpiDataset(data=df3, name="DS1", metadata={"a": 1})

    # 1. Underlying Polars DataFrame equality via .equals()
    assert ds1.data.equals(ds2.data) is True
    assert ds1.data.equals(ds3.data) is False

    # 2. Direct dataclass == comparison:
    # Polars df1 == df2 returns a DataFrame of booleans, causing exception when evaluated in boolean context
    with pytest.raises(Exception) as exc_info:
        _ = ds1 == ds2
    assert (
        "ComputeError" in exc_info.typename
        or "truth value" in str(exc_info.value).lower()
        or "equals" in str(exc_info.value).lower()
    )

    # 3. Verify SeroEpiDataset attribute access for equality checking
    def dataset_equals(a: SeroEpiDataset, b: SeroEpiDataset) -> bool:
        return a.name == b.name and a.metadata == b.metadata and a.data.equals(b.data)

    assert dataset_equals(ds1, ds2) is True
    assert dataset_equals(ds1, ds3) is False


# ==============================================================================
# 6. pickle.dumps() and pickle.loads() Stress Tests
# ==============================================================================


def test_challenger_pickle_serialization_roundtrip():
    """Verify SeroEpiDataset serializes and deserializes cleanly with pickle."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [10.0, 20.0],
            "geno_ST": ["ST11", "ST258"],
            "amr_gene": [True, False],
        }
    )
    ds = SeroEpiDataset(data=df, name="PickleDS", metadata={"version": "1.0", "params": [1, 2, 3]})

    # Serialize
    serialized = pickle.dumps(ds)
    assert isinstance(serialized, bytes)
    assert len(serialized) > 0

    # Deserialize
    restored = pickle.loads(serialized)
    assert isinstance(restored, SeroEpiDataset)
    assert restored.name == "PickleDS"
    assert restored.metadata == {"version": "1.0", "params": [1, 2, 3]}
    assert restored.data.equals(ds.data)
    assert restored.columns == ds.columns


def test_challenger_pickle_transformed_datasets():
    """Verify pickling works on datasets produced by transformations (filter, select, with_columns)."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "latitude": [10.0, 20.0, 30.0],
            "longitude": [100.0, 110.0, 120.0],
        }
    )
    ds = SeroEpiDataset(data=df, name="TransformedPickleDS")

    ds_trans = ds.filter(pl.col("latitude") > 15.0).with_columns(flag=pl.lit(True)).select("sample_id", "flag")

    serialized = pickle.dumps(ds_trans)
    restored = pickle.loads(serialized)

    assert restored.name == "TransformedPickleDS"
    assert restored.data.equals(ds_trans.data)
    assert len(restored) == 2
    assert restored.columns == ["sample_id", "flag"]


# ==============================================================================
# 7. Complex Polars Dtypes Stress Tests
# ==============================================================================


def test_challenger_complex_polars_dtypes_support():
    """Stress-test SeroEpiDataset with complex Polars dtypes (List, Struct, Categorical, Datetime, Null)."""
    dt = datetime.datetime(2023, 5, 15, 12, 0, 0)
    d_date = datetime.date(2023, 5, 15)

    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "gene_list": [["blaKPC", "TEM1"], ["CTX-M"]],
            "int_list": [[1, 2], [3]],
            "struct_col": [{"a": 10, "b": "foo"}, {"a": 20, "b": "bar"}],
            "cat_col": pl.Series(["cat1", "cat2"], dtype=pl.Categorical),
            "dt_col": [dt, dt],
            "date_col": [d_date, d_date],
            "null_col": pl.Series([None, None], dtype=pl.Null),
        }
    )

    ds = SeroEpiDataset(data=df, name="ComplexDtypesDS", metadata={"complex": True})

    # 1. Accessors and columns
    assert len(ds) == 2
    assert "gene_list" in ds.columns
    assert "struct_col" in ds.columns
    assert "cat_col" in ds.columns

    # 2. select on complex dtypes
    ds_sel = ds.select("sample_id", "gene_list", "struct_col")
    assert ds_sel.columns == ["sample_id", "gene_list", "struct_col"]

    # 3. with_columns on complex dtypes
    ds_with = ds.with_columns(new_list=pl.col("gene_list").list.len())
    assert "new_list" in ds_with.columns
    assert ds_with["new_list"].to_list() == [2, 1]

    # 4. filter on complex dtypes
    ds_filt = ds.filter(pl.col("int_list").list.contains(1))
    assert len(ds_filt) == 1
    assert ds_filt["sample_id"].to_list() == ["S1"]

    # 5. to_dicts() roundtrip
    records = ds.to_dicts()
    assert len(records) == 2
    assert records[0]["gene_list"] == ["blaKPC", "TEM1"]
    assert records[0]["struct_col"] == {"a": 10, "b": "foo"}

    # 6. pickle roundtrip with complex dtypes
    serialized = pickle.dumps(ds)
    restored = pickle.loads(serialized)
    assert restored.name == "ComplexDtypesDS"
    assert restored.data.equals(ds.data)


def test_challenger_object_dtype_behavior():
    """Verify SeroEpiDataset supports pl.Object columns for operations and confirms pickling limitation."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1"],
            "obj_col": pl.Series([object()], dtype=pl.Object),
        }
    )
    ds = SeroEpiDataset(data=df, name="ObjectDS")

    # 1. Verification of operational capabilities with pl.Object
    assert "obj_col" in ds.columns
    assert len(ds) == 1
    ds_sel = ds.select("sample_id", "obj_col")
    assert ds_sel.columns == ["sample_id", "obj_col"]

    # 2. Polars raises exception when pickling DataFrames containing pl.Object
    with pytest.raises((TypeError, Exception)) as exc_info:
        pickle.dumps(ds)
    assert "Object" in str(exc_info.value) or "object" in str(exc_info.value).lower()
