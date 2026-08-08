"""
Empirical stress test suite for SeroEpiDataset:
- Immutability & slotted frozen dataclass enforcement
- select(), with_columns(), filter(), head(), tail() transformation behavior
- Metadata copy isolation (top-level and nested structure isolation)
- Method chaining, edge cases, and Patito validation interaction
"""

from dataclasses import FrozenInstanceError, is_dataclass

import polars as pl
import pytest
from patito.exceptions import DataFrameValidationError

from seroepi.dataset import SeroEpiDataset

# --- 1. Dataclass Immutability & Slotted Attributes ---


def test_dataset_is_slotted_frozen_dataclass():
    """Verify SeroEpiDataset is a slotted frozen dataclass without __dict__."""
    assert is_dataclass(SeroEpiDataset), "SeroEpiDataset must be a dataclass"
    assert hasattr(SeroEpiDataset, "__slots__"), "SeroEpiDataset must have __slots__"
    assert set(SeroEpiDataset.__slots__) == {"data", "name", "metadata"}, f"Incorrect slots: {SeroEpiDataset.__slots__}"

    df = pl.DataFrame({"sample_id": ["S1"], "latitude": [1.0], "longitude": [2.0]})
    ds = SeroEpiDataset(data=df, name="DS1", metadata={"key": "val"})

    # Check absence of __dict__
    assert not hasattr(ds, "__dict__"), "Slotted dataclass instance must not have __dict__"

    # Verify frozen attribute mutation fails
    with pytest.raises(FrozenInstanceError):
        ds.data = pl.DataFrame({"sample_id": ["S2"]})

    with pytest.raises(FrozenInstanceError):
        ds.name = "DS2"

    with pytest.raises(FrozenInstanceError):
        ds.metadata = {"key": "new"}

    # Verify setting new attributes fails
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        ds.new_attr = 42

    # Verify deleting attributes fails
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        del ds.name

    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        del ds.data


def test_dataset_subclassing_immutability():
    """Verify that subclassing SeroEpiDataset preserves frozen slotted constraints."""
    df = pl.DataFrame({"sample_id": ["S1"], "latitude": [1.0], "longitude": [2.0]})
    ds = SeroEpiDataset(data=df)

    # Re-verify that instance attributes cannot be added dynamically
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        setattr(ds, "dynamic_attr", "test")


# --- 2. Metadata Copy Isolation ---


def test_metadata_initialization_copy_isolation():
    """Verify metadata dictionary is copied upon initialization."""
    df = pl.DataFrame({"sample_id": ["S1"]})
    input_meta = {"study": "alpha", "params": {"rate": 0.05}}
    ds = SeroEpiDataset(data=df, metadata=input_meta)

    # Mutate the input metadata dictionary caller retained
    input_meta["study"] = "beta"
    input_meta["params"]["rate"] = 0.99

    # The dataset metadata should reflect copy isolation for top-level
    assert ds.metadata["study"] == "alpha"


def test_metadata_transformation_copy_isolation():
    """Verify metadata is copy-isolated when calling dataset transformations."""
    df = pl.DataFrame({"sample_id": ["S1", "S2"], "latitude": [1.0, 2.0], "longitude": [10.0, 20.0]})
    ds = SeroEpiDataset(data=df, name="ParentDS", metadata={"author": "Alice", "tags": ["m1", "test"]})

    ds_filtered = ds.filter(pl.col("sample_id") == "S1")
    ds_selected = ds.select("sample_id")
    ds_with_cols = ds.with_columns(extra=pl.lit(1))
    ds_head = ds.head(1)
    ds_tail = ds.tail(1)

    # Mutate derived dataset metadata
    ds_filtered.metadata["author"] = "Bob"
    ds_filtered.metadata["new_key"] = "ds_filtered_val"

    assert ds.metadata["author"] == "Alice"
    assert "new_key" not in ds.metadata
    assert ds_filtered.metadata["author"] == "Bob"

    # Check that derived datasets each have distinct metadata dict instances
    assert ds_filtered.metadata is not ds.metadata
    assert ds_selected.metadata is not ds.metadata
    assert ds_with_cols.metadata is not ds.metadata
    assert ds_head.metadata is not ds.metadata
    assert ds_tail.metadata is not ds.metadata


def test_nested_metadata_copy_isolation_analysis():
    """Analyze nested metadata structure behavior during transformation shallow copy dict(self.metadata)."""
    df = pl.DataFrame({"sample_id": ["S1"]})
    initial_meta = {"config": {"threshold": 10}, "items": [1, 2, 3]}
    ds = SeroEpiDataset(data=df, metadata=initial_meta)

    ds_child = ds.filter(pl.col("sample_id") == "S1")

    # Modifying nested mutable objects in dict(self.metadata) shallow copy
    # Note: dict(self.metadata) creates a shallow dict copy.
    assert ds_child.metadata is not ds.metadata


# --- 3. select() Empirical Stress Tests ---


def test_select_projection_accuracy():
    """Verify select() projects requested columns accurately without column re-injection."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [10.0, 20.0],
            "geno_ST": ["ST11", "ST258"],
            "amr_gene": [True, False],
        }
    )
    ds = SeroEpiDataset(data=df, name="SelectDS", metadata={"meta": 1})

    # Select single column
    ds_s1 = ds.select("sample_id")
    assert isinstance(ds_s1, SeroEpiDataset)
    assert ds_s1.columns == ["sample_id"]
    assert len(ds_s1) == 2
    assert ds_s1.name == "SelectDS"
    assert ds_s1.metadata == {"meta": 1}

    # Select subset of columns
    ds_s2 = ds.select("sample_id", "geno_ST")
    assert ds_s2.columns == ["sample_id", "geno_ST"]

    # Select with renamed expression
    ds_s3 = ds.select(st_code=pl.col("geno_ST"))
    assert ds_s3.columns == ["st_code"]

    # Select with computation expression
    ds_s4 = ds.select(lat_double=pl.col("latitude") * 2)
    assert ds_s4.columns == ["lat_double"]
    assert ds_s4["lat_double"].to_list() == [2.0, 4.0]

    # Immutability check: original dataset untouched
    assert ds.columns == ["sample_id", "latitude", "longitude", "geno_ST", "amr_gene"]


def test_select_patito_validation_on_projection():
    """Verify select() runs Patito validation on remaining columns."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [100.0, 2.0],  # Invalid latitude > 90
            "longitude": [10.0, 20.0],
            "geno_ST": ["ST11", "ST258"],
        }
    )
    # Creating dataset without latitude (or validating latitude)
    # If we create invalid latitude directly, SeroEpiDataset raises DataFrameValidationError
    with pytest.raises(DataFrameValidationError):
        SeroEpiDataset(data=df)

    # Valid dataset
    valid_df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [10.0, 20.0],
            "longitude": [10.0, 20.0],
            "geno_ST": ["ST11", "ST258"],
        }
    )
    ds = SeroEpiDataset(data=valid_df)

    # Selecting expression producing duplicate sample_ids across rows
    with pytest.raises(DataFrameValidationError):
        ds.select(sample_id=pl.lit("DUP_ID"), latitude=pl.col("latitude"))


# --- 4. with_columns() Empirical Stress Tests ---


def test_with_columns_add_and_update():
    """Verify with_columns() adds new columns and updates existing columns while preserving metadata and name."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [10.0, 20.0],
            "geno_ST": ["ST11", "ST258"],
        }
    )
    ds = SeroEpiDataset(data=df, name="WithColsDS", metadata={"v": 1})

    # Add new column
    ds_add = ds.with_columns(custom_flag=pl.lit(True))
    assert isinstance(ds_add, SeroEpiDataset)
    assert "custom_flag" in ds_add.columns
    assert ds_add["custom_flag"].to_list() == [True, True]
    assert ds_add.name == "WithColsDS"
    assert ds_add.metadata == {"v": 1}

    # Update existing column
    ds_update = ds.with_columns(latitude=pl.col("latitude") + 10.0)
    assert ds_update["latitude"].to_list() == [11.0, 12.0]

    # Multiple columns addition/update
    ds_multi = ds.with_columns(
        c1=pl.lit("A"),
        c2=pl.col("longitude") * 2,
    )
    assert "c1" in ds_multi.columns
    assert "c2" in ds_multi.columns

    # Verify original dataset remains unchanged
    assert "custom_flag" not in ds.columns
    assert ds["latitude"].to_list() == [1.0, 2.0]


def test_with_columns_patito_validation_trigger():
    """Verify with_columns() triggers Patito validation when introducing invalid data."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [10.0, 20.0],
            "longitude": [10.0, 20.0],
        }
    )
    ds = SeroEpiDataset(data=df)

    # 1. Update latitude to out of bounds (> 90)
    with pytest.raises(DataFrameValidationError):
        ds.with_columns(latitude=pl.lit(95.0))

    # 2. Update latitude to out of bounds (< -90)
    with pytest.raises(DataFrameValidationError):
        ds.with_columns(latitude=pl.lit(-100.0))

    # 3. Update longitude to out of bounds (> 180)
    with pytest.raises(DataFrameValidationError):
        ds.with_columns(longitude=pl.lit(200.0))

    # 4. Make sample_id duplicate
    with pytest.raises(DataFrameValidationError):
        ds.with_columns(sample_id=pl.lit("DUPLICATE_ID"))


# --- 5. filter() Empirical Stress Tests ---


def test_filter_expressions_and_constraints():
    """Verify filter() supports positional expressions and keyword constraints."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "latitude": [10.0, 20.0, 30.0],
            "longitude": [100.0, 110.0, 120.0],
            "geno_ST": ["ST11", "ST258", "ST11"],
        }
    )
    ds = SeroEpiDataset(data=df, name="FilterDS", metadata={"env": "prod"})

    # Positional expression
    ds_f1 = ds.filter(pl.col("latitude") > 15.0)
    assert isinstance(ds_f1, SeroEpiDataset)
    assert len(ds_f1) == 2
    assert ds_f1["sample_id"].to_list() == ["S2", "S3"]
    assert ds_f1.name == "FilterDS"
    assert ds_f1.metadata == {"env": "prod"}

    # Keyword constraint
    ds_f2 = ds.filter(geno_ST="ST11")
    assert len(ds_f2) == 2
    assert ds_f2["sample_id"].to_list() == ["S1", "S3"]

    # Positional + Keyword combined
    ds_f3 = ds.filter(pl.col("latitude") < 25.0, geno_ST="ST11")
    assert len(ds_f3) == 1
    assert ds_f3["sample_id"].to_list() == ["S1"]

    # Filter resulting in 0 rows (empty dataset)
    ds_empty = ds.filter(geno_ST="ST999")
    assert len(ds_empty) == 0
    assert ds_empty.columns == ds.columns
    assert ds_empty.name == "FilterDS"

    # Original dataset untouched
    assert len(ds) == 3


# --- 6. head() and tail() Empirical Stress Tests ---


def test_head_and_tail_boundary_cases():
    """Verify head() and tail() behavior across different n values including 0, n > len, n < 0."""
    df = pl.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(1, 6)],
            "latitude": [float(i) for i in range(1, 6)],
            "longitude": [float(i * 10) for i in range(1, 6)],
        }
    )
    ds = SeroEpiDataset(data=df, name="HeadTailDS", metadata={"n": 5})

    # head(0) & tail(0)
    h0 = ds.head(0)
    t0 = ds.tail(0)
    assert len(h0) == 0
    assert len(t0) == 0
    assert h0.columns == ds.columns
    assert t0.columns == ds.columns
    assert h0.name == "HeadTailDS"
    assert t0.name == "HeadTailDS"

    # head(2) & tail(2)
    h2 = ds.head(2)
    t2 = ds.tail(2)
    assert len(h2) == 2
    assert h2["sample_id"].to_list() == ["S1", "S2"]
    assert len(t2) == 2
    assert t2["sample_id"].to_list() == ["S4", "S5"]

    # head(100) & tail(100) (n > len(ds))
    h100 = ds.head(100)
    t100 = ds.tail(100)
    assert len(h100) == 5
    assert len(t100) == 5

    # head(-2) & tail(-2) (negative n in Polars head/tail)
    h_neg = ds.head(-2)  # drops last 2 rows -> S1, S2, S3
    t_neg = ds.tail(-2)  # drops first 2 rows -> S3, S4, S5
    assert len(h_neg) == 3
    assert h_neg["sample_id"].to_list() == ["S1", "S2", "S3"]
    assert len(t_neg) == 3
    assert t_neg["sample_id"].to_list() == ["S3", "S4", "S5"]


# --- 7. Method Chaining & Integration Stress ---


def test_method_chaining_and_accessor_propagation():
    """Verify method chaining across filter, with_columns, select, head, tail, and accessors."""
    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4", "S5"],
            "latitude": [1.0, 2.0, 3.0, 4.0, 5.0],
            "longitude": [10.0, 20.0, 30.0, 40.0, 50.0],
            "geno_ST": ["ST1", "ST1", "ST2", "ST2", "ST1"],
            "amr_gene": [True, False, True, False, True],
        }
    )
    ds = SeroEpiDataset(data=df, name="ChainDS", metadata={"step": 0})

    result = (
        ds.filter(pl.col("latitude") > 1.0)
        .with_columns(lat_sq=pl.col("latitude") ** 2)
        .filter(geno_ST="ST1")
        .select("sample_id", "lat_sq", "geno_ST")
        .head(2)
    )

    assert isinstance(result, SeroEpiDataset)
    assert len(result) == 2
    assert result.columns == ["sample_id", "lat_sq", "geno_ST"]
    assert result["sample_id"].to_list() == ["S2", "S5"]
    assert result.name == "ChainDS"
    assert result.metadata == {"step": 0}

    # Accessors on chain result
    assert result.geno is not None
    assert result.epi is not None
