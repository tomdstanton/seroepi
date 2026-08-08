"""Shared pytest fixtures for E2E and unit testing of SeroEpi."""

import datetime

import polars as pl
import pytest
from shiny import reactive

from seroepi.constants import AggregationType
from seroepi.dataset import SeroEpiDataset
from seroepi.estimators.base import PrevalenceEstimates


# -----------------------------------------------------------------------------
# Polars DataFrame Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def sample_polars_df() -> pl.DataFrame:
    """Provide a standard, valid Polars DataFrame with full domain columns."""
    return pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "latitude": [1.23, 1.24, 1.25, 1.26],
            "longitude": [103.81, 103.82, 103.83, 103.84],
            "spatial_Continent": ["Asia", "Asia", "Asia", "Asia"],
            "spatial_Country": ["Singapore", "Singapore", "Malaysia", "Malaysia"],
            "geno_ST": ["ST11", "ST258", "ST11", "ST258"],
            "amr_blaKPC": [True, False, True, True],
            "vir_geneA": [True, True, False, False],
            "temporal_Collection_Date": [
                datetime.datetime(2023, 1, 1),
                datetime.datetime(2023, 1, 2),
                datetime.datetime(2023, 1, 3),
                datetime.datetime(2023, 1, 4),
            ],
            "K_locus": ["K1", "K1", "K2", "K3"],
        }
    )


@pytest.fixture
def mock_kleborate_df() -> pl.DataFrame:
    """Provide a raw Kleborate/Pathogenwatch output DataFrame before parsing."""
    return pl.DataFrame(
        {
            "Genome Name": ["Isolate_01", "Isolate_02"],
            "Collection Date": ["2023-05-15", "2023-05-16"],
            "ST": ["ST258", "ST11"],
            "K_locus": ["K1", "K2"],
            "O_locus": ["O1", "O2"],
            "Bla_KPC_genes": ["blaKPC-2", "-"],
            "Latitude": [1.23, 1.24],
            "Longitude": [103.81, 103.82],
        }
    )


@pytest.fixture
def sample_dataset(sample_polars_df: pl.DataFrame) -> SeroEpiDataset:
    """Provide a pre-instantiated SeroEpiDataset object."""
    return SeroEpiDataset(data=sample_polars_df, name="TestDataset")


@pytest.fixture
def invalid_patito_df() -> pl.DataFrame:
    """Provide a DataFrame with invalid lat/lon values to test Patito validation failure."""
    return pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [999.0, 20.0],  # 999.0 violates latitude boundary [-90, 90]
            "longitude": [100.0, 110.0],
        }
    )


@pytest.fixture
def mock_prevalence_estimates() -> PrevalenceEstimates:
    """Provide a pre-calculated PrevalenceEstimates object for formulation testing."""
    prev_data = pl.DataFrame(
        {
            "target": ["K1", "K2", "K3"],
            "estimate": [0.5, 0.3, 0.2],
            "hospital": ["H1", "H1", "H1"],
        }
    )
    return PrevalenceEstimates(
        data=prev_data,
        stratified_by=["hospital"],
        adjusted_for=None,
        trait="K_locus",
        aggregation_type=AggregationType.COMPOSITIONAL,
        method="unpooled_wilson",
    )


# -----------------------------------------------------------------------------
# Shiny UI & Server Reactive State Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture
def mock_app_state() -> dict:
    """Provide a fresh app_state dictionary matching app.app main_server structure."""
    return {
        "shared_df": reactive.Value(None),
        "shared_dist": reactive.Value(None),
        "shared_agg_df": reactive.Value(None),
        "prev_results": reactive.Value(None),
        "fitted_estimator": reactive.Value(None),
        "pw_collections_cache": reactive.Value({}),
        "baseline_res": reactive.Value(None),
        "current_formulation": reactive.Value(None),
        "results_registry": reactive.Value({}),
        "formulation_registry": reactive.Value({}),
        "dataset_registry": reactive.Value({}),
        "active_dataset_name": reactive.Value(None),
        "active_run_name": reactive.Value(None),
        "active_vac_name": reactive.Value(None),
    }
