"""Empirical stress tests written by Challenger 2 for SeroEpi Milestone 2 Iteration 4.

Focus: Statistical estimators, metadata fallbacks, StrEnum traits, edge cases, SVI pipelines.
"""

from enum import StrEnum
import tempfile
import numpy as np
import polars as pl
import pytest

from seroepi.constants import AggregationType, BayesianInferenceMethod
from seroepi.dataset import SeroEpiDataset
from seroepi.domains.base import register_dataframe_metadata, get_dataframe_metadata, MetaDataFrame
from seroepi.estimators.diversity import (
    AlphaDiversityEstimator,
    BetaDiversityEstimator,
)
from seroepi.estimators.incidence import (
    BayesianIncidenceEstimator,
    GLMIncidenceEstimator,
)
from seroepi.estimators.prevalence import (
    BayesianPrevalenceEstimator,
    GLMPrevalenceEstimator,
    SpatialPrevalenceEstimator,
    UnpooledPrevalenceEstimator,
)
from seroepi.traits import ChoiceEnum, KpAmrTrait, KpSeroTrait


class CustomTrait(ChoiceEnum):
    GENE_X = "gene_X"
    GENE_Y = "gene_Y"


# ==============================================================================
# 1. UnpooledPrevalenceEstimator Tests
# ==============================================================================

def test_unpooled_prevalence_estimator_with_and_without_metadata():
    """Test UnpooledPrevalenceEstimator with metadata, missing metadata, and custom StrEnum."""
    # Case A: Explicit dataset metadata
    df = pl.DataFrame({
        "region": ["North", "South", "North", "South"],
        "target": ["blaKPC", "blaKPC", "blaNDM", "blaNDM"],
        "event": [5, 10, 0, 2],
        "n": [50, 50, 40, 40],
    })
    meta = {
        "trait": KpAmrTrait.BLA_ACQUIRED,
        "stratified_by": ["region"],
        "aggregation_type": AggregationType.TRAIT,
    }
    register_dataframe_metadata(df, meta)

    estimator = UnpooledPrevalenceEstimator(method="wilson")
    res = estimator.calculate(df)

    assert res.trait == "amr_Bla_acquired"
    assert res.stratified_by == ["region"]
    assert len(res.data) == 4
    assert "estimate" in res.data.columns
    assert "lower" in res.data.columns
    assert "upper" in res.data.columns

    # Case B: Missing metadata (no metadata registered)
    df_no_meta = pl.DataFrame({
        "region": ["North", "South"],
        "target": ["gene_A", "gene_A"],
        "event": [3, 7],
        "n": [30, 30],
    })
    res_no_meta = estimator.calculate(df_no_meta)
    assert res_no_meta.trait == "gene_A"  # Inferred from single target value

    # Case C: Non-unique target column, missing metadata
    df_multi_target = pl.DataFrame({
        "region": ["North", "South"],
        "target": ["gene_A", "gene_B"],
        "event": [3, 7],
        "n": [30, 30],
    })
    res_multi = estimator.calculate(df_multi_target)
    assert res_multi.trait == "target"  # Fallback to "target"


def test_unpooled_prevalence_estimator_methods_and_edge_values():
    """Test all interval estimation methods with 0 events, 100% events, and 0 denominator."""
    methods = ["wilson", "wald", "agresti_coull", "clopper_pearson", "jeffreys"]
    df = pl.DataFrame({
        "target": ["T1", "T2", "T3"],
        "event": [0, 50, 0],
        "n": [50, 50, 0],  # n=0 triggers division by zero
    })
    
    for method in methods:
        estimator = UnpooledPrevalenceEstimator(method=method)  # type: ignore
        res = estimator.calculate(df)
        assert len(res.data) == 3
        # Check no NaN values propagate in result DataFrame
        assert not np.isnan(res.data["estimate"].to_numpy()).any()
        assert not np.isnan(res.data["lower"].to_numpy()).any()
        assert not np.isnan(res.data["upper"].to_numpy()).any()


def test_unpooled_prevalence_estimator_invalid_method():
    with pytest.raises(ValueError, match="Unknown method"):
        UnpooledPrevalenceEstimator(method="invalid_method")  # type: ignore


# ==============================================================================
# 2. GLMPrevalenceEstimator Tests
# ==============================================================================

def test_glm_prevalence_estimator_fit_predict_pipeline():
    """Test GLMPrevalenceEstimator fit/predict pipeline with custom StrEnum and missing metadata."""
    df_train = pl.DataFrame({
        "region": ["North", "North", "South", "South"],
        "target": ["gene_A", "gene_B", "gene_A", "gene_B"],
        "event": [10, 20, 5, 15],
        "n": [100, 100, 100, 100],
    })
    meta = {
        "trait": CustomTrait.GENE_X,
        "stratified_by": ["region"],
    }
    register_dataframe_metadata(df_train, meta)

    estimator = GLMPrevalenceEstimator()
    fitted = estimator.fit(df_train)
    assert fitted.is_fitted_

    res = fitted.predict(df_train)
    assert res.trait == "gene_X"
    assert "estimate" in res.data.columns
    assert len(res.data) == 4

    # Test save/load model
    with tempfile.NamedTemporaryFile(suffix=".joblib") as tmp:
        fitted.save_model(tmp.name)
        loaded = GLMPrevalenceEstimator.load_model(tmp.name)
        assert loaded.is_fitted_
        res_loaded = loaded.predict(df_train)
        assert len(res_loaded.data) == 4


def test_glm_prevalence_estimator_strata_validation():
    """Test that continuous float strata raise ValueError in _validate_suitability."""
    df = pl.DataFrame({
        "float_strata": [1.5, 2.5, 3.5],
        "event": [1, 2, 3],
        "n": [10, 10, 10],
    })
    estimator = GLMPrevalenceEstimator()
    with pytest.raises(ValueError, match="continuous float"):
        estimator.fit(df)


# ==============================================================================
# 3. BayesianPrevalenceEstimator Tests (MCMC & SVI)
# ==============================================================================

def test_bayesian_prevalence_estimator_zero_padding_guard():
    """Test that BayesianPrevalenceEstimator checks for zero padding metadata."""
    df = pl.DataFrame({
        "region": ["North", "South"],
        "target": ["T1", "T2"],
        "event": [5, 10],
        "n": [50, 50],
    })
    meta = {"is_zero_padded": False}
    register_dataframe_metadata(df, meta)

    estimator = BayesianPrevalenceEstimator(method=BayesianInferenceMethod.SVI, svi_steps=50, num_samples=100)
    with pytest.raises(ValueError, match="zero-padded"):
        estimator.fit(df)


def test_bayesian_prevalence_estimator_svi_pipeline():
    """Test BayesianPrevalenceEstimator using Stochastic Variational Inference (SVI)."""
    df = pl.DataFrame({
        "region": ["North", "North", "South", "South"],
        "target": ["T1", "T2", "T1", "T2"],
        "event": [10, 5, 20, 15],
        "n": [100, 100, 100, 100],
    })
    meta = {
        "trait": KpSeroTrait.K_LOCUS,
        "stratified_by": ["region"],
        "is_zero_padded": True,
    }
    register_dataframe_metadata(df, meta)

    estimator = BayesianPrevalenceEstimator(
        method=BayesianInferenceMethod.SVI, svi_steps=50, num_samples=100
    )
    fitted = estimator.fit(df)
    assert fitted.is_fitted_

    res = fitted.predict(df)
    assert res.trait == "geno_K_locus"
    assert "estimate" in res.data.columns
    assert len(res.data) == 4
    assert res.method == "bayesian_svi"


def test_bayesian_prevalence_estimator_mcmc_pipeline_and_diagnostics():
    """Test BayesianPrevalenceEstimator MCMC pipeline and diagnostics extraction."""
    df = pl.DataFrame({
        "region": ["North", "North", "South", "South"],
        "target": ["T1", "T2", "T1", "T2"],
        "event": [10, 5, 20, 15],
        "n": [100, 100, 100, 100],
    })
    meta = {"is_zero_padded": True}
    register_dataframe_metadata(df, meta)

    estimator = BayesianPrevalenceEstimator(
        method=BayesianInferenceMethod.MCMC, num_warmup=10, num_samples=20, num_chains=1
    )
    fitted = estimator.fit(df)
    assert fitted.is_fitted_

    res = fitted.predict(df)
    assert res.method == "bayesian_mcmc"

    diag_df = fitted.diagnostics()
    assert isinstance(diag_df, pl.DataFrame)
    assert "Parameter" in diag_df.columns


# ==============================================================================
# 4. AlphaDiversityEstimator Tests
# ==============================================================================

def test_alpha_diversity_estimator_with_custom_strenum_and_missing_meta():
    """Test AlphaDiversityEstimator with custom StrEnum target and fallback logic."""
    df = pl.DataFrame({
        "region": ["North", "North", "South", "South"],
        "target": ["K1", "K2", "K1", "K2"],
        "variant_count": [10, 20, 5, 15],
        "n_total": [30, 30, 20, 20],
    })
    
    # Case A: Target specified in init via StrEnum
    estimator = AlphaDiversityEstimator(target=KpSeroTrait.K_LOCUS)
    res = estimator.calculate(df)
    assert res.trait == "geno_K_locus"
    assert "shannon" in res.data.columns
    assert "simpson" in res.data.columns
    assert "richness" in res.data.columns

    # Case B: Missing target in init, inferred from metadata
    meta = {"trait": CustomTrait.GENE_Y, "stratified_by": ["region"]}
    register_dataframe_metadata(df, meta)
    estimator_no_init = AlphaDiversityEstimator()
    res_meta = estimator_no_init.calculate(df)
    assert res_meta.trait == "gene_Y"

    # Case C: No target in init, no metadata -> inferred from target column if unique/non-unique
    df_no_meta = pl.DataFrame({
        "region": ["North", "South"],
        "variant_count": [10, 20],
        "target": ["single_trait", "single_trait"],
    })
    res_inferred = AlphaDiversityEstimator().calculate(df_no_meta)
    assert res_inferred.trait == "single_trait"


def test_alpha_diversity_estimator_zero_counts():
    """Test AlphaDiversityEstimator handles zero variant counts without crashing."""
    df = pl.DataFrame({
        "region": ["North", "North"],
        "target": ["K1", "K2"],
        "variant_count": [0, 0],
    })
    estimator = AlphaDiversityEstimator(target="K_locus")
    res = estimator.calculate(df)
    assert len(res.data) == 0


def test_alpha_diversity_estimator_missing_target_raises():
    """Test that missing target raises ValueError."""
    df = pl.DataFrame({
        "region": ["North", "South"],
        "variant_count": [10, 20],
    })
    estimator = AlphaDiversityEstimator()
    with pytest.raises(ValueError, match="Target trait must be defined"):
        estimator.calculate(df)


# ==============================================================================
# 5. BetaDiversityEstimator Tests
# ==============================================================================

def test_beta_diversity_estimator_matrix_generation():
    """Test BetaDiversityEstimator distance matrix calculation."""
    df = pl.DataFrame({
        "region": ["North", "North", "South", "South"],
        "target": ["K1", "K2", "K1", "K2"],
        "variant_count": [10, 20, 5, 25],
    })
    meta = {"trait": KpSeroTrait.K_LOCUS, "stratified_by": ["region"]}
    register_dataframe_metadata(df, meta)

    estimator = BetaDiversityEstimator(metric="braycurtis")
    res = estimator.calculate(df)
    assert res.trait == "geno_K_locus"
    assert res.metric == "braycurtis"
    assert res.data.shape == (2, 2)
    assert "North" in res.data.columns
    assert "South" in res.data.columns


def test_beta_diversity_estimator_missing_strata_raises():
    """Test that BetaDiversityEstimator without strata raises ValueError."""
    df = pl.DataFrame({
        "target": ["K1", "K2"],
        "variant_count": [10, 20],
    })
    estimator = BetaDiversityEstimator(target="K_locus")
    with pytest.raises(ValueError, match="Beta diversity requires at least one stratification level"):
        estimator.calculate(df)


# ==============================================================================
# 6. SpatialPrevalenceEstimator Tests
# ==============================================================================

def test_spatial_prevalence_estimator_svi_pipeline():
    """Test SpatialPrevalenceEstimator fit and predict using SVI."""
    df = pl.DataFrame({
        "lat": [-33.8, -33.8, -37.8, -37.8],
        "lon": [151.2, 151.2, 144.9, 144.9],
        "target": ["T1", "T2", "T1", "T2"],
        "event": [5, 15, 2, 8],
        "n": [50, 50, 50, 50],
    })
    meta = {
        "trait": KpAmrTrait.BLA_ACQUIRED,
        "is_zero_padded": True,
    }
    register_dataframe_metadata(df, meta)

    estimator = SpatialPrevalenceEstimator(
        method=BayesianInferenceMethod.SVI, svi_steps=50, num_samples=100
    )
    fitted = estimator.fit(df)
    assert fitted.is_fitted_

    res = fitted.predict(df)
    assert res.trait == "amr_Bla_acquired"
    assert "estimate" in res.data.columns
    assert len(res.data) == 4


def test_spatial_prevalence_estimator_missing_coords_raises():
    """Test that missing lat/lon columns raise KeyError."""
    df = pl.DataFrame({
        "target": ["T1", "T2"],
        "event": [5, 10],
        "n": [50, 50],
    })
    meta = {"is_zero_padded": True}
    register_dataframe_metadata(df, meta)

    estimator = SpatialPrevalenceEstimator(method=BayesianInferenceMethod.SVI, svi_steps=10)
    with pytest.raises(KeyError, match="Spatial estimator requires"):
        estimator.fit(df)


# ==============================================================================
# 7. GLMIncidenceEstimator & BayesianIncidenceEstimator Tests
# ==============================================================================

def test_glm_incidence_estimator_fit_predict():
    """Test GLMIncidenceEstimator fit and predict with forecasting horizon."""
    dates = [
        "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01",
        "2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"
    ]
    df = pl.DataFrame({
        "date": pl.Series(dates).str.to_date(),
        "region": ["North"] * 4 + ["South"] * 4,
        "target": ["gene_A"] * 8,
        "variant_count": [5, 8, 12, 15, 2, 4, 3, 6],
        "total_sequenced": [100, 100, 100, 100, 100, 100, 100, 100],
    })
    meta = {"trait": "gene_A", "freq": "month", "stratified_by": ["region"]}
    register_dataframe_metadata(df, meta)

    estimator = GLMIncidenceEstimator(forecast_horizon=2)
    fitted = estimator.fit(df)
    assert fitted.is_fitted_

    res = fitted.predict(df)
    assert res.trait == "gene_A"
    assert "IRR" in res.model_results.columns
    # Check that predictions include future forecast dates
    assert len(res.data) > len(df)


def test_bayesian_incidence_estimator_fit_predict():
    """Test BayesianIncidenceEstimator BSTS model with SVI."""
    dates = ["2025-01-01", "2025-02-01", "2025-03-01", "2025-04-01"]
    df = pl.DataFrame({
        "date": pl.Series(dates).str.to_date(),
        "region": ["North"] * 4,
        "target": ["gene_A"] * 4,
        "variant_count": [5, 8, 12, 15],
        "total_sequenced": [100, 100, 100, 100],
    })
    meta = {"trait": "gene_A", "freq": "month", "stratified_by": ["region"]}
    register_dataframe_metadata(df, meta)

    estimator = BayesianIncidenceEstimator(
        method=BayesianInferenceMethod.SVI, svi_steps=50, num_samples=100, forecast_horizon=2
    )
    fitted = estimator.fit(df)
    assert fitted.is_fitted_

    res = fitted.predict(df)
    assert res.trait == "gene_A"
    assert res.method == "bsts_forecast_svi"
