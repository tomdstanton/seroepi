"""Unit tests for Formulation class and formulation designers supporting StrEnum traits."""

import polars as pl

from seroepi.constants import AggregationType
from seroepi.estimators import PrevalenceEstimates
from seroepi.formulation import CustomFormulationDesigner, Formulation, PostHocFormulationDesigner
from seroepi.traits import CoreTrait, KpSeroTrait


def test_formulation_assess_coverage_strenum():
    """Verify Formulation assess_coverage works with StrEnum trait."""
    df = pl.DataFrame({"geno_K_locus": ["K1", "K2", "K3"]})
    rankings = pl.DataFrame({"target": ["K1", "K2"]})
    stability = pl.DataFrame()
    history = pl.DataFrame()

    f = Formulation(
        trait=KpSeroTrait.K_LOCUS,
        max_valency=2,
        rankings=rankings,
        stability_metrics=stability,
        permutation_history=history,
    )

    res_df = f.assess_coverage(df)
    assert "Vaccine_Coverage" in res_df.data.columns
    assert res_df.data["Vaccine_Coverage"].to_list() == [True, True, False]


def test_custom_formulation_designer_strenum():
    """Verify CustomFormulationDesigner supports StrEnum targets list."""
    designer = CustomFormulationDesigner(targets=[KpSeroTrait.K_LOCUS, "K2"])
    assert designer.targets == ["geno_K_locus", "K2"]

    raw_df = pl.DataFrame({"target": ["geno_K_locus", "K2"], "estimate": [0.6, 0.4]})
    prev_est = PrevalenceEstimates(
        data=raw_df,
        stratified_by=[],
        adjusted_for=None,
        trait="geno_K_locus",
        aggregation_type=AggregationType.COMPOSITIONAL,
        method="wilson",
    )

    fitted = designer.fit(prev_est)
    assert fitted.formulation_ is not None
    assert fitted.formulation_.get_formulation() == ["geno_K_locus", "K2"]


def test_post_hoc_formulation_designer_strenum_loo():
    """Verify PostHocFormulationDesigner accepts StrEnum for loo_col."""
    raw_df = pl.DataFrame(
        {
            "target": ["K1", "K2", "K1", "K2"],
            "estimate": [0.3, 0.2, 0.4, 0.1],
            "spatial_Country": ["France", "France", "Germany", "Germany"],
        }
    )

    prev_est = PrevalenceEstimates(
        data=raw_df,
        stratified_by=["spatial_Country"],
        adjusted_for=None,
        trait="geno_K_locus",
        aggregation_type=AggregationType.COMPOSITIONAL,
        method="wilson",
    )

    designer = PostHocFormulationDesigner(valency=2)
    fitted = designer.fit(prev_est, loo_col=CoreTrait.COUNTRY)
    assert fitted.formulation_ is not None
    assert not fitted.formulation_.permutation_history.is_empty()
    assert set(fitted.formulation_.permutation_history["holdout_group"].to_list()) == {"France", "Germany"}
