import dataclasses
import datetime

import polars as pl
import pytest

from app.utils import (
    ColMapper,
    _clean_ui_label,
    build_grouped_choices,
    format_metadata_ui,
)
from seroepi.constants import AggregationType, Domain
from seroepi.dataset import SeroEpiDataset
from seroepi.estimators.base import IncidenceEstimates, PrevalenceEstimates
from seroepi.estimators.prevalence import UnpooledPrevalenceEstimator
from seroepi.formulation import (
    CustomFormulationDesigner,
    Formulation,
    PostHocFormulationDesigner,
)


def test_formulation_dataclass_frozen_slotted():
    """Verify Formulation is a frozen slotted dataclass storing Polars DataFrames."""
    assert dataclasses.is_dataclass(Formulation)
    assert hasattr(Formulation, "__slots__")

    rankings = pl.DataFrame({"target": ["K1", "K2", "K3"], "estimate": [0.5, 0.3, 0.1], "baseline_rank": [1, 2, 3]})
    metrics = pl.DataFrame(
        {
            "target": ["K1", "K2"],
            "mean_loo_rank": [1.0, 2.0],
            "rank_variance": [0.0, 0.0],
            "probability_in_top_n": [1.0, 1.0],
        }
    )
    history = pl.DataFrame(
        {"target": ["K1", "K2"], "estimate": [0.5, 0.3], "loo_rank": [1, 2], "holdout_group": ["G1", "G1"]}
    )

    form = Formulation(
        trait="K_locus", max_valency=2, rankings=rankings, stability_metrics=metrics, permutation_history=history
    )

    assert isinstance(form.rankings, pl.DataFrame)
    assert isinstance(form.stability_metrics, pl.DataFrame)
    assert isinstance(form.permutation_history, pl.DataFrame)
    assert form.get_formulation() == ["K1", "K2"]

    with pytest.raises(dataclasses.FrozenInstanceError):
        form.max_valency = 5  # type: ignore


def test_formulation_assess_coverage():
    """Verify Formulation.assess_coverage adds binary coverage column to Polars DataFrame."""
    rankings = pl.DataFrame({"target": ["K1", "K2"], "estimate": [0.6, 0.3]})
    empty_df = pl.DataFrame()

    form = Formulation(
        trait="K_locus", max_valency=2, rankings=rankings, stability_metrics=empty_df, permutation_history=empty_df
    )

    isolates = pl.DataFrame({"sample_id": ["S1", "S2", "S3"], "K_locus": ["K1", "K3", "K2"]})

    cov_df = form.assess_coverage(isolates, col_name="Vaccine_Coverage")
    assert cov_df is not None
    assert isinstance(cov_df.data, pl.DataFrame)
    assert "Vaccine_Coverage" in cov_df.data.columns
    assert cov_df.data["Vaccine_Coverage"].to_list() == [True, False, True]


def test_formulation_evaluate_longevity():
    """Verify Formulation.evaluate_longevity assesses projected cases over time."""
    rankings = pl.DataFrame({"target": ["K1"], "estimate": [0.8]})
    empty_df = pl.DataFrame()
    form = Formulation(
        trait="K_locus", max_valency=1, rankings=rankings, stability_metrics=empty_df, permutation_history=empty_df
    )

    d1 = datetime.date(2023, 1, 1)
    d2 = datetime.date(2023, 2, 1)

    forecast_df = pl.DataFrame(
        {"date": [d1, d1, d2, d2], "target": ["K1", "K2", "K1", "K2"], "estimate": [10.0, 5.0, 15.0, 5.0]}
    )

    forecast = IncidenceEstimates(
        data=forecast_df,
        stratified_by=[],
        adjusted_for=None,
        trait="K_locus",
        freq="month",
        aggregation_type=AggregationType.TRAIT,
        model_results=empty_df,
    )

    longevity = form.evaluate_longevity(forecast)
    assert isinstance(longevity, pl.DataFrame)
    assert "total_cases" in longevity.columns
    assert "covered_cases" in longevity.columns
    assert "coverage_pct" in longevity.columns
    assert longevity["total_cases"].to_list() == [15.0, 20.0]
    assert longevity["covered_cases"].to_list() == [10.0, 15.0]


def test_posthoc_and_custom_formulation_designers():
    """Test PostHocFormulationDesigner and CustomFormulationDesigner with Polars inputs."""
    prev_data = pl.DataFrame(
        {
            "target": ["K1", "K2", "K3", "K1", "K2", "K3"],
            "estimate": [0.4, 0.3, 0.1, 0.5, 0.2, 0.1],
            "hospital": ["H1", "H1", "H1", "H2", "H2", "H2"],
        }
    )

    prev_res = PrevalenceEstimates(
        data=prev_data,
        stratified_by=["hospital"],
        adjusted_for=None,
        trait="K_locus",
        aggregation_type=AggregationType.COMPOSITIONAL,
        method="unpooled_wilson",
    )

    # PostHoc
    designer = PostHocFormulationDesigner(valency=2)
    designer.fit(prev_res, loo_col="hospital")
    assert designer.formulation_ is not None
    assert isinstance(designer.formulation_.rankings, pl.DataFrame)
    assert designer.formulation_.get_formulation() == ["K1", "K2"]

    # Custom
    custom_designer = CustomFormulationDesigner(targets=["K2", "K1"])
    custom_designer.fit(prev_res)
    assert custom_designer.formulation_ is not None
    assert custom_designer.formulation_.get_formulation() == ["K2", "K1"]


def test_app_utils_helpers():
    """Test ColMapper, _clean_ui_label, build_grouped_choices, format_metadata_ui."""
    cols = ["sample_id", "collection_date", "country_name", "genotype_ST", "other_col"]
    mapper = ColMapper(cols)
    assert mapper.guess("map_id") == "sample_id"
    assert mapper.guess("map_date") == "collection_date"
    assert mapper.guess("map_spatial") == "country_name"

    assert _clean_ui_label(f"{Domain.GENOTYPE.value}_K_locus") == "K Locus"
    assert _clean_ui_label("sample_id") == "Sample Id"

    choices = build_grouped_choices(cols)
    assert isinstance(choices, dict)

    meta_ui = format_metadata_ui({"stratified_by": ["region"], "trait": "K_locus"})
    assert isinstance(meta_ui, list)
    assert len(meta_ui) == 2


def test_end_to_end_m3_pipeline():
    """Verify end-to-end dataset -> estimator -> formulation pipeline with Polars."""
    df = pl.DataFrame(
        {"sample_id": ["S1", "S2", "S3", "S4"], "K_locus": ["K1", "K1", "K2", "K3"], "region": ["R1", "R1", "R2", "R2"]}
    )

    dataset = SeroEpiDataset(data=df, name="EndToEnd")

    # Aggregate prevalence (assume counts format for estimator input)
    agg_df = pl.DataFrame(
        {"target": ["K1", "K2", "K3"], "event": [2, 1, 1], "n": [4, 4, 4], "region": ["All", "All", "All"]}
    )
    agg_df.metadata = {
        "metric_meta": {
            "trait": "K_locus",
            "stratified_by": ["region"],
            "aggregation_type": AggregationType.COMPOSITIONAL,
        }
    }  # type: ignore

    # Run estimator
    est = UnpooledPrevalenceEstimator()
    res = est.calculate(agg_df)
    assert isinstance(res.data, pl.DataFrame)

    # Design formulation
    designer = PostHocFormulationDesigner(valency=2)
    designer.fit(res)
    form = designer.formulation_
    assert form is not None
    assert isinstance(form.rankings, pl.DataFrame)

    # Assess coverage on dataset
    covered = form.assess_coverage(dataset)
    assert isinstance(covered, pl.DataFrame)
    assert "Vaccine_Coverage" in covered.columns
