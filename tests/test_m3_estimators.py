import dataclasses
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from seroepi.constants import AggregationType
from seroepi.dataset import SeroEpiDataset
from seroepi.estimators.base import (
    AlphaDiversityEstimates,
    BetaDiversityEstimates,
    Estimates,
    IncidenceEstimates,
    PrevalenceEstimates,
)
from seroepi.estimators.prevalence import (
    BayesianPrevalenceEstimator,
    GLMPrevalenceEstimator,
    SpatialPrevalenceEstimator,
    UnpooledPrevalenceEstimator,
)
from seroepi.estimators.diversity import (
    AlphaDiversityEstimator,
    BetaDiversityEstimator,
)
from seroepi.estimators.incidence import (
    GLMIncidenceEstimator,
)


def test_estimates_dataclasses_frozen_slotted():
    """Verify Estimates result classes are frozen slotted dataclasses holding pl.DataFrame."""
    classes = [
        Estimates,
        PrevalenceEstimates,
        AlphaDiversityEstimates,
        BetaDiversityEstimates,
        IncidenceEstimates,
    ]

    sample_df = pl.DataFrame({"target": ["K1", "K2"], "estimate": [0.6, 0.4]})

    for cls in classes:
        assert dataclasses.is_dataclass(cls), f"{cls.__name__} must be a dataclass"
        assert hasattr(cls, "__slots__"), f"{cls.__name__} must have __slots__"

    prev = PrevalenceEstimates(
        data=sample_df,
        stratified_by=["region"],
        adjusted_for="none",
        trait="K_locus",
        aggregation_type=AggregationType.COMPOSITIONAL,
        method="unpooled_wilson",
    )

    assert isinstance(prev.data, pl.DataFrame)
    assert prev.trait == "K_locus"
    assert prev.method == "unpooled_wilson"

    with pytest.raises(dataclasses.FrozenInstanceError):
        prev.method = "other"  # type: ignore

    with pytest.raises(AttributeError):
        prev.extra_attr = 123  # type: ignore

    inc_model_results = pl.DataFrame({"IRR": [1.25], "p_value": [0.01]})
    inc = IncidenceEstimates(
        data=sample_df,
        stratified_by=["year"],
        adjusted_for=None,
        trait="K_locus",
        aggregation_type=AggregationType.TRAIT,
        freq="month",
        model_results=inc_model_results,
    )
    assert isinstance(inc.data, pl.DataFrame)
    assert isinstance(inc.model_results, pl.DataFrame)


def test_unpooled_prevalence_estimator_polars():
    """Verify UnpooledPrevalenceEstimator operates on pl.DataFrame and SeroEpiDataset."""
    agg_data = pl.DataFrame(
        {"target": ["K1", "K2", "K3"], "event": [10, 20, 5], "n": [100, 100, 50], "region": ["North", "North", "South"]}
    )

    dataset = SeroEpiDataset(
        data=agg_data,
        name="Test Prev",
        metadata={
            "metric_meta": {"stratified_by": ["region"], "trait": "K_locus", "aggregation_type": AggregationType.TRAIT}
        },
    )

    estimator = UnpooledPrevalenceEstimator(method="wilson", alpha=0.05)

    # Test on SeroEpiDataset
    res1 = estimator.calculate(dataset)
    assert isinstance(res1, PrevalenceEstimates)
    assert isinstance(res1.data, pl.DataFrame)
    assert "estimate" in res1.data.columns
    assert "lower" in res1.data.columns
    assert "upper" in res1.data.columns

    estimates = res1.data["estimate"].to_numpy()
    lowers = res1.data["lower"].to_numpy()
    uppers = res1.data["upper"].to_numpy()

    assert np.all(lowers <= estimates)
    assert np.all(estimates <= uppers)
    assert np.all((0 <= lowers) & (uppers <= 1))

    # Test directly on pl.DataFrame
    res2 = estimator.calculate(agg_data)
    assert isinstance(res2, PrevalenceEstimates)
    assert isinstance(res2.data, pl.DataFrame)


def test_alpha_diversity_estimator_polars():
    """Verify AlphaDiversityEstimator calculates diversity metrics on Polars DataFrames."""
    div_data = pl.DataFrame(
        {
            "target": ["K1", "K2", "K3", "K1", "K2"],
            "variant_count": [50, 30, 20, 10, 90],
            "hospital": ["H1", "H1", "H1", "H2", "H2"],
        }
    )

    meta_df = div_data.with_columns()
    meta_df.metadata = {"metric_meta": {"stratified_by": ["hospital"], "trait": "K_locus"}}  # type: ignore

    estimator = AlphaDiversityEstimator(target="K_locus", metrics=["shannon", "simpson", "richness"])
    res = estimator.calculate(meta_df)

    assert isinstance(res, AlphaDiversityEstimates)
    assert isinstance(res.data, pl.DataFrame)
    assert "hospital" in res.data.columns
    assert "shannon" in res.data.columns
    assert "simpson" in res.data.columns
    assert "richness" in res.data.columns
    assert len(res.data) == 2


def test_beta_diversity_estimator_polars():
    """Verify BetaDiversityEstimator calculates distance matrix on Polars DataFrames."""
    div_data = pl.DataFrame(
        {"target": ["K1", "K2", "K1", "K2"], "variant_count": [10, 5, 2, 20], "hospital": ["H1", "H1", "H2", "H2"]}
    )

    meta_df = div_data.with_columns()
    meta_df.metadata = {"metric_meta": {"stratified_by": ["hospital"], "trait": "K_locus"}}  # type: ignore

    estimator = BetaDiversityEstimator(target="K_locus", metric="braycurtis")
    res = estimator.calculate(meta_df)

    assert isinstance(res, BetaDiversityEstimates)
    assert isinstance(res.data, pl.DataFrame)
    assert res.data.shape == (2, 2)
    assert res.metric == "braycurtis"


def test_glm_prevalence_estimator_polars():
    """Verify Frequentist GLMPrevalenceEstimator fits and predicts on Polars DataFrames."""
    df = pl.DataFrame(
        {
            "target": ["K1", "K2", "K1", "K2"],
            "event": [12, 5, 8, 15],
            "n": [50, 50, 50, 50],
            "region": ["A", "A", "B", "B"],
        }
    )

    estimator = GLMPrevalenceEstimator()
    res = estimator.calculate(df)

    assert isinstance(res, PrevalenceEstimates)
    assert isinstance(res.data, pl.DataFrame)
    assert "estimate" in res.data.columns
    assert "lower" in res.data.columns
    assert "upper" in res.data.columns
    assert len(res.data) == 4


def test_glm_incidence_estimator_polars():
    """Verify GLMIncidenceEstimator fits negative binomial model on Polars DataFrames."""
    dates = [
        "2023-01-01",
        "2023-02-01",
        "2023-03-01",
        "2023-04-01",
        "2023-01-01",
        "2023-02-01",
        "2023-03-01",
        "2023-04-01",
    ]
    df = pl.DataFrame(
        {
            "date": [pl.Series([d]).str.to_date()[0] for d in dates],
            "target": ["K1", "K1", "K1", "K1", "K2", "K2", "K2", "K2"],
            "variant_count": [5, 8, 12, 18, 2, 4, 3, 5],
            "total_sequenced": [100, 110, 105, 120, 100, 110, 105, 120],
        }
    )
    df.metadata = {"metric_meta": {"trait": "K_locus", "freq": "month", "stratified_by": []}}  # type: ignore

    estimator = GLMIncidenceEstimator(use_relative_incidence=True, forecast_horizon=2)
    res = estimator.calculate(df)

    assert isinstance(res, IncidenceEstimates)
    assert isinstance(res.data, pl.DataFrame)
    assert isinstance(res.model_results, pl.DataFrame)
    assert "IRR" in res.model_results.columns
    assert len(res.data) > len(df)  # Future horizon rows added


def test_zero_pandas_in_src():
    """Verify that no import pandas or import pandera statements remain anywhere in src/seroepi."""
    src_dir = Path(__file__).parent.parent / "src" / "seroepi"
    forbidden = ["import pandas", "import pandera", "from pandas", "from pandera", "pd."]

    violations = []
    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for line_no, line in enumerate(content.splitlines(), 1):
            for pattern in forbidden:
                if pattern in line and not line.strip().startswith("#"):
                    violations.append(f"{py_file.relative_to(src_dir)}:{line_no}: {line.strip()}")

    assert not violations, "Forbidden legacy Pandas/Pandera references found in src/:\n" + "\n".join(violations)


def test_estimators_strenum_parameters():
    """Verify that estimators natively accept StrEnum column parameters and return string attributes."""
    from seroepi.estimators.diversity import AlphaDiversityEstimator, BetaDiversityEstimator
    from seroepi.estimators.prevalence import GLMPrevalenceEstimator, SpatialPrevalenceEstimator
    from seroepi.traits import CoreTrait, KpSeroTrait

    # Alpha diversity with StrEnum target
    estimator_alpha = AlphaDiversityEstimator(target=KpSeroTrait.K_LOCUS)
    assert estimator_alpha.target == "geno_K_locus"
    assert isinstance(estimator_alpha.target, str)

    # Beta diversity with StrEnum target
    estimator_beta = BetaDiversityEstimator(target=KpSeroTrait.O_LOCUS)
    assert estimator_beta.target == "geno_O_locus"
    assert isinstance(estimator_beta.target, str)

    # Spatial estimator with CoreTrait lat/lon
    estimator_spatial = SpatialPrevalenceEstimator(
        lat_col=CoreTrait.LATITUDE, lon_col=CoreTrait.LONGITUDE, target_event="event", target_n="n"
    )
    assert estimator_spatial.lat_col == "latitude"
    assert estimator_spatial.lon_col == "longitude"
    assert isinstance(estimator_spatial.lat_col, str)
    assert isinstance(estimator_spatial.lon_col, str)

    # GLM prevalence estimator with StrEnum target event/n
    estimator_glm = GLMPrevalenceEstimator(target_event="event", target_n="n")
    assert estimator_glm.target_event == "event"
    assert estimator_glm.target_n == "n"


def test_estimator_metadata_fallback_and_svi_predict():
    """Verify estimators handle missing metadata via column inference and SpatialPrevalenceEstimator SVI predict works cleanly."""
    from seroepi.constants import BayesianInferenceMethod
    from seroepi.estimators.diversity import BetaDiversityEstimator
    from seroepi.estimators.prevalence import UnpooledPrevalenceEstimator
    from seroepi.estimators.incidence import GLMIncidenceEstimator
    from seroepi.estimators.prevalence import SpatialPrevalenceEstimator

    # 1. UnpooledPrevalenceEstimator with missing metadata (custom column name preservation)
    custom_df = pl.DataFrame(
        {
            "my_country": ["C1", "C2"],
            "target": ["gene_A", "gene_A"],
            "event": [5, 10],
            "n": [50, 50],
        }
    )
    res_unpooled = UnpooledPrevalenceEstimator().calculate(custom_df)
    assert type(res_unpooled.trait) is str
    assert res_unpooled.trait in ("gene_A", "target")

    # 2. BetaDiversityEstimator with missing metadata (inferred strata)
    div_df_no_meta = pl.DataFrame(
        {
            "my_country": ["C1", "C1", "C2", "C2"],
            "target": ["KL1", "KL2", "KL1", "KL2"],
            "variant_count": [10, 5, 2, 20],
            "n_total": [15, 15, 22, 22],
        }
    )
    res_beta = BetaDiversityEstimator(target="KL1").calculate(div_df_no_meta)
    assert isinstance(res_beta, BetaDiversityEstimates)
    assert res_beta.metric == "braycurtis"
    # 3. GLMIncidenceEstimator with missing metadata
    inc_df_no_meta = pl.DataFrame(
        {
            "date": [pl.Series(["2023-01-01"]).str.to_date()[0], pl.Series(["2023-02-01"]).str.to_date()[0]],
            "target": ["KL1", "KL1"],
            "variant_count": [5, 10],
            "total_sequenced": [100, 100],
        }
    )
    res_inc = GLMIncidenceEstimator().fit(inc_df_no_meta).predict(inc_df_no_meta)
    assert type(res_inc.trait) is str
    assert res_inc.trait == "KL1"

    # 4. SpatialPrevalenceEstimator SVI predict
    spatial_df = pl.DataFrame(
        {
            "latitude": [1.0, 2.0],
            "longitude": [10.0, 20.0],
            "event": [5, 2],
            "n": [10, 10],
            "target": ["amr_Bla_acquired", "amr_Bla_acquired"],
        }
    )
    spatial_est = SpatialPrevalenceEstimator(
        lat_col="latitude",
        lon_col="longitude",
        method=BayesianInferenceMethod.SVI,
        svi_steps=10,
    )
    res_spatial = spatial_est.fit(spatial_df).predict(spatial_df)
    assert type(res_spatial.trait) is str
    assert len(res_spatial.data) == 2


def test_force_of_infection_r0_hit():
    """Verify calculate_r0 and calculate_hit correctly approximate values from FOI."""
    from seroepi.estimators.base import ForceOfInfectionEstimates
    
    foi_res = ForceOfInfectionEstimates(
        data=pl.DataFrame(),
        stratified_by=[],
        adjusted_for=None,
        trait="serology",
        aggregation_type="trait",
        lambda_foi=0.05,
        rho_recovery=None,
        age_strata=["0-10", "11-20"]
    )
    
    r0 = foi_res.calculate_r0(70.0)
    assert abs(r0 - 4.5) < 1e-5
    
    hit = foi_res.calculate_hit(70.0)
    assert abs(hit - (1.0 - 1.0/4.5)) < 1e-5

def test_reproduction_number_estimator_mock():
    """Verify ReproductionNumberEstimator properly estimates Re > 1 for growth and Re < 1 for decline."""
    from seroepi.estimators.incidence import ReproductionNumberEstimator
    from datetime import date, timedelta
    
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(20)]
    cases = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
    
    df = pl.DataFrame({
        "date": dates,
        "cases": cases
    })
    
    estimator = ReproductionNumberEstimator(
        time_column="date",
        case_column="cases",
        generation_time_mean=5.0,
        generation_time_std=2.0,
        num_warmup=100,
        num_samples=100
    )
    
    res = estimator.calculate(df)
    
    growth_re = res.data["Re_mean"][5:9].mean()
    assert growth_re > 1.0
    
    decline_re = res.data["Re_mean"][15:19].mean()
    assert decline_re < 1.0
