"""
Unit and integration tests for Milestone 2 domains package (src/seroepi/domains/).
"""

import polars as pl

from seroepi.constants import AggregationType, PlotType
from seroepi.dataset import SeroEpiDataset
from seroepi.domains import (
    AlphaDiversityPlotter,
    BaseDomainMixin,
    BasePlotter,
    BetaHeatmapPlotter,
    ChoroplethPlotter,
    CompositionBarPlotter,
    CompositionHeatmapPlotter,
    CumulativeCoveragePlotter,
    EpicurvePlotter,
    EpiMixin,
    ForestPlotter,
    GenoMixin,
    GeoMixin,
    LongevityPlotter,
    LongitudinalPrevalencePlotter,
    NetworkPlotter,
    QcMixin,
    SpatialSurfacePlotter,
    StabilityBumpPlotter,
    render_plot,
)


def test_domains_mixins_slotted():
    """Verify that all domain mixin classes are slotted with __slots__ = ()."""
    for cls in [BaseDomainMixin, GeoMixin, EpiMixin, GenoMixin, QcMixin]:
        assert hasattr(cls, "__slots__")
        assert cls.__slots__ == ()


def test_dataset_inherits_mixins():
    """Verify SeroEpiDataset inherits directly from domain mixins."""
    assert issubclass(SeroEpiDataset, GeoMixin)
    assert issubclass(SeroEpiDataset, EpiMixin)
    assert issubclass(SeroEpiDataset, GenoMixin)
    assert issubclass(SeroEpiDataset, QcMixin)

    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [1.0, 2.0],
            "longitude": [10.0, 20.0],
            "spatial_Country": ["France", "Germany"],
            "geno_ST": ["ST1", "ST2"],
            "amr_gene": ["bla1", "bla2"],
            "qc_N50": [50000, 60000],
        }
    )
    ds = SeroEpiDataset(data=df, name="Test Domains")

    # Direct calling on dataset
    assert hasattr(ds, "standardize_and_impute")
    assert hasattr(ds, "aggregate_prevalence")
    assert hasattr(ds, "has_any")
    assert hasattr(ds, "filter_assemblies")

    # Backward compatibility properties
    assert ds.geo is ds
    assert ds.epi is ds
    assert ds.geno is ds
    assert ds.qc is ds


def test_domains_plotters_reexported():
    """Verify all 13 concrete plotters, BasePlotter, and render_plot are exported."""
    plotters = [
        ChoroplethPlotter,
        SpatialSurfacePlotter,
        CompositionBarPlotter,
        CompositionHeatmapPlotter,
        ForestPlotter,
        EpicurvePlotter,
        LongitudinalPrevalencePlotter,
        CumulativeCoveragePlotter,
        StabilityBumpPlotter,
        LongevityPlotter,
        AlphaDiversityPlotter,
        BetaHeatmapPlotter,
        NetworkPlotter,
    ]
    for plotter in plotters:
        assert issubclass(plotter, BasePlotter)


def test_render_plot_registration():
    """Verify render_plot routes correctly across domain plotters."""
    from seroepi.estimators.base import PrevalenceEstimates

    est_df = pl.DataFrame({"target": ["KL1"], "estimate": [0.5], "lower": [0.4], "upper": [0.6]})
    pe = PrevalenceEstimates(
        data=est_df,
        stratified_by=[],
        adjusted_for=None,
        trait="K_locus",
        aggregation_type=AggregationType.COMPOSITIONAL,
        method="wilson",
    )

    fig = render_plot(pe, PlotType.COMPOSITION_BAR)
    assert fig is not None


def test_geno_has_any_strenum_and_full_column_name():
    """Verify has_any works seamlessly with StrEnum, full column name, and unprefixed name without double prefixing."""
    from seroepi.traits import KpAmrTrait

    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "amr_Bla_acquired": [True, False],
        }
    )
    ds = SeroEpiDataset(data=df, name="Test Has Any")

    # Pass StrEnum member (value: "amr_Bla_acquired")
    res1 = ds.has_any([KpAmrTrait.BLA_ACQUIRED])
    assert res1.to_list() == [True, False]

    # Pass full column name string
    res2 = ds.has_any(["amr_Bla_acquired"])
    assert res2.to_list() == [True, False]

    # Pass unprefixed string
    res3 = ds.has_any(["Bla_acquired"])
    assert res3.to_list() == [True, False]


def test_geno_has_all_strenum_and_full_column_name():
    """Verify has_all works seamlessly with StrEnum, full column name, and unprefixed name without double prefixing."""
    from seroepi.traits import KpVirulenceTrait

    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "virulence_Yersiniabactin": [True, True],
            "virulence_Aerobactin": [True, False],
        }
    )
    ds = SeroEpiDataset(data=df, name="Test Has All")

    # Pass StrEnum members
    res1 = ds.has_all([KpVirulenceTrait.YERSINIABACTIN, KpVirulenceTrait.AEROBACTIN])
    assert res1.to_list() == [True, False]

    # Pass full column names
    res2 = ds.has_all(["virulence_Yersiniabactin", "virulence_Aerobactin"])
    assert res2.to_list() == [True, False]

    # Pass unprefixed strings
    res3 = ds.has_all(["Yersiniabactin", "Aerobactin"])
    assert res3.to_list() == [True, False]


def test_geo_reverse_geocode_strenum():
    """Verify reverse_geocode accepts StrEnum for target_spatial_name and avoids double prefixing."""
    from seroepi.traits import CoreTrait

    df = pl.DataFrame(
        {
            "sample_id": ["S1"],
            "latitude": [48.8566],
            "longitude": [2.3522],
        }
    )
    ds = SeroEpiDataset(data=df, name="Test Reverse Geocode")

    res_ds = ds.reverse_geocode(target_spatial_name=CoreTrait.COUNTRY)
    # Should create "spatial_Country", not "spatial_spatial_Country"
    assert "spatial_Country" in res_ds.data.columns
    assert "spatial_spatial_Country" not in res_ds.data.columns


def test_epi_aggregate_methods_strenum():
    """Verify EpiMixin aggregate methods accept StrEnum for stratify_by, trait_col, cluster_col."""
    from seroepi.traits import CoreTrait, KpAmrTrait, KpSeroTrait

    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "spatial_Country": ["France", "France", "Germany"],
            "temporal_Collection_Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "geno_ST": ["ST11", "ST11", "ST258"],
            "amr_Bla_acquired": [True, False, True],
        }
    ).with_columns(pl.col("temporal_Collection_Date").str.to_date())

    ds = SeroEpiDataset(data=df, name="Test Epi Aggregates")

    # Prevalence
    prev_df = ds.aggregate_prevalence(stratify_by=[CoreTrait.COUNTRY], trait_col=KpAmrTrait.BLA_ACQUIRED)
    assert prev_df is not None
    assert "spatial_Country" in prev_df.columns
    assert "target" in prev_df.columns

    # Diversity
    div_df = ds.aggregate_diversity(stratify_by=[KpSeroTrait.ST], trait_col=KpAmrTrait.BLA_ACQUIRED)
    assert div_df is not None
    assert "geno_ST" in div_df.columns

    # Incidence
    inc_df = ds.aggregate_incidence(stratify_by=[CoreTrait.COUNTRY], trait_col=KpAmrTrait.BLA_ACQUIRED)
    assert inc_df is not None
    assert "spatial_Country" in inc_df.columns


def test_geo_reverse_geocode_ocean_unmatched_points():
    """Verify reverse_geocode handles ocean/unmatched coordinates and existing columns without ComputeError."""
    df_ocean = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [0.0, 0.0],
            "longitude": [0.0, 0.0],
        }
    )
    ds_ocean = SeroEpiDataset(data=df_ocean, name="Ocean Test")
    res_ocean = ds_ocean.reverse_geocode(target_spatial_name="Country")
    assert "spatial_Country" in res_ocean.data.columns
    assert res_ocean.data["spatial_Country"].to_list() == [None, None]

    df_existing = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "latitude": [-33.8688, 0.0],
            "longitude": [151.2093, 0.0],
            "spatial_Country": ["Existing_Land", "Existing_Ocean"],
        }
    )
    ds_existing = SeroEpiDataset(data=df_existing, name="Existing Col Test")
    res_existing = ds_existing.reverse_geocode(target_spatial_name="Country")
    assert res_existing.data["spatial_Country"].to_list() == ["Australia", "Existing_Ocean"]


def test_geno_has_any_has_all_unprefixed_columns():
    """Verify has_any and has_all resolve unprefixed column names when passed StrEnum members."""
    from seroepi.traits import KpAmrTrait

    df_unprefixed = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            "Bla_acquired": [True, False, True],
        }
    )

    res_any = df_unprefixed.geno.has_any([KpAmrTrait.BLA_ACQUIRED], domain="amr")
    assert res_any.to_list() == [True, False, True]

    res_all = df_unprefixed.geno.has_all([KpAmrTrait.BLA_ACQUIRED], domain="amr")
    assert res_all.to_list() == [True, False, True]


def test_epi_resolve_temporal_col_unprefixed():
    """Verify _resolve_temporal_col resolves unprefixed temporal column names with StrEnum inputs."""
    from seroepi.traits import CoreTrait

    df_unprefixed = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            "Collection_Date": ["2023-01-01", "2023-01-02"],
        }
    )

    resolved_enum = df_unprefixed.epi._resolve_temporal_col(CoreTrait.COLLECTION_DATE)
    assert resolved_enum == "Collection_Date"

    resolved_none = df_unprefixed.epi._resolve_temporal_col(None)
    assert resolved_none == "Collection_Date"


def test_epi_accessor_metadata_preservation():
    """Verify aggregate methods set .metadata dictionary on returned DataFrames."""
    from seroepi.constants import TemporalResolution
    from seroepi.traits import CoreTrait, KpAmrTrait, KpSeroTrait

    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2"],
            CoreTrait.COUNTRY.value: ["Australia", "Singapore"],
            KpSeroTrait.K_LOCUS.value: ["KL1", "KL2"],
            KpAmrTrait.BLA_ACQUIRED.value: ["+", "-"],
            CoreTrait.COLLECTION_DATE.value: [
                pl.Series(["2023-01-01"]).str.to_date()[0],
                pl.Series(["2023-01-02"]).str.to_date()[0],
            ],
        }
    )

    # Prevalence
    agg_prev = df.epi.aggregate_prevalence(stratify_by=[CoreTrait.COUNTRY], trait_col=KpAmrTrait.BLA_ACQUIRED)
    assert hasattr(agg_prev, "metadata")
    assert isinstance(agg_prev.metadata, dict)
    assert agg_prev.metadata["metric_meta"]["trait"] == KpAmrTrait.BLA_ACQUIRED.value

    # Diversity
    agg_div = df.epi.aggregate_diversity(stratify_by=[CoreTrait.COUNTRY, KpSeroTrait.K_LOCUS])
    assert hasattr(agg_div, "metadata")
    assert isinstance(agg_div.metadata, dict)
    assert agg_div.metadata["metric_meta"]["trait"] == KpSeroTrait.K_LOCUS.value

    # Incidence
    agg_inc = df.epi.aggregate_incidence(stratify_by=[KpSeroTrait.K_LOCUS], freq=TemporalResolution.MONTH)
    assert hasattr(agg_inc, "metadata")
    assert isinstance(agg_inc.metadata, dict)
    assert agg_inc.metadata["metric_meta"]["freq"] == "1mo"


def test_epi_accessor_to_estimator_pipeline_strenum():
    """Verify end-to-end accessor -> estimator pipeline preserves trait contract and primitive strings with StrEnum."""
    from seroepi.constants import TemporalResolution
    from seroepi.estimators import (
        AlphaDiversityEstimator,
        BetaDiversityEstimator,
        GLMIncidenceEstimator,
        UnpooledPrevalenceEstimator,
    )
    from seroepi.traits import CoreTrait, KpAmrTrait, KpSeroTrait

    df = pl.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            CoreTrait.COUNTRY.value: ["Australia", "Australia", "Singapore", "Singapore"],
            KpSeroTrait.K_LOCUS.value: ["KL1", "KL2", "KL1", "KL2"],
            KpAmrTrait.BLA_ACQUIRED.value: ["+", "-", "+", "-"],
            CoreTrait.COLLECTION_DATE.value: [
                pl.Series(["2023-01-01"]).str.to_date()[0],
                pl.Series(["2023-01-02"]).str.to_date()[0],
                pl.Series(["2023-02-01"]).str.to_date()[0],
                pl.Series(["2023-02-02"]).str.to_date()[0],
            ],
        }
    )

    # 1. Prevalence pipeline
    agg_prev = df.epi.aggregate_prevalence(stratify_by=[CoreTrait.COUNTRY], trait_col=KpAmrTrait.BLA_ACQUIRED)
    res_prev = UnpooledPrevalenceEstimator().calculate(agg_prev)
    assert type(res_prev.trait) is str
    assert res_prev.trait == KpAmrTrait.BLA_ACQUIRED.value

    # 2. Diversity pipeline
    agg_div = df.epi.aggregate_diversity(stratify_by=[CoreTrait.COUNTRY, KpSeroTrait.K_LOCUS])
    res_alpha = AlphaDiversityEstimator().calculate(agg_div)
    assert type(res_alpha.trait) is str
    assert res_alpha.trait == KpSeroTrait.K_LOCUS.value
    for s in res_alpha.stratified_by:
        assert type(s) is str

    res_beta = BetaDiversityEstimator().calculate(agg_div)
    assert type(res_beta.trait) is str
    assert res_beta.trait == KpSeroTrait.K_LOCUS.value
    for s in res_beta.stratified_by:
        assert type(s) is str

    # 3. Incidence pipeline
    agg_inc = df.epi.aggregate_incidence(stratify_by=[KpSeroTrait.K_LOCUS], freq=TemporalResolution.MONTH)
    res_inc = GLMIncidenceEstimator().fit(agg_inc).predict(agg_inc)
    assert type(res_inc.trait) is str
    assert res_inc.trait == KpSeroTrait.K_LOCUS.value
    assert type(res_inc.freq) is str
