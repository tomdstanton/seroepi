"""Empirical edge case tests written by Challenger 1 for Milestone 2 Iteration 4.

Focus:
1. StrEnum members passed to domain accessors and estimators.
2. Custom DataFrames without metadata passed to accessors and estimators.
3. IEEE 754 NaN coordinates in GeoMixin.reverse_geocode.
4. Single vs list trait inputs in GenoMixin.has_any and GenoMixin.has_all.
"""

from enum import StrEnum
import polars as pl
import pytest
import numpy as np

from seroepi.traits import ChoiceEnum, CoreTrait, KpAmrTrait, KpSeroTrait, KpVirulenceTrait
from seroepi.domains.epi import EpiMixin
from seroepi.domains.geno import GenoMixin
from seroepi.domains.geo import GeoMixin
from seroepi.domains.qc import QcMixin
from seroepi.estimators.prevalence import (
    AlphaDiversityEstimator,
    BetaDiversityEstimator,
    UnpooledPrevalenceEstimator,
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
)
from seroepi.estimators.diversity import (
    AlphaDiversityEstimator,
    BetaDiversityEstimator,
)
from seroepi.constants import AggregationType, BayesianInferenceMethod


class CustomStrEnumTrait(ChoiceEnum):
    GENE_A = "geno_gene_A"
    GENE_B = "amr_gene_B"
    MARKER_C = "virulence_marker_C"
    LOCATION_X = "spatial_Country"


# Helper dummy class wrapping mixins for testing plain DataFrames
class EpiDummy(EpiMixin):
    def __init__(self, df: pl.DataFrame):
        self._df_obj = df

class GenoDummy(GenoMixin):
    def __init__(self, df: pl.DataFrame):
        self._df_obj = df

class GeoDummy(GeoMixin):
    def __init__(self, df: pl.DataFrame):
        self._df_obj = df

class QcDummy(QcMixin):
    def __init__(self, df: pl.DataFrame):
        self._df_obj = df


# ==============================================================================
# 1. StrEnum members passed to accessors & estimators
# ==============================================================================

def test_strenum_passed_to_accessors():
    """Verify StrEnum members work seamlessly across all domain mixins."""
    df = pl.DataFrame({
        "sample_id": ["S1", "S2"],
        "geno_K_locus": ["K1", "K2"],
        "amr_Bla_acquired": [True, False],
        "virulence_Yersiniabactin": [True, True],
        "spatial_Country": ["Australia", "Fiji"],
        "latitude": [-33.8, -17.7],
        "longitude": [151.2, 178.0],
        "qc_coverage": [30.0, 50.0],
    })

    geno_obj = GenoDummy(df)
    geo_obj = GeoDummy(df)

    # Test GenoMixin with StrEnum
    has_any_res = geno_obj.has_any(KpAmrTrait.BLA_ACQUIRED)
    assert isinstance(has_any_res, pl.Series)
    assert has_any_res.to_list() == [True, False]

    has_all_res = geno_obj.has_all(KpVirulenceTrait.YERSINIABACTIN)
    assert isinstance(has_all_res, pl.Series)
    assert has_all_res.to_list() == [True, True]

    sorted_df = geno_obj.sort_loci(KpSeroTrait.K_LOCUS)
    assert isinstance(sorted_df, pl.DataFrame)
    assert "geno_K_locus" in sorted_df.columns

    # Test GeoMixin with StrEnum
    imputed = geo_obj.standardize_and_impute(CoreTrait.COUNTRY)
    assert isinstance(imputed, pl.DataFrame)
    assert "latitude" in imputed.columns


def test_strenum_passed_to_estimators():
    """Verify StrEnum members can be passed as target/trait parameters in estimators."""
    df = pl.DataFrame({
        "spatial_Country": ["Australia", "Australia", "Fiji", "Fiji"],
        "target": ["K1", "K2", "K1", "K2"],
        "event": [10, 20, 5, 15],
        "n": [50, 50, 50, 50],
        "variant_count": [10, 20, 5, 15],
        "n_total": [50, 50, 50, 50],
    })

    # UnpooledPrevalenceEstimator with StrEnum trait in __init__
    unpooled = UnpooledPrevalenceEstimator(trait=KpSeroTrait.K_LOCUS)
    res_unpooled = unpooled.calculate(df)
    assert res_unpooled.trait == "geno_K_locus"

    # AlphaDiversityEstimator with StrEnum target in __init__
    alpha = AlphaDiversityEstimator(target=KpSeroTrait.K_LOCUS)
    res_alpha = alpha.calculate(df)
    assert res_alpha.trait == "geno_K_locus"

    # BetaDiversityEstimator with StrEnum target in __init__
    beta = BetaDiversityEstimator(target=KpSeroTrait.K_LOCUS)
    res_beta = beta.calculate(df)
    assert res_beta.trait == "geno_K_locus"

    # GLMPrevalenceEstimator fit and predict with DataFrame containing StrEnum target
    glm_prev = GLMPrevalenceEstimator()
    glm_prev.fit(df)
    res_glm = glm_prev.predict(df)
    assert "estimate" in res_glm.data.columns


# ==============================================================================
# 2. Custom DataFrames without metadata
# ==============================================================================

def test_custom_dataframe_without_metadata_accessors():
    """Test domain mixins when passed a plain DataFrame without metadata."""
    df_plain = pl.DataFrame({
        "amr_Bla_acquired": [True, False, True],
        "virulence_Yersiniabactin": [True, True, False],
        "latitude": [-33.8, np.nan, -37.8],
        "longitude": [151.2, 144.9, np.nan],
    })

    geno_plain = GenoDummy(df_plain)
    geo_plain = GeoDummy(df_plain)

    # Any check for AMR trait
    any_res = geno_plain.has_any("Bla_acquired", domain="amr")
    assert any_res.to_list() == [True, False, True]

    # All check with full trait StrEnum objects matching actual dataframe columns
    all_res = geno_plain.has_all([KpAmrTrait.BLA_ACQUIRED, KpVirulenceTrait.YERSINIABACTIN])
    assert all_res.to_list() == [True, False, False]

    # Impute on plain df without metadata attribute
    imputed_plain = geo_plain.standardize_and_impute()
    assert isinstance(imputed_plain, pl.DataFrame)
    # Output should still retain or construct a metadata dictionary getter seamlessly
    assert hasattr(imputed_plain, "metadata")
    assert isinstance(imputed_plain.metadata, dict)


def test_custom_dataframe_without_metadata_estimators():
    """Test estimators when passed a plain DataFrame without metadata."""
    df_plain = pl.DataFrame({
        "region": ["North", "South", "North", "South"],
        "target": ["blaKPC", "blaKPC", "blaNDM", "blaNDM"],
        "event": [5, 10, 2, 4],
        "n": [50, 50, 50, 50],
        "variant_count": [5, 10, 2, 4],
    })

    # UnpooledPrevalenceEstimator infers target from target column
    est = UnpooledPrevalenceEstimator()
    res = est.calculate(df_plain)
    assert res.trait == "target"
    assert len(res.data) == 4

    # AlphaDiversityEstimator with explicit target parameter
    alpha_est = AlphaDiversityEstimator(target="res_gene")
    res_alpha = alpha_est.calculate(df_plain)
    assert res_alpha.trait == "res_gene"


# ==============================================================================
# 3. IEEE 754 NaN coordinates in GeoMixin.reverse_geocode
# ==============================================================================

def test_ieee_754_nan_coordinates_reverse_geocode():
    """Test GeoMixin.reverse_geocode handles float NaN, None, and valid coords safely."""
    df_coords = pl.DataFrame({
        "sample_id": ["S1", "S2", "S3", "S4", "S5"],
        "latitude": [float("nan"), -33.8688, np.nan, None, -37.8136],
        "longitude": [float("nan"), 151.2093, np.nan, 144.9631, float("nan")],
    })

    geo_obj = GeoDummy(df_coords)
    res = geo_obj.reverse_geocode(target_spatial_name="Country")

    assert isinstance(res, pl.DataFrame)
    assert "spatial_Country" in res.columns
    assert "spatial_res_Country" in res.columns
    assert len(res) == 5

    # Check S2 (Sydney: -33.8688, 151.2093) reverse geocodes to Australia
    s2_country = res.filter(pl.col("sample_id") == "S2")["spatial_Country"][0]
    assert s2_country == "Australia"

    # Check S1, S3, S4, S5 (containing NaNs or Nones) do not crash shapely geometry construction
    s1_country = res.filter(pl.col("sample_id") == "S1")["spatial_Country"][0]
    assert s1_country is None


# ==============================================================================
# 4. Single vs list trait inputs in GenoMixin.has_any / has_all
# ==============================================================================

def test_single_vs_list_trait_inputs():
    """Verify single string, single StrEnum, list of strings, and list of StrEnums give identical results."""
    df = pl.DataFrame({
        "amr_Bla_acquired": [True, False, False],
        "amr_Col_acquired": [False, True, False],
        "virulence_Yersiniabactin": [True, True, False],
        "virulence_Aerobactin": [False, True, True],
    })

    geno = GenoDummy(df)

    # --- Single str vs List[str] in has_any ---
    single_str_any = geno.has_any("Bla_acquired", domain="amr")
    list_str_any = geno.has_any(["Bla_acquired"], domain="amr")
    assert single_str_any.to_list() == list_str_any.to_list() == [True, False, False]

    # --- Single StrEnum vs List[StrEnum] in has_any ---
    single_enum_any = geno.has_any(KpAmrTrait.BLA_ACQUIRED)
    list_enum_any = geno.has_any([KpAmrTrait.BLA_ACQUIRED])
    assert single_enum_any.to_list() == list_enum_any.to_list() == [True, False, False]

    # --- Multiple traits: list vs sequence ---
    multi_any = geno.has_any([KpAmrTrait.BLA_ACQUIRED, KpAmrTrait.COL_ACQUIRED])
    assert multi_any.to_list() == [True, True, False]

    # --- Single str vs List[str] in has_all ---
    single_str_all = geno.has_all("Yersiniabactin", domain="virulence")
    list_str_all = geno.has_all(["Yersiniabactin"], domain="virulence")
    assert single_str_all.to_list() == list_str_all.to_list() == [True, True, False]

    # --- Single StrEnum vs List[StrEnum] in has_all ---
    single_enum_all = geno.has_all(KpVirulenceTrait.YERSINIABACTIN)
    list_enum_all = geno.has_all([KpVirulenceTrait.YERSINIABACTIN])
    assert single_enum_all.to_list() == list_enum_all.to_list() == [True, True, False]

    # --- Multiple traits in has_all ---
    multi_all = geno.has_all([KpVirulenceTrait.YERSINIABACTIN, KpVirulenceTrait.AEROBACTIN])
    assert multi_all.to_list() == [False, True, False]
