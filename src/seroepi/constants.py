"""Enums for non-user-facing API constants - mostly to help with the app"""  # noqa: D415

from enum import StrEnum, auto
from typing import Any

# Enums ----------------------------------------------------------------------------------------------------------------
from enum import StrEnum, auto



class PlotType(StrEnum):  # noqa: D101
    FOREST = auto()
    EPICURVE = auto()
    CHOROPLETH = auto()
    COMPOSITION_BAR = auto()
    COMPOSITION_HEATMAP = auto()
    LONGITUDINAL_PREVALENCE = auto()
    CUMULATIVE_COVERAGE = auto()
    STABILITY_BUMP = auto()
    SPATIAL_SURFACE = auto()
    ALPHA_DIVERSITY = auto()
    BETA_HEATMAP = auto()
    NETWORK = auto()
    LONGEVITY = auto()
    PYRAMID = auto()


class HoldoutStrategy(StrEnum):  # noqa: D101
    COUNTRY = auto()
    TRANSMISSION_CLUSTER = auto()
    STUDY = auto()


class FormulationStrategy(StrEnum):
    """The theoretical approach used to design a vaccine formulation."""
    POSTHOC = "posthoc"
    CROSS_VALIDATED = "cv"
    CUSTOM = "custom"
    GREEDY_COVERAGE = "greedy"

    @classmethod
    def ui_labels(cls) -> dict[Any, str]:
        return {
            cls.POSTHOC.value: "Post-Hoc (Fast)",
            cls.CROSS_VALIDATED.value: "Cross-Validated (Rigorous)",
            cls.CUSTOM.value: "Custom Override",
            cls.GREEDY_COVERAGE.value: "Greedy Coverage",
        }
        
    @classmethod
    def designer_ui_labels(cls) -> dict[str, str]:
        return {
            cls.POSTHOC.value: "Post-Hoc (Fast)",
            cls.CROSS_VALIDATED.value: "Cross-Validated (Rigorous)",
        }


class EpidemiologicalDomain(StrEnum):  # noqa: D101
    PREVALENCE = auto()
    DIVERSITY = auto()
    INCIDENCE = auto()
    SEROLOGY = auto()
    FORCE_OF_INFECTION = auto()


class AggregationType(StrEnum):  # noqa: D101
    TRAIT = auto()
    COMPOSITIONAL = auto()

    @classmethod
    def ui_labels(cls) -> dict[str, str]:
        return {
            cls.COMPOSITIONAL.value: "Compositional",
            cls.TRAIT.value: "Trait",
        }


class ConfidenceIntervalMethod(StrEnum):
    WILSON = "wilson"
    WALD = "wald"
    AGRESTI_COULL = "agresti_coull"
    CLOPPER_PEARSON = "clopper_pearson"
    JEFFREYS = "jeffreys"
    
    @classmethod
    def ui_labels(cls) -> dict[str, str]:
        return {
            cls.WILSON.value: "Wilson Score",
            cls.WALD.value: "Wald",
            cls.AGRESTI_COULL.value: "Agresti-Coull",
            cls.CLOPPER_PEARSON.value: "Clopper-Pearson (Exact)",
            cls.JEFFREYS.value: "Jeffreys",
        }


class AlphaDiversityMetric(StrEnum):
    SHANNON = "shannon"
    SIMPSON = "simpson"
    RICHNESS = "richness"
    
    @classmethod
    def ui_labels(cls) -> dict[str, str]:
        return {
            cls.SHANNON.value: "Shannon Index",
            cls.SIMPSON.value: "Simpson Index",
            cls.RICHNESS.value: "Richness",
        }


class BetaDiversityMetric(StrEnum):
    BRAYCURTIS = "braycurtis"
    JACCARD = "jaccard"
    EUCLIDEAN = "euclidean"
    CITYBLOCK = "cityblock"
    
    @classmethod
    def ui_labels(cls) -> dict[str, str]:
        return {
            cls.BRAYCURTIS.value: "Bray-Curtis",
            cls.JACCARD.value: "Jaccard",
            cls.EUCLIDEAN.value: "Euclidean",
            cls.CITYBLOCK.value: "Manhattan (Cityblock)",
        }


class Domain(StrEnum):  # noqa: D101
    AMR = auto()
    VIRULENCE = auto()
    QC = auto()
    SPATIAL = auto()
    TEMPORAL = auto()
    SPATIAL_RES = auto()
    TEMPORAL_RES = auto()
    GENOTYPE = "geno"
    PHENOTYPE = "pheno"
    CLUSTER = auto()


class DistanceFlavour(StrEnum):  # noqa: D101
    PATHOGENWATCH = auto()
    SKA2 = auto()
    NEWICK = auto()


class GenotypeFlavour(StrEnum):  # noqa: D101
    PATHOGENWATCH_KLEBORATE = "pathogenwatch-kleborate"


class EstimatorType(StrEnum):  # noqa: D101
    UNPOOLED = auto()
    GLM = auto()
    BAYESIAN = auto()
    SPATIAL = auto()

    @classmethod
    def ui_labels(cls) -> dict[Any, str]:  # noqa: D102
        return {
            cls.UNPOOLED: "Frequentist (Unpooled CIs)",
            cls.GLM: "Frequentist (GLM)",
            cls.BAYESIAN: "Bayesian (Hierarchical)",
            cls.SPATIAL: "Bayesian (Spatial GP)",
        }

    @property
    def class_name(self) -> str:  # noqa: D102
        return {
            self.UNPOOLED: "UnpooledPrevalenceEstimator",
            self.BAYESIAN: "BayesianPrevalenceEstimator",
            self.GLM: "GLMPrevalenceEstimator",
            self.SPATIAL: "SpatialPrevalenceEstimator",
        }[self]


class BayesianInferenceMethod(StrEnum):  # noqa: D101
    MCMC = auto()
    SVI = auto()


class TemporalResolution(StrEnum):  # noqa: D101
    YEAR = auto()
    MONTH = auto()
    WEEK = auto()
    DAY = auto()
    UNKNOWN = auto()

    @classmethod
    def _missing_(cls, value):  # noqa: ANN001, ANN206
        return cls.UNKNOWN

    @property
    def polars_interval(self) -> str:
        """Returns Polars duration/interval string for dt.truncate or group_by_dynamic."""  # noqa: D421
        return {
            self.YEAR: "1y",
            self.MONTH: "1mo",
            self.WEEK: "1w",
            self.DAY: "1d",
            self.UNKNOWN: "",
        }[self]


class SpatialResolution(StrEnum):  # noqa: D101
    GLOBAL = auto()
    CONTINENT = auto()
    REGION = auto()
    COUNTRY = auto()
    CITY = auto()
    HOSPITAL = auto()
    EXACT = auto()
    UNKNOWN = auto()

    @classmethod
    def _missing_(cls, value):  # noqa: ANN001, ANN206
        return cls.UNKNOWN


class DistanceEpidemiologicalDomain(StrEnum):
    """Enumeration of supported metric types for pairwise comparisons.

    These metric types define how to interpret the numerical values
    in a distance/similarity matrix.

    Attributes:
        ABSOLUTE_DISTANCE: An absolute distance measure (e.g., 5 SNPs).
        RELATIVE_DISTANCE: A relative distance measure typically between 0.0 and 1.0 (e.g., 0.05 Hamming).
        ABSOLUTE_SIMILARITY: An absolute similarity measure (e.g., 95 shared nucleotides).
        RELATIVE_SIMILARITY: A relative similarity measure typically between 0.0 and 1.0 (e.g., 0.95 Jaccard).
    """

    ABSOLUTE_DISTANCE = auto()  # e.g., 5 SNPs
    RELATIVE_DISTANCE = auto()  # e.g., 0.05 Hamming
    ABSOLUTE_SIMILARITY = auto()  # e.g., 95 shared nucleotides
    RELATIVE_SIMILARITY = auto()  # e.g., 0.95 Jaccard

    @classmethod
    def _missing_(cls, value):  # noqa: ANN001, ANN206
        if isinstance(value, str):
            if v := cls.__members__.get(value.upper().replace(" ", "_").replace("-", "_")):
                return v
        return cls.ABSOLUTE_DISTANCE
