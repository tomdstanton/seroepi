"""Module for estimating trait prevalence, diversity and incidence among isolates."""

from .base import (
    AlphaDiversityEstimates,
    BaseEstimator,
    BetaDiversityEstimates,
    Estimates,
    ForceOfInfectionEstimates,
    IncidenceEstimates,
    PrevalenceEstimates,
    SeropositivityEstimates,
    VaccineCoverageEstimates,
)
from .diversity import (
    AlphaDiversityEstimator,
    BetaDiversityEstimator,
)
from .prevalence import UnpooledPrevalenceEstimator

__all__ = (
    "AlphaDiversityEstimates",
    "AlphaDiversityEstimator",
    "BaseEstimator",
    "BetaDiversityEstimates",
    "BetaDiversityEstimator",
    "Estimates",
    "ForceOfInfectionEstimates",
    "IncidenceEstimates",
    "PrevalenceEstimates",
    "SeropositivityEstimates",
    "UnpooledPrevalenceEstimator",
    "VaccineCoverageEstimates",
)

try:
    from .mixins import ModelledMixin, BayesianMixin
    from .prevalence import (
        BayesianPrevalenceEstimator,
        GLMPrevalenceEstimator,
        SpatialPrevalenceEstimator,
    )
    from .incidence import (
        BayesianIncidenceEstimator,
        GLMIncidenceEstimator,
    )
    from .serology import (
        SerocatalyticEstimator,
        TiterClassificationEstimator,
    )

    __all__ += (
        "BayesianIncidenceEstimator",
        "BayesianMixin",
        "BayesianPrevalenceEstimator",
        "GLMIncidenceEstimator",
        "GLMPrevalenceEstimator",
        "ModelledMixin",
        "SerocatalyticEstimator",
        "SpatialPrevalenceEstimator",
        "TiterClassificationEstimator",
    )
except ImportError:
    pass
