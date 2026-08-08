from abc import ABC, abstractmethod  # noqa: D100
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Generic, TypeVar
from warnings import warn

import polars as pl

from seroepi.constants import AggregationType, Domain
from seroepi.domains.base import get_dataframe_metadata

# Classes --------------------------------------------------------------------------------------------------------------
# Define a TypeVar that represents ANY dataclass you might invent in the future
T_Result = TypeVar("T_Result")


# The BaseEstimator inherits from Generic[T_Result]
class BaseEstimator(ABC, Generic[T_Result]):
    """The universal contract for all seroepi statistical models.

    All prevalence, diversity, and incidence estimators must inherit from this
    class and implement the `calculate` method or be wrapped by a higher level
    estimator base.
    """

    def _extract_strata(
        self,
        agg_df: Any,  # noqa: ANN401
        exclude_cols: Sequence[str | StrEnum] | None = None,  # noqa: ANN401
    ) -> tuple[list[str], dict[str, Any]]:
        """Extracts stratification columns and metadata from an aggregated DataFrame or SeroEpiDataset.

        Args:
            agg_df: The aggregated DataFrame or SeroEpiDataset.
            exclude_cols: Column names to exclude from stratification.

        Returns:
            A tuple containing the list of strata columns and the metadata dictionary.
        """
        exclude_str = [f"{c.value}" if hasattr(c, "value") else f"{c}" for c in exclude_cols] if exclude_cols else []

        df = agg_df.data if hasattr(agg_df, "data") else agg_df

        raw_meta = (
            getattr(df, "metadata", getattr(df, "meta", None))
            or getattr(agg_df, "metadata", getattr(agg_df, "meta", None))
            or get_dataframe_metadata(agg_df)
            or get_dataframe_metadata(df)
            or {}
        )
        meta: dict[str, Any] = raw_meta.get("metric_meta", raw_meta) if isinstance(raw_meta, dict) else {}

        inferred_strata = [
            f"{col.value}" if hasattr(col, "value") else f"{col}" for col in df.columns if col not in exclude_str
        ]
        stratified_by = meta.get("stratified_by", inferred_strata)
        stratified_by_str = [f"{s.value}" if hasattr(s, "value") else f"{s}" for s in stratified_by]
        self._validate_suitability(df, stratified_by_str)
        return stratified_by_str, meta

    @staticmethod
    def _validate_suitability(df: pl.DataFrame, strata: Sequence[str | StrEnum]):  # noqa: ANN205
        """Validates that the stratification columns are suitable for modeling.

        Checks for common statistical traps such as continuous variables being
        used as strata or over-stratification by unique identifiers.

        Args:
            df: The DataFrame to check.
            strata: The list of stratification columns.

        Raises:
            ValueError: If a stratum is a continuous float.
        """
        for col in strata:
            col_str = str(col)
            if col_str not in df.columns:
                continue
            dtype = df[col_str].dtype
            # 1. The Continuous Float Trap
            if dtype in (pl.Float32, pl.Float64) or dtype.is_float():
                raise ValueError(
                    f"Strata column '{col_str}' is a continuous float. "
                    "Prevalence estimators require discrete categorical groups. "
                    "Please bin this variable (e.g., using pl.cut()) before aggregating."
                )

            # 2. The Raw Datetime Trap (Checks dtype, standard 'date' name, OR the 'temporal_' prefix)
            if (
                dtype in (pl.Date, pl.Datetime)
                or dtype.is_temporal()
                or "date" in col_str.lower()
                or col_str.startswith(f"{Domain.TEMPORAL.value}_")
            ):
                warn(
                    f"You are stratifying on a raw date column '{col_str}'. "
                    "This will calculate prevalence for every single day. "
                    "Consider bucketing by month/year using .dt.to_period('M') first.",
                    UserWarning,
                )

        # 3. The Primary Key / Over-Stratification Trap
        if "n" in df.columns:
            avg_group_size = float(df["n"].mean() or 0.0) if len(df) > 0 else 0.0  # type: ignore
            if len(df) > 1 and avg_group_size < 1.5:
                warn(
                    f"The average group size is {avg_group_size:.2f}. "
                    "Did you accidentally stratify by a unique identifier (like 'sample_id')? "
                    "This will cause your models to overfit or crash.",
                    UserWarning,
                )

    @abstractmethod
    def calculate(self, df: Any) -> T_Result:  # noqa: ANN401
        """Executes the estimator's logic on the provided DataFrame.

        Args:
            df: The input DataFrame (usually aggregated).

        Returns:
            An Estimates object (e.g., PrevalenceEstimates).
        """
        pass


# Result dataclasses ---------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Estimates:
    """Base container for statistical estimates.

    Attributes:
        data: A Polars DataFrame containing the estimates and original strata.
        stratified_by: List of columns used for stratification.
        adjusted_for: Column name used for cluster adjustment, if any.
        trait: The trait variable for which estimates were calculated.
    """

    data: pl.DataFrame
    stratified_by: list[str]
    adjusted_for: str | None
    trait: str  # e.g., "blaKPC" or "Serotype"
    aggregation_type: AggregationType | str  # "trait" or "compositional"


@dataclass(frozen=True)
class PrevalenceEstimates(Estimates):
    """Container for prevalence results.

    Attributes:
        method: The statistical method used (e.g., 'bayesian_mcmc').
    """

    method: str


@dataclass(frozen=True)
class AlphaDiversityEstimates(Estimates):
    """Container for Alpha Diversity results.

    Attributes:
        metrics: List of diversity metrics calculated (e.g., ['shannon', 'simpson']).
    """

    metrics: list[str]


@dataclass(frozen=True)
class BetaDiversityEstimates(Estimates):
    """Container for Beta Diversity results (distance matrices).

    Attributes:
        metric: The distance metric used (e.g., 'braycurtis').
    """

    metric: str


@dataclass(frozen=True)
class IncidenceEstimates(Estimates):
    """Container for time-series incidence results.

    Attributes:
        freq: The time resolution used (e.g., TemporalResolution.MONTH.value).
        model_results: A Polars DataFrame containing regression outputs (IRR, CIs, P-values).
        method: The statistical method used (e.g., 'bsts_forecast_svi').
    """

    freq: str  # The time resolution (e.g., TemporalResolution.MONTH.value)
    model_results: pl.DataFrame  # The regression outputs (IRR, CIs, P-values)
    method: str = "incidence"


@dataclass(frozen=True)
class VaccineCoverageEstimates(Estimates):
    """Container for vaccine coverage evaluation results.
    
    Attributes:
        formulation: The Formulation object evaluated.
        overall_coverage: The scalar percentage of the population covered.
        antigen_breakdown: A DataFrame detailing the coverage contribution of each specific antigen.
    """
    formulation: Any  # Cannot type hint Formulation directly to avoid circular import
    overall_coverage: float
    antigen_breakdown: pl.DataFrame


@dataclass(frozen=True)
class ReproductionEstimates(Estimates):
    """Container for time-varying reproduction numbers."""
    time_column: str
    generation_time_mean: float


@dataclass(frozen=True)
class SeropositivityEstimates(Estimates):
    """Container for GMM-based titer classification results.
    
    Attributes:
        cutoff: The calculated continuous threshold distinguishing negative from positive.
        negative_component_mean: Mean of the unexposed distribution.
        positive_component_mean: Mean of the exposed distribution.
        model_fit_metrics: DataFrame containing AIC/BIC or ELBO for the GMM.
    """
    cutoff: float
    negative_component_mean: float
    positive_component_mean: float
    model_fit_metrics: pl.DataFrame


@dataclass(frozen=True)
class ForceOfInfectionEstimates(Estimates):
    """Container for Serocatalytic model results.
    
    Attributes:
        lambda_foi: Force of Infection (rate of seroconversion).
        rho_recovery: Rate of seroreversion (if using a reversible model).
        age_strata: The age bucketing used for the calculation.
    """
    lambda_foi: float
    rho_recovery: float | None
    age_strata: list[str]

    def calculate_r0(self, life_expectancy: float) -> float:
        """Approximates R0 from FOI: R0 ≈ 1 + (lambda * L)"""
        return 1.0 + (self.lambda_foi * life_expectancy)
        
    def calculate_hit(self, life_expectancy: float) -> float:
        """Calculates Herd Immunity Threshold: 1 - (1 / R0)"""
        r0 = self.calculate_r0(life_expectancy)
        return max(0.0, 1.0 - (1.0 / r0))


class BaseStatefulEstimator(BaseEstimator[T_Result]):
    """Abstract base class for stateful estimators (e.g., Bayesian, GLM, GMM).
    
    Automatically handles prediction formatting and metadata resolution.
    Subclasses should implement `_generate_predictions` to perform core math.
    """
    
    @abstractmethod
    def _generate_predictions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Subclasses must implement the core prediction logic here."""
        pass
        
    def predict(self, agg_df: Any) -> T_Result:
        """Universal predict wrapper that formats metadata automatically."""
        # Note: self.check_is_fitted() must be called by subclasses in predict
        # or we assume ModelledMixin is also inherited by the subclass.
        if hasattr(self, "check_is_fitted"):
            self.check_is_fitted()
            
        df = agg_df.data if hasattr(agg_df, "data") else agg_df
        
        # 1. Run subclass math
        result_df = self._generate_predictions(df)
        
        # 2. Safely resolve metadata
        meta = getattr(self, "meta_", {})
        strata = getattr(self, "strata_", [])
        
        # 3. Create the T_Result dataclass (we extract the actual class using self.__orig_bases__)
        # Fallback to Estimates if we can't reflect it, or let subclasses override _result_class
        result_class = getattr(self, "_result_class", Estimates)
        
        return result_class(
            data=result_df,
            stratified_by=[str(s) for s in strata],
            adjusted_for=str(meta.get("adjusted_for", "None")),
            trait=str(meta.get("trait", "unknown")),
            aggregation_type=str(meta.get("aggregation_type", "unknown"))
        )

