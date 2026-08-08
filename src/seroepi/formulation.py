"""Module for abstracting a vaccine formulation using trait prevalence and stability."""

import copy
from abc import ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import polars as pl
from joblib import Parallel, delayed
from joblib import dump as joblib_dump
from joblib import load as joblib_load

from seroepi.estimators import BaseEstimator, IncidenceEstimates, ModelledMixin, PrevalenceEstimates


# Classes --------------------------------------------------------------------------------------------------------------
def _to_str(val: Any) -> str:  # noqa: ANN401
    """Converts val to a primitive str, extracting .value if val is an Enum."""
    if hasattr(val, "value"):
        return f"{val.value}"
    return f"{val}"


@dataclass(frozen=True, slots=True)
class CrossProtectionProfile:
    """Matrix defining immunological cross-reactivity between traits."""
    
    efficacy_matrix: dict[str | StrEnum, dict[str | StrEnum, float]]
    
    def apply(self, formulation_targets: list[str]) -> dict[str, float]:
        """Calculates the theoretical protection profile for a given formulation."""
        protected = {}
        for target in formulation_targets:
            target_str = str(target.value) if hasattr(target, "value") else str(target)
            protected[target_str] = max(protected.get(target_str, 0.0), 1.0)
            
            if target_str in self.efficacy_matrix:
                for cross_target, efficacy in self.efficacy_matrix[target_str].items():
                    cross_str = str(cross_target.value) if hasattr(cross_target, "value") else str(cross_target)
                    protected[cross_str] = max(protected.get(cross_str, 0.0), float(efficacy))
        return protected

@dataclass(frozen=True, slots=True)
class Formulation:
    """Represents a proposed vaccine formulation based on target prevalence and stability.

    This class holds the results of a formulation design process, including rankings,
    stability metrics from cross-validation, and permutation history.

    Attributes:
        trait: The trait type (e.g., 'K_locus').
        max_valency: The maximum number of targets in the formulation.
        rankings: A Polars DataFrame containing the definitive ranking of targets
            (target, estimate, baseline_rank, ...).
        stability_metrics: A Polars DataFrame containing metrics from LOO stability analysis
            (target, mean_loo_rank, rank_variance, probability_in_top_n).
        permutation_history: A Polars DataFrame containing the full history of ranks
            across all LOO permutations.
    """

    trait: str | StrEnum  # e.g., 'K_locus'
    max_valency: int  # e.g., 6 (for a hexavalent vaccine)
    rankings: pl.DataFrame
    stability_metrics: pl.DataFrame
    permutation_history: pl.DataFrame

    def __post_init__(self) -> None:  # noqa: D105
        if hasattr(self.trait, "value"):
            object.__setattr__(self, "trait", f"{self.trait.value}")
        else:
            object.__setattr__(self, "trait", f"{self.trait}")

    def save(self, filepath: str | Path) -> None:
        """Serializes the Formulation instance to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(self, path)

    @classmethod
    def load(cls: type["Formulation"], filepath: str | Path) -> "Formulation":
        """Loads a serialized Formulation from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No formulation found at {path}")
        formulation = joblib_load(path)
        if not isinstance(formulation, cls):
            raise TypeError(f"Type mismatch: Expected {cls.__name__}, got {type(formulation).__name__}.")
        return formulation

    def get_formulation(self) -> list[str]:
        """Returns the top N targets for the proposed vaccine.

        Returns:
            A list of target names.
        """
        if "target" in self.rankings.columns and len(self.rankings) > 0:
            return [_to_str(t) for t in self.rankings.head(self.max_valency)["target"].to_list()]
        return []

    def assess_coverage(
        self, 
        df: Any, 
        col_name: str = "Vaccine_Coverage",
        cross_protection: CrossProtectionProfile | None = None
    ) -> "VaccineCoverageEstimates":
        """Appends a boolean/float column indicating whether each row's trait variant
        is covered by the targets in this formulation.
        """
        df_in = df.data if hasattr(df, "data") else df
        trait_col = str(self.trait)

        if trait_col not in df_in.columns:
            raise KeyError(f"Active dataset is missing the formulated trait: {trait_col}")

        targets = self.get_formulation()
        
        if cross_protection is None:
            # Binary logic
            covered_df = df_in.with_columns(**{col_name: pl.col(trait_col).is_in(targets).cast(pl.Float64)})
        else:
            # Cross-protection logic
            protection_dict = cross_protection.apply(targets)
            
            mapping_df = pl.DataFrame({
                trait_col: list(protection_dict.keys()),
                col_name: list(protection_dict.values())
            }).with_columns(pl.col(trait_col).cast(df_in[trait_col].dtype))
            
            covered_df = df_in.join(mapping_df, on=trait_col, how="left").with_columns(
                pl.col(col_name).fill_null(0.0)
            )

        overall_cov = float(covered_df[col_name].mean() or 0.0)
        
        breakdown = covered_df.group_by(trait_col).agg(
            pl.col(col_name).mean().alias("protection_efficacy"),
            pl.len().alias("count")
        ).with_columns(
            contribution=(pl.col("protection_efficacy") * pl.col("count") / len(covered_df))
        ).sort("contribution", descending=True)
        
        from seroepi.estimators.base import VaccineCoverageEstimates
        return VaccineCoverageEstimates(
            data=covered_df,
            stratified_by=[],
            adjusted_for=None,
            trait=trait_col,
            aggregation_type="formulation",
            formulation=self,
            overall_coverage=overall_cov,
            antigen_breakdown=breakdown
        )

    def evaluate_longevity(self, forecast: IncidenceEstimates) -> pl.DataFrame:
        """Evaluates the formulation against a time-series incidence forecast to
        determine its historical and projected longevity.

        Returns a Polars DataFrame tracking the absolute case burden and the percentage
        of that burden covered by this formulation over time.
        """  # noqa: D205
        df = forecast.data

        targets = self.get_formulation()

        total_cases = df.group_by("date").agg(pl.col("estimate").sum().alias("total_cases"))
        covered_df = df.filter(pl.col("target").is_in(targets))
        covered_cases = covered_df.group_by("date").agg(pl.col("estimate").sum().alias("covered_cases"))

        longevity = total_cases.join(covered_cases, on="date", how="left").fill_null(0)
        longevity = longevity.with_columns(
            coverage_pct=pl.when(pl.col("total_cases") > 0)
            .then((pl.col("covered_cases") / pl.col("total_cases")) * 100.0)
            .otherwise(0.0)
        ).sort("date")

        return longevity


class BaseFormulationDesigner(ModelledMixin, ABC):
    """Abstract base class for formulation designers."""

    def __init__(self, valency: int = 6, n_jobs: int = -1):  # noqa: ANN204, D107
        self.valency = valency
        self.n_jobs = n_jobs
        self.formulation_: Formulation | None = None

    def fit(  # noqa: D102
        self,
        *args,  # noqa: ANN002
        progress_callback: Callable[[int, int], None] | None = None,
        **kwargs,  # noqa: ANN002, ANN003
    ) -> "BaseFormulationDesigner":
        raise NotImplementedError

    def predict(self, df: Any) -> pl.DataFrame:  # noqa: ANN401
        """Uses the fitted formulation to predict vaccine coverage on a given DataFrame.
        Returns only the rows that are covered by the designed formulation.
        """  # noqa: D205
        self.check_is_fitted()

        if self.formulation_ is None:
            raise RuntimeError("Designer has not been fitted yet.")
        coverage_estimates = self.formulation_.assess_coverage(df, col_name="_tmp_coverage")
        return coverage_estimates.data.filter(pl.col("_tmp_coverage") > 0.0).drop(["_tmp_coverage"])


class PostHocFormulationDesigner(BaseFormulationDesigner):
    """Fast formulation design using post-hoc estimation."""

    def fit(  # type: ignore # noqa: ANN401, D102
        self,
        result: PrevalenceEstimates,
        loo_col: str | StrEnum | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> "PostHocFormulationDesigner":
        raw_df = result.data
        trait_name = result.trait
        loo_col_str = str(loo_col) if loo_col is not None else None

        # 1. Baseline
        baseline = _extract_ranks(raw_df, "baseline_rank")

        if loo_col_str and loo_col_str in raw_df.columns:
            # 2. Permutations
            total_estimates = raw_df.group_by("target").agg(pl.col("estimate").sum().alias("total_estimate"))
            group_target_estimates = raw_df.group_by([loo_col_str, "target"]).agg(
                pl.col("estimate").sum().alias("group_estimate")
            )

            unique_groups = raw_df[loo_col_str].unique().to_list()
            total_groups = len(unique_groups)
            loo_records = []
            for i, group in enumerate(unique_groups, 1):
                group_sub = group_target_estimates.filter(pl.col(loo_col_str) == group)
                if len(group_sub) > 0:
                    loo_estimates = (
                        total_estimates.join(group_sub, on="target", how="left")
                        .with_columns(estimate=(pl.col("total_estimate") - pl.col("group_estimate").fill_null(0)))
                        .select(["target", "estimate"])
                    )
                else:
                    loo_estimates = total_estimates.rename({"total_estimate": "estimate"})

                loo_ranks = loo_estimates.sort("estimate", descending=True).with_columns(
                    loo_rank=pl.int_range(1, pl.len() + 1), holdout_group=pl.lit(str(group))
                )
                loo_records.append(loo_ranks)

                if progress_callback:
                    progress_callback(i, total_groups)

            history = (
                pl.concat(loo_records)
                if loo_records
                else pl.DataFrame(
                    schema={"target": pl.Utf8, "estimate": pl.Float64, "loo_rank": pl.Int64, "holdout_group": pl.Utf8}
                )
            )
        else:
            history = pl.DataFrame(
                schema={"target": pl.Utf8, "estimate": pl.Float64, "loo_rank": pl.Int64, "holdout_group": pl.Utf8}
            )
            if progress_callback:
                progress_callback(1, 1)

        # 3. Compile
        self.formulation_ = _compile_stability_metrics(baseline, history, trait_name, self.valency)
        self.is_fitted_ = True
        return self


class CustomFormulationDesigner(BaseFormulationDesigner):
    """Formulation designer that generates a custom formulation from a user-defined list of targets."""

    def __init__(self, targets: Sequence[str | StrEnum]):  # noqa: ANN204, D107
        self.targets = [_to_str(t) for t in targets]
        super().__init__(valency=len(targets), n_jobs=1)

    def fit(  # type: ignore # noqa: ANN401, D102
        self, baseline_result: PrevalenceEstimates, progress_callback: Callable[[int, int], None] | None = None
    ) -> "CustomFormulationDesigner":
        trait_name = _to_str(baseline_result.trait)
        raw_df = baseline_result.data

        # 1. Calculate the true baseline prevalence for everything
        baseline = (
            raw_df.group_by("target")
            .agg(pl.col("estimate").sum())
            .with_columns(pl.col("target").cast(pl.Utf8))
            .sort("estimate", descending=True)
        )

        # 2. Filter and reorder the baseline to match the user's custom list exactly
        custom_df = pl.DataFrame(
            {"target": self.targets, "custom_rank": list(range(1, len(self.targets) + 1))}
        ).with_columns(pl.col("target").cast(pl.Utf8))
        custom_rankings = (
            custom_df.join(baseline, on="target", how="left")
            .fill_null(0)
            .sort("custom_rank")
            .with_columns(baseline_rank=pl.col("custom_rank"))
            .drop(["custom_rank"])
        )

        empty_stability = pl.DataFrame(
            schema={
                "target": pl.Utf8,
                "mean_loo_rank": pl.Float64,
                "rank_variance": pl.Float64,
                "probability_in_top_n": pl.Float64,
            }
        )
        empty_history = pl.DataFrame(
            schema={"target": pl.Utf8, "estimate": pl.Float64, "loo_rank": pl.Int64, "holdout_group": pl.Utf8}
        )

        self.formulation_ = Formulation(
            trait=trait_name,
            max_valency=len(self.targets),
            rankings=custom_rankings,
            stability_metrics=empty_stability,
            permutation_history=empty_history,
        )
        self.is_fitted_ = True
        return self


class CVFormulationDesigner(BaseFormulationDesigner):
    """Rigorous formulation design using true Leave-One-Out (LOO) cross-validation."""

    def fit(  # type: ignore # noqa: ANN401, D102
        self,
        estimator: BaseEstimator[Any],
        agg_df: Any,  # noqa: ANN401
        loo_col: str | StrEnum | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> "CVFormulationDesigner":
        df = agg_df.data if hasattr(agg_df, "data") else agg_df
        baseline_result = _clone_estimator(estimator).calculate(agg_df)
        trait_name = baseline_result.trait
        baseline = _extract_ranks(baseline_result.data, "baseline_rank")
        loo_col_str = str(loo_col) if loo_col is not None else None

        if loo_col_str and loo_col_str in df.columns:
            groups = [str(g) for g in df[loo_col_str].unique().to_list()]
            total_groups = len(groups)

            jobs = (delayed(_run_cv_fold)(estimator, agg_df, loo_col_str, group) for group in groups)

            loo_records = []
            try:
                with Parallel(n_jobs=self.n_jobs, return_as="generator") as parallel:
                    for i, result in enumerate(parallel(jobs), 1):
                        loo_records.append(result)
                        if progress_callback:
                            progress_callback(i, total_groups)
            except TypeError:
                with Parallel(n_jobs=self.n_jobs) as parallel:
                    loo_records = parallel(jobs)
                    if progress_callback:
                        progress_callback(total_groups, total_groups)

            history = (
                pl.concat(loo_records)
                if loo_records
                else pl.DataFrame(
                    schema={"target": pl.Utf8, "estimate": pl.Float64, "loo_rank": pl.Int64, "holdout_group": pl.Utf8}
                )
            )
        else:
            history = pl.DataFrame(
                schema={"target": pl.Utf8, "estimate": pl.Float64, "loo_rank": pl.Int64, "holdout_group": pl.Utf8}
            )
            if progress_callback:
                progress_callback(1, 1)

        self.formulation_ = _compile_stability_metrics(baseline, history, trait_name, self.valency)
        self.is_fitted_ = True
        return self


# Functions ------------------------------------------------------------------------------------------------------------
def _extract_ranks(df: pl.DataFrame, rank_col_name: str) -> pl.DataFrame:
    """Consistently groups, sums, sorts, and ranks traits."""
    ranks = df.group_by("target").agg(pl.col("estimate").sum()).sort("estimate", descending=True)
    return ranks.with_columns(**{rank_col_name: pl.int_range(1, pl.len() + 1)})


def _run_cv_fold(estimator: BaseEstimator[Any], agg_df: Any, loo_col: str | StrEnum, group: str) -> pl.DataFrame:  # noqa: ANN401
    """Helper function to execute a single CV fold in parallel."""
    df = agg_df.data if hasattr(agg_df, "data") else agg_df
    loo_col_str = str(loo_col)
    holdout_df = df.filter(pl.col(loo_col_str).cast(pl.Utf8) != str(group))
    if hasattr(agg_df, "data"):
        from seroepi.dataset import SeroEpiDataset

        holdout_input = SeroEpiDataset(data=holdout_df, name=agg_df.name, metadata=dict(agg_df.metadata))
    else:
        holdout_input = holdout_df

    loo_result = _clone_estimator(estimator).calculate(holdout_input)
    ranks = _extract_ranks(loo_result.data, "loo_rank")
    return ranks.with_columns(holdout_group=pl.lit(str(group)))


def _compile_stability_metrics(
    baseline: pl.DataFrame, history: pl.DataFrame, trait_name: str | StrEnum, valency: int
) -> Formulation:
    """Compiles the final variance and probability matrix for the formulation."""
    trait_name_str = _to_str(trait_name)
    if history.is_empty():
        stability = pl.DataFrame(
            schema={
                "target": pl.Utf8,
                "mean_loo_rank": pl.Float64,
                "rank_variance": pl.Float64,
                "probability_in_top_n": pl.Float64,
            }
        )
    else:
        stability = history.group_by("target").agg(
            [
                pl.col("loo_rank").mean().alias("mean_loo_rank"),
                pl.col("loo_rank").var().fill_null(0.0).alias("rank_variance"),
                (pl.col("loo_rank") <= valency).mean().alias("probability_in_top_n"),
            ]
        )

        # Join to baseline to maintain baseline target ordering
        stability = baseline.select(["target"]).join(stability, on="target", how="left").fill_null(0.0)

    return Formulation(
        trait=trait_name_str,
        max_valency=valency,
        rankings=baseline,
        stability_metrics=stability,
        permutation_history=history,
    )


def _clone_estimator(estimator: BaseEstimator[Any]) -> BaseEstimator[Any]:
    if hasattr(estimator, "get_params"):
        # Create a new instance of the exact same class using its params
        return type(estimator)(**estimator.get_params())  # type: ignore
    return copy.deepcopy(estimator)
