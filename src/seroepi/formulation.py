"""
Module for abstracting a vaccine _formulation using trait prevalence and stability.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from abc import ABC
from pathlib import Path
from typing import Optional, Callable, Union, Type
from joblib import Parallel, delayed, dump as joblib_dump, load as joblib_load
from seroepi.estimators import BaseEstimator, ModelledMixin


# Classes --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Formulation:
    """
    Represents a proposed vaccine _formulation based on target prevalence and stability.

    This class holds the results of a _formulation design process, including rankings,
    stability metrics from cross-validation, and permutation history.

    Attributes:
        trait: The trait type (e.g., 'K_locus').
        max_valency: The maximum number of targets in the _formulation.
        rankings: A DataFrame containing the definitive ranking of targets
            (Trait, Rank, Prevalence, Cumulative Coverage).
        stability_metrics: A DataFrame containing metrics from LOO stability analysis
            (e.g., Mean LOO Rank, Rank Variance, Probability in Top N).
        permutation_history: A DataFrame containing the full history of ranks
            across all LOO permutations.

    Examples:
        >>> import pandas as pd
        >>> from seroepi.formulation import Formulation
        >>> rankings = pd.DataFrame({'trait': ['K1', 'K2'], 'estimate': [0.5, 0.3]})
        >>> _formulation = Formulation(
        ...     trait='K_locus',
        ...     max_valency=2,
        ...     rankings=rankings,
        ...     stability_metrics=pd.DataFrame(),
        ...     permutation_history=pd.DataFrame()
        ... )
        >>> print(_formulation.get_formulation())
        ['K1', 'K2']
    """
    trait: str  # e.g., 'K_locus'
    max_valency: int  # e.g., 6 (for a hexavalent vaccine)
    # The definitive ranking matrix (Antigen, Rank, Prevalence, Cumulative Coverage)
    rankings: pd.DataFrame
    # Leave-One-Out (LOO) Stability Data
    # Rows = Antigens, Cols = 'Rank Variance', 'Prob in Top N', etc.
    stability_metrics: pd.DataFrame
    # Full history of ranks across all LOO permutations (used for plotting)
    permutation_history: pd.DataFrame

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes the Formulation instance to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(self, path)

    @classmethod
    def load(cls: Type['Formulation'], filepath: Union[str, Path]) -> 'Formulation':
        """Loads a serialized Formulation from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No formulation found at {path}")
        formulation = joblib_load(path)
        if not isinstance(formulation, cls):
            raise TypeError(f"Type mismatch: Expected {cls.__name__}, got {type(formulation).__name__}.")
        return formulation

    def get_formulation(self) -> list[str]:
        """
        Returns the top N targets for the proposed vaccine.

        Returns:
            A list of target names.
        """
        return self.rankings.head(self.max_valency)['target'].tolist()

    def assess_coverage(self, df: pd.DataFrame, col_name: str = 'Vaccine_Coverage') -> pd.DataFrame:
        """
        Appends a boolean column indicating whether each row's trait variant 
        is covered by the targets in this formulation.
        """
        if self.trait not in df.columns:
            raise KeyError(f"Active dataset is missing the formulated trait: {self.trait}")
            
        targets = self.get_formulation()
        
        # Assign binary coverage trait
        coverage_series = pd.Series(False, index=df.index)
        valid_mask = df[self.trait].notna()
        coverage_series.loc[valid_mask] = df.loc[valid_mask, self.trait].isin(targets)
        return df.assign(**{col_name: coverage_series})

    def evaluate_longevity(self, forecast: 'IncidenceEstimates') -> pd.DataFrame:
        """
        Evaluates the formulation against a time-series incidence forecast to
        determine its historical and projected longevity.

        Returns a DataFrame tracking the absolute case burden and the percentage
        of that burden covered by this formulation over time.
        """
        df = forecast.data.copy()

        # 1. Extract the specific antigens in this vaccine
        targets = self.get_formulation()

        # 2. Calculate the total expected cases across ALL circulating strains per time step
        total_cases = df.groupby('date')['estimate'].sum().rename('total_cases')

        # 3. Calculate the expected cases caused ONLY by strains in our vaccine
        covered_df = df[df['target'].isin(targets)]
        covered_cases = covered_df.groupby('date')['estimate'].sum().rename('covered_cases')

        # 4. Merge and calculate the moving coverage percentage
        longevity = pd.merge(total_cases, covered_cases, left_index=True, right_index=True, how='left').fillna(0)

        # Safe division to prevent NaNs if a specific month has 0 total cases projected
        longevity['coverage_pct'] = np.where(
            longevity['total_cases'] > 0,
            (longevity['covered_cases'] / longevity['total_cases']) * 100,
            0.0
        )

        return longevity.reset_index()


class BaseFormulationDesigner(ModelledMixin, ABC):
    """
    Abstract base class for _formulation designers.

    Designers are responsible for evaluating prevalence estimates and generating
    a vaccine _formulation with stability metrics.

    Attributes:
        valency: The target valency (number of targets) for the vaccine.
        n_jobs: The number of CPU cores to use for processing (-1 for all).
        formulation_: The resulting Formulation object after fitting.
    """

    def __init__(self, valency: int = 6, n_jobs: int = -1):
        """
        Initializes the designer.

        Args:
            valency: The target valency. Defaults to 6.
            n_jobs: Number of concurrent workers. Defaults to -1 (all available).
        """
        self.valency = valency
        self.n_jobs = n_jobs
        self.formulation_: Optional[Formulation] = None

    def fit(self, *args, progress_callback: Optional[Callable[[int, int], None]] = None, **kwargs) -> 'BaseFormulationDesigner':
        """Calculates the _formulation and stores it in self.formulation_"""
        pass
        
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Uses the fitted _formulation to predict vaccine coverage on a given DataFrame.
        Returns only the rows that are covered by the designed _formulation.
        """
        self.check_is_fitted()
        
        df_covered = self.formulation_.assess_coverage(df, col_name='_tmp_coverage')
        return df_covered[df_covered['_tmp_coverage']].drop(columns=['_tmp_coverage']).copy()


class PostHocFormulationDesigner(BaseFormulationDesigner):
    """
    Fast formulation design using post-hoc estimation.

    This method computes stability exactly for Frequentist estimates where retraining
    is not required for Leave-One-Out (LOO) analysis. For complex modelled estimates 
    (e.g., Bayesian, Spatial), it can be used as a fast, linear approximation 
    of stability (ignoring non-linear shrinkage and spatial correlation).
    """

    def fit(self, result: 'PrevalenceEstimates', loo_col: Optional[str] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None) -> 'PostHocFormulationDesigner':
        """
        Evaluates prevalence results to design a _formulation.

        Args:
            result: The prevalence estimates to evaluate.
            loo_col: The column name to use for Leave-One-Out cross-validation.

        Returns:
            The fitted designer instance.
        """
        raw_df = result.data
        trait_name = result.trait

        # 1. Baseline
        baseline = _extract_ranks(raw_df, 'baseline_rank')

        if loo_col and loo_col in raw_df.columns:
            # 2. Permutations (Vectorized O(N) Subtraction)
            # Pre-calculate global sums and the individual group sums
            total_estimates = raw_df.groupby('target')['estimate'].sum()
            group_target_estimates = raw_df.groupby([loo_col, 'target'])['estimate'].sum().unstack(fill_value=0)
    
            unique_groups = raw_df[loo_col].unique()
            total_groups = len(unique_groups)
            loo_records = []
            for i, group in enumerate(unique_groups, 1):
                # Subtract the holdout group's contribution from the global total
                if group in group_target_estimates.index:
                    loo_estimates = total_estimates - group_target_estimates.loc[group]
                else:
                    loo_estimates = total_estimates.copy()
    
                loo_ranks = loo_estimates.sort_values(ascending=False).reset_index()
                loo_ranks.columns = ['target', 'estimate']
                loo_ranks['loo_rank'] = loo_ranks.index + 1
                loo_ranks['holdout_group'] = group
                loo_records.append(loo_ranks)
                
                if progress_callback:
                    progress_callback(i, total_groups)
    
            history = pd.concat(loo_records, ignore_index=True)
        else:
            history = pd.DataFrame(columns=['target', 'estimate', 'loo_rank', 'holdout_group'])
            if progress_callback: progress_callback(1, 1)

        # 3. Compile
        self.formulation_ = _compile_stability_metrics(baseline, history, trait_name, self.valency)
        self.is_fitted_ = True
        return self


class CustomFormulationDesigner(BaseFormulationDesigner):
    """
    Formulation designer that generates a custom formulation from a user-defined list of targets.
    Calculates the baseline coverage for these specific targets without running cross-validation.
    """
    def __init__(self, targets: list[str]):
        """
        Initializes the custom designer.

        Args:
            targets: The list of custom targets to evaluate.
        """
        self.targets = targets
        super().__init__(valency=len(targets), n_jobs=1)

    def fit(self, baseline_result: 'PrevalenceEstimates', progress_callback: Optional[Callable[[int, int], None]] = None) -> 'CustomFormulationDesigner':
        """Evaluates prevalence results against the custom target list."""
        trait_name = baseline_result.trait
        raw_df = baseline_result.data

        # 1. Calculate the true baseline prevalence for everything
        baseline = raw_df.groupby('target')['estimate'].sum().sort_values(ascending=False).reset_index()

        # 2. Filter and reorder the baseline to match the user's custom list exactly
        baseline['target_cat'] = pd.Categorical(baseline['target'], categories=self.targets, ordered=True)
        custom_rankings = baseline.dropna(subset=['target_cat']).sort_values('target_cat').drop(columns=['target_cat'])
        custom_rankings['baseline_rank'] = range(1, len(self.targets) + 1)

        # 3. Create empty stability metrics (since this is a manual override, not a LOO calculation)
        empty_stability = pd.DataFrame(columns=['target', 'mean_loo_rank', 'rank_variance', 'probability_in_top_n']).set_index('target')
        empty_history = pd.DataFrame(columns=['target', 'estimate', 'loo_rank', 'holdout_group'])

        self.formulation_ = Formulation(
            trait=trait_name,
            max_valency=len(self.targets),
            rankings=custom_rankings,
            stability_metrics=empty_stability,
            permutation_history=empty_history
        )
        self.is_fitted_ = True
        return self

class CVFormulationDesigner(BaseFormulationDesigner):
    """
    Rigorous _formulation design using true Leave-One-Out (LOO) cross-validation.

    This method retrains the model for each LOO permutation, which is more
    computationally expensive but necessary for complex models.
    """
    def fit(self, estimator: BaseEstimator, agg_df: pd.DataFrame, loo_col: Optional[str] = None,
            progress_callback: Optional[Callable[[int, int], None]] = None) -> 'CVFormulationDesigner':
        """
        Evaluates an estimator using LOO cross-validation to design a _formulation.

        Args:
            estimator: The estimator instance to use.
            agg_df: The aggregated data for the estimator.
            loo_col: The column name to use for Leave-One-Out cross-validation.

        Returns:
            The fitted designer instance.
        """
        # 1. Baseline
        baseline_result = _clone_estimator(estimator).calculate(agg_df)
        trait_name = baseline_result.trait
        baseline = _extract_ranks(baseline_result.data, 'baseline_rank')

        if loo_col and loo_col in agg_df.columns:
            # 2. Permutations (Parallel Processing)
            groups = agg_df[loo_col].unique()
            total_groups = len(groups)
            
            jobs = (delayed(_run_cv_fold)(estimator, agg_df, loo_col, group) for group in groups)
    
            loo_records = []
            try:
                with Parallel(n_jobs=self.n_jobs, return_as="generator") as parallel:
                    for i, result in enumerate(parallel(jobs), 1):
                        loo_records.append(result)
                        if progress_callback:
                            progress_callback(i, total_groups)
            except TypeError:
                # Fallback for joblib < 1.3 where return_as="generator" isn't supported
                with Parallel(n_jobs=self.n_jobs) as parallel:
                    loo_records = parallel(jobs)
                    if progress_callback:
                        progress_callback(total_groups, total_groups)
    
            history = pd.concat(loo_records, ignore_index=True)
        else:
            history = pd.DataFrame(columns=['target', 'estimate', 'loo_rank', 'holdout_group'])
            if progress_callback: progress_callback(1, 1)

        # 3. Compile
        self.formulation_ = _compile_stability_metrics(baseline, history, trait_name, self.valency)
        self.is_fitted_ = True
        return self


# Functions ------------------------------------------------------------------------------------------------------------
def _extract_ranks(df: pd.DataFrame, rank_col_name: str) -> pd.DataFrame:
    """Consistently groups, sums, sorts, and ranks traits."""
    ranks = df.groupby('target')['estimate'].sum().sort_values(ascending=False).reset_index()
    ranks[rank_col_name] = ranks.index + 1
    return ranks


def _run_cv_fold(estimator: BaseEstimator, agg_df: pd.DataFrame, loo_col: str, group: str) -> pd.DataFrame:
    """Helper function to execute a single CV fold in parallel."""
    holdout_df = agg_df[agg_df[loo_col] != group]
    # Retrain from scratch on the subset
    loo_result = _clone_estimator(estimator).calculate(holdout_df)
    ranks = _extract_ranks(loo_result.data, 'loo_rank')
    ranks['holdout_group'] = group
    return ranks


def _compile_stability_metrics(
        baseline: pd.DataFrame,
        history: pd.DataFrame,
        trait_name: str,
        valency: int
) -> 'Formulation':
    """Compiles the final variance and probability matrix for the _formulation."""
    if history.empty:
        stability = pd.DataFrame(columns=['target', 'mean_loo_rank', 'rank_variance', 'probability_in_top_n']).set_index('target')
    else:
        stability_list = []
        for t in baseline['target']:
            v_hist = history[history['target'] == t]
            var = v_hist['loo_rank'].var()
            stability_list.append({
                'target': t,
                'mean_loo_rank': v_hist['loo_rank'].mean(),
                'rank_variance': float(var) if pd.notna(var) else 0.0,
                'probability_in_top_n': (v_hist['loo_rank'] <= valency).mean()
            })
        stability = pd.DataFrame(stability_list).set_index('target')

    return Formulation(
        trait=trait_name,
        max_valency=valency,
        rankings=baseline,
        stability_metrics=stability,
        permutation_history=history
    )


def _clone_estimator(estimator: BaseEstimator) -> BaseEstimator:
    """
    Safely creates a fresh, unfitted instance of the estimator.
    Avoids deepcopy pickling errors caused by compiled JAX/NumPyro models.
    """
    if hasattr(estimator, 'get_params'):
        # Scikit-learn API compatibility (safely transfers hyperparameters)
        return type(estimator)(**estimator.get_params())
    # Default seroepi fallback
    return type(estimator)()
