"""Module for classical serology modeling, including titer classification and serocatalytic FOI estimation."""

from typing import Any, Literal
import polars as pl
import numpy as np

try:
    from sklearn.mixture import GaussianMixture
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

try:
    import numpyro
    import numpyro.distributions as dist
    import jax.numpy as jnp
    from jax import random
    _JAX_AVAILABLE = True
except ImportError:
    _JAX_AVAILABLE = False


from seroepi.estimators.base import BaseStatefulEstimator, SeropositivityEstimates, ForceOfInfectionEstimates
from seroepi.estimators.mixins import ModelledMixin, BayesianMixin


class TiterClassificationEstimator(ModelledMixin, BaseStatefulEstimator[SeropositivityEstimates]):
    """Fits a Gaussian Mixture Model to continuous assay data to classify serostatus.
    
    Identifies the optimal cutoff between the unexposed (negative) and exposed (positive)
    distributions in continuous antibody titer data.
    """
    
    _result_class = SeropositivityEstimates

    def __init__(self, titer_column: str, n_components: int = 2):
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn must be installed to use TiterClassificationEstimator")
        self.titer_column = titer_column
        self.n_components = n_components
        self._gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        self.cutoff_ = None
        self.neg_mean_ = None
        self.pos_mean_ = None
        
    def fit(self, df: Any) -> "TiterClassificationEstimator":
        raw_df = df.data if hasattr(df, "data") else df
        if self.titer_column not in raw_df.columns:
            raise ValueError(f"Column '{self.titer_column}' not found in dataframe.")
            
        # Extract 1D continuous assay data and drop nulls
        X = raw_df.select(self.titer_column).drop_nulls().to_numpy().reshape(-1, 1)
        self._gmm.fit(X)
        
        means = self._gmm.means_.flatten()
        # Sort components by mean (assume higher mean = seropositive)
        sorted_indices = np.argsort(means)
        self.neg_mean_ = means[sorted_indices[0]]
        self.pos_mean_ = means[sorted_indices[-1]]
        
        # Calculate intersection of PDFs to define the cutoff
        # (For simplicity here, we use the midpoint between means, but ideally it's the PDF intersection)
        self.cutoff_ = (self.neg_mean_ + self.pos_mean_) / 2.0
        
        self.is_fitted_ = True
        return self
        
    def _generate_predictions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Appends a boolean `is_seropositive` column based on the fitted cutoff."""
        self.check_is_fitted()
        return df.with_columns(
            is_seropositive=(pl.col(self.titer_column) >= self.cutoff_)
        )


class SerocatalyticEstimator(ModelledMixin, BayesianMixin, BaseStatefulEstimator[ForceOfInfectionEstimates]):
    """Age-structured catalytic model for estimating Force of Infection.
    
    Fits a reversible or irreversible catalytic model to age-stratified binary
    seropositivity data using Bayesian inference.
    """
    
    _result_class = ForceOfInfectionEstimates

    def __init__(
        self, 
        age_column: str, 
        status_column: str, 
        model_type: Literal["reversible", "irreversible"] = "irreversible", 
        **kwargs
    ):
        if not _JAX_AVAILABLE:
            raise ImportError("numpyro and jax must be installed to use SerocatalyticEstimator")
            
        self.age_column = age_column
        self.status_column = status_column
        self.model_type = model_type
        self._init_bayesian(**kwargs)
        
    def _model(self, age, seropositive, N):
        """NumPyro Bayesian model for the catalytic equations."""
        lam = numpyro.sample("lambda", dist.Exponential(1.0))
        
        if self.model_type == "irreversible":
            p = 1.0 - jnp.exp(-lam * age)
        else:
            rho = numpyro.sample("rho", dist.Exponential(1.0))
            p = (lam / (lam + rho)) * (1.0 - jnp.exp(-(lam + rho) * age))
            
        numpyro.sample("obs", dist.Binomial(N, probs=p), obs=seropositive)

    def fit(self, df: Any) -> "SerocatalyticEstimator":
        raw_df = df.data if hasattr(df, "data") else df
        
        # Aggregate by age group
        agg_df = raw_df.group_by(self.age_column).agg(
            N=pl.len(),
            seropositive=pl.col(self.status_column).cast(pl.Int64).sum()
        ).drop_nulls()
        
        jax_data = {
            "age": jnp.array(agg_df[self.age_column].to_numpy(dtype=np.float32)),
            "seropositive": jnp.array(agg_df["seropositive"].to_numpy(dtype=np.int32)),
            "N": jnp.array(agg_df["N"].to_numpy(dtype=np.int32))
        }
        
        rng_key = random.PRNGKey(self.seed)
        self.samples_ = self._run_inference(jax_data, rng_key)
        self.is_fitted_ = True
        return self

    def _generate_predictions(self, df: pl.DataFrame) -> pl.DataFrame:
        """Returns the fitted Force of Infection (lambda) and Recovery (rho) rates."""
        self.check_is_fitted()
        lam_mean = float(np.mean(self.samples_["lambda"]))
        rho_mean = float(np.mean(self.samples_["rho"])) if "rho" in self.samples_ else None
        
        return pl.DataFrame({
            "lambda_foi": [lam_mean],
            "rho_recovery": [rho_mean]
        })
