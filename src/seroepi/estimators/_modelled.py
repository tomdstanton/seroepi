from typing import Literal, Union, TypeVar, Type
from abc import ABC, abstractmethod
from pathlib import Path
from joblib import dump as joblib_dump, load as joblib_load
from warnings import warn
import functools
try:
    from os import process_cpu_count as cpu_count
except ImportError:
    from os import cpu_count

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import jax.numpy as jnp
import jax.scipy as jsp
from jax.nn import sigmoid
from jax import random, vmap, jit
from numpyro import optim, distributions as dist, diagnostics as diag, sample as samp, set_host_device_count, plate
from numpyro.infer import MCMC, NUTS, Trace_ELBO, SVI, autoguide, Predictive

from seroepi.estimators._base import BaseEstimator, PrevalenceEstimates, IncidenceEstimates
from seroepi.constants import BayesianInferenceMethod, TemporalResolution


# Set-up ---------------------------------------------------------------------------------------------------------------
# Tell JAX to split the CPU into multiple virtual devices for parallel chains
set_host_device_count(cpu_count())


# TypeVars -------------------------------------------------------------------------------------------------------------
T_Modelled = TypeVar('T_Modelled', bound='ModelledMixin')


# Mixins ---------------------------------------------------------------------------------------------------------------
class ModelledMixin(ABC):
    """
    Contract for estimators with an internal fitted state.

    Enforces the scikit-learn fit/predict paradigm and provides universal
    serialization for fitted models.

    Attributes:
        is_fitted_: Boolean indicating if the model has been fitted.
    """

    # State tracking
    is_fitted_: bool = False

    def check_is_fitted(self):
        """
        Checks if the model is fitted.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "This estimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> 'ModelledMixin':
        """Calculates the internal state (e.g., MCMC samples) and saves it to self."""
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame):
        """Uses the fitted internal state to generate predictions on the dataframe."""
        pass

    def calculate(self, df: pd.DataFrame):
        """One-liner to fit and predict on the same data."""
        return self.fit(df).predict(df)

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Universally serializes the fitted estimator instance to disk.

        Args:
            filepath: Path where the model should be saved.
        """
        if not self.is_fitted_:
            warn(f"You are saving a {self.__class__.__name__} that hasn't been fitted yet.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(self, path)

    @classmethod
    def load_model(cls: Type[T_Modelled], filepath: Union[str, Path]) -> T_Modelled:
        """
        Loads a serialized estimator from disk.

        Args:
            filepath: Path to the serialized model file.

        Returns:
            The loaded estimator instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            TypeError: If the loaded model is not of the expected type.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")

        estimator = joblib_load(path)

        # Strict Type Guard
        if not isinstance(estimator, cls):
            raise TypeError(
                f"Type mismatch: Attempted to load into {cls.__name__}, "
                f"but the file contains a {type(estimator).__name__}."
            )

        return estimator


class BayesianMixin:
    """
    Shared inference logic for NumPyro-based Bayesian estimators.
    """
    def _init_bayesian(self, method: BayesianInferenceMethod, num_samples: int, num_chains: int,
                       num_warmup: int, svi_steps: int, seed: int):
        self.method = BayesianInferenceMethod(method) if isinstance(method, str) else method
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.svi_steps = svi_steps
        self.seed = seed
        self.samples_ = None
        self.extra_fields_ = None

    def _check_zero_padding(self, df: pd.DataFrame):
        """Ensures the incoming dataframe has been properly padded for Bayesian inference."""
        meta = df.attrs.get('metric_meta', {})

        # If metadata is entirely missing, we assume it's a raw pandas df and warn,
        # but let the math fail naturally if it's jagged.
        if 'is_zero_padded' in meta and not meta['is_zero_padded']:
            raise ValueError(
                f"Mathematical Integrity Error: {self.__class__.__name__} requires a strictly rectangular, "
                "zero-padded matrix to construct the posterior geometry. "
                "Please regenerate the dataset using `.epi.aggregate_...(pad_zeros=True)`."
            )

    def _run_inference(self, jax_data: dict, rng_key: random.PRNGKey):
        """Routes to the correct inference engine based on self.method."""
        if self.method == BayesianInferenceMethod.MCMC:
            return self._mcmc_inference(jax_data, rng_key)
        elif self.method == BayesianInferenceMethod.SVI:
            return self._svi_inference(jax_data, rng_key)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from MCMC or SVI.")

    def _mcmc_inference(self, jax_data: dict, rng_key: random.PRNGKey):
        """Runs MCMC inference using NUTS."""
        mcmc = MCMC(NUTS(self._model), num_warmup=self.num_warmup, num_samples=self.num_samples,
                    num_chains=self.num_chains, progress_bar=False)
        mcmc.run(rng_key, **jax_data)
        self.extra_fields_ = mcmc.get_extra_fields()
        return mcmc.get_samples()

    def _svi_inference(self, jax_data: dict, rng_key: random.PRNGKey):
        """Runs Stochastic Variational Inference."""
        opt_key, pred_key = random.split(rng_key)
        guide = autoguide.AutoNormal(self._model)
        optimizer = optim.Adam(step_size=0.01)
        svi = SVI(self._model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(opt_key, num_steps=self.svi_steps, **jax_data)
        predictive = Predictive(self._model, guide=guide, params=svi_result.params, num_samples=self.num_samples)
        return predictive(pred_key, **jax_data)

    def diagnostics(self) -> pd.DataFrame:
        """Returns MCMC diagnostics (R-hat, ESS) as a formatted DataFrame."""
        if hasattr(self, 'check_is_fitted'):
            self.check_is_fitted()
        if self.method != BayesianInferenceMethod.MCMC:
            raise TypeError("Diagnostics are only available for MCMC inference.")
            
        summary_dict = diag.summary(self.samples_, prob=0.95, group_by_chain=False)
        
        # Flatten multi-dimensional parameters (like fixed/random effects) into individual rows
        rows = []
        for param, stats in summary_dict.items():
            param_shape = np.shape(stats['mean'])
            
            if len(param_shape) == 0:
                row = {'Parameter': param}
                row.update({k: float(v) for k, v in stats.items()})
                rows.append(row)
            else:
                it = np.nditer(np.empty(param_shape), flags=['multi_index'])
                for _ in it:
                    idx = it.multi_index
                    idx_str = ",".join(map(str, idx))
                    row = {'Parameter': f"{param}[{idx_str}]"}
                    row.update({k: float(np.asarray(v)[idx]) for k, v in stats.items()})
                    rows.append(row)
                    
        # Safely extract and append the MCMC sampler's internal extra fields
        if getattr(self, 'extra_fields_', None) is not None:
            for field, values in self.extra_fields_.items():
                val_array = np.asarray(values, dtype=float)
                rows.append({
                    'Parameter': f"sampler_{field}",
                    'mean': float(np.mean(val_array)),
                    'std': float(np.std(val_array)),
                    'sum': float(np.sum(val_array))
                })

        return pd.DataFrame(rows)


# Estimators -----------------------------------------------------------------------------------------------------------
class BayesianPrevalenceEstimator(ModelledMixin, BayesianMixin, BaseEstimator[PrevalenceEstimates]):
    """
    Bayesian hierarchical model for prevalence estimation.

    This estimator uses MCMC or SVI to fit a binomial model with random effects
    for groups and fixed effects for targets. It handles overdispersion and
    provides credible intervals.

    Examples:
        >>> from seroepi.estimators import BayesianPrevalenceEstimator
        >>> estimator = BayesianPrevalenceEstimator(method='mcmc')
        >>> # result = estimator.calculate(agg_df)
    """
    def __init__(self, method: BayesianInferenceMethod = BayesianInferenceMethod.MCMC, num_samples: int = 1500, num_chains: int = 4,
                 num_warmup: int = 1000, svi_steps: int = 3000, target_event: str = 'event', target_n: str = 'n', seed: int = 42):
        """
        Initializes the BayesianPrevalenceEstimator.

        Args:
            method: Inference method ('mcmc' or 'svi'). Defaults to 'mcmc'.
            num_samples: Number of posterior samples to draw. Defaults to 1500.
            num_chains: Number of MCMC chains. Defaults to 4.
            num_warmup: Number of warmup steps for MCMC. Defaults to 1000.
            svi_steps: Number of optimization steps for SVI. Defaults to 3000.
            target_event: Column name for event counts. Defaults to 'event'.
            target_n: Column name for total counts (denominators). Defaults to 'n'.
            seed: Random seed for reproducibility. Defaults to 42.
        """
        self._init_bayesian(method, num_samples, num_chains, num_warmup, svi_steps, seed)
        self._method_label = f'bayesian_{self.method.value}'
        
        self.target_event = target_event
        self.target_n = target_n

        # Fitted attributes (trailing underscores)
        self.encoders_ = {}
        self.strata_ = []
        self.meta_ = {}

    def _model(self, target_idx, group_idx, n, n_targets, n_groups, event=None):
        """Internal NumPyro model definition."""
        # 1. Global Intercept (Regularized)
        alpha = samp("alpha", dist.Normal(0, 1.5))

        # 2. Fixed effect: Target deviation from baseline
        b_target = samp("b_target", dist.Normal(0, 1).expand([n_targets]))

        # 3. Random effect: Group variation (Non-centered parameterization)
        sd_group = samp("sd_group", dist.HalfNormal(1))
        z_group = samp("z_group", dist.Normal(0, 1).expand([n_groups]))
        r_group = z_group * sd_group

        # 4. Generalized Logit link
        logit_p = alpha + b_target[target_idx] + r_group[group_idx]

        # 5. Likelihood
        samp("obs", dist.Binomial(total_count=n, logits=logit_p), obs=event)

    def fit(self, agg_df: pd.DataFrame) -> 'BayesianPrevalenceEstimator':
        """
        Parses data, fits encoders, and runs inference to get posterior samples.

        Args:
            agg_df: The aggregated DataFrame.

        Returns:
            The fitted estimator instance.
        """
        self._check_zero_padding(agg_df)

        self.strata_, self.meta_ = self._extract_strata(agg_df, exclude_cols=[self.target_event, self.target_n, 'target'])

        df_fit = agg_df.copy()
        
        if not self.strata_:
            df_fit['_dummy_group'] = 'Global'
            group_col = '_dummy_group'
        else:
            group_col = self.strata_[0]
            
        target_col = 'target'

        # Fit and store the encoders exactly once!
        for col in [group_col, target_col]:
            le = LabelEncoder()
            # We fit_transform here, establishing the strict mapping
            df_fit[f'{col}_idx'] = le.fit_transform(df_fit[col].astype(str))
            self.encoders_[col] = le

        jax_data = {
            "target_idx": jnp.array(df_fit[f'{target_col}_idx'].values),
            "group_idx": jnp.array(df_fit[f'{group_col}_idx'].values),
            "n": jnp.array(df_fit[self.target_n].values),
            "event": jnp.array(df_fit[self.target_event].values),
            "n_targets": len(self.encoders_[target_col].classes_),
            "n_groups": len(self.encoders_[group_col].classes_)
        }

        # Run inference
        rng_key = random.PRNGKey(self.seed)
        self.samples_ = self._run_inference(jax_data, rng_key)

        self.is_fitted_ = True
        return self

    def predict(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        """
        Uses the fitted samples and encoders to calculate prevalence bounds.

        Args:
            agg_df: The DataFrame to generate predictions for.

        Returns:
            A PrevalenceEstimates object.
        """
        self.check_is_fitted()

        predict_df = agg_df.copy()
        
        if not self.strata_:
            predict_df['_dummy_group'] = 'Global'
            group_col = '_dummy_group'
        else:
            group_col = self.strata_[0]

        target_col = 'target'

        target_idx = jnp.array(self.encoders_[target_col].transform(predict_df[target_col].astype(str)))
        group_idx = jnp.array(self.encoders_[group_col].transform(predict_df[group_col].astype(str)))

        # Matrix math against self.samples_ ...
        estimate, lower, upper = _compute_prevalence_posterior(
            self.samples_["alpha"],
            self.samples_["b_target"],
            self.samples_["z_group"],
            self.samples_["sd_group"],
            target_idx,
            group_idx
        )

        # Fast assignment (ignores the deep copy overhead)
        result_df = agg_df.assign(
            estimate=np.array(estimate),
            lower=np.array(lower),
            upper=np.array(upper)
        )

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=self.strata_,
            adjusted_for=self.meta_.get("adjusted_for", 'unknown'),
            method=self._method_label,
            aggregation_type=self.meta_.get("aggregation_type", "unknown"),
            trait=self.meta_.get("trait", "unknown")
        )


class GLMPrevalenceEstimator(ModelledMixin, BaseEstimator[PrevalenceEstimates]):
    """
    Frequentist binomial GLM for prevalence estimation.

    Uses statsmodels to fit a Generalized Linear Model with a binomial family
    and logit link.
    """
    def __init__(self, target_event: str = 'event', target_n: str = 'n'):
        """
        Initializes the GLMPrevalenceEstimator.

        Args:
            target_event: Column name for event counts. Defaults to 'event'.
            target_n: Column name for total counts. Defaults to 'n'.
        """
        self.target_event = target_event
        self.target_n = target_n
        self._method_label = "binomial_glm"

    def fit(self, agg_df: pd.DataFrame) -> 'GLMPrevalenceEstimator':
        """Fits the binomial GLM."""
        self.strata_, self.meta_ = self._extract_strata(agg_df, exclude_cols=[self.target_event, self.target_n, 'target'])
        
        feature_cols = self.strata_ + ['target'] if 'target' in agg_df.columns else self.strata_

        # 1. Fit the encoder (We still use sklearn here because statsmodels' categorical handling can be clunky)
        self.encoder_ = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        X_encoded = self.encoder_.fit_transform(agg_df[feature_cols])

        # Add the intercept
        X = sm.add_constant(X_encoded)

        # 2. The Statsmodels Magic: Just pass [Successes, Failures] directly!
        successes = agg_df[self.target_event].values
        failures = agg_df[self.target_n].values - successes
        Y = np.column_stack((successes, failures))

        # 3. Fit the Binomial GLM safely
        # It handles the Fisher Information / Hessian inversion automatically
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm_model = sm.GLM(Y, X, family=sm.families.Binomial())
            self.fit_results_ = glm_model.fit()

        self.is_fitted_ = True
        return self

    def predict(self, agg_df: pd.DataFrame) -> PrevalenceEstimates:
        """Generates predictions and confidence intervals."""
        self.check_is_fitted()

        # Transform new data
        feature_cols = self.strata_ + ['target'] if 'target' in agg_df.columns else self.strata_
        X_encoded = self.encoder_.transform(agg_df[feature_cols])
        X = sm.add_constant(X_encoded, has_constant='add')

        # statsmodels natively handles the delta method and inverse-link transformations
        predictions = self.fit_results_.get_prediction(X).summary_frame(alpha=0.05)

        result_df = agg_df.assign(
            estimate=predictions['mean'].values,
            lower=predictions['mean_ci_lower'].values,
            upper=predictions['mean_ci_upper'].values
        )

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=self.strata_,
            adjusted_for=self.meta_.get("adjusted_for", 'unknown'),
            method=self._method_label,
            aggregation_type=self.meta_.get("aggregation_type", "unknown"),
            trait=self.meta_.get("trait", "unknown")
        )


class SpatialPrevalenceEstimator(ModelledMixin, BayesianMixin, BaseEstimator[PrevalenceEstimates]):
    """
    Gaussian Process (GP) based spatial prevalence estimator.

    Fits a GP model to spatial binomial data, allowing for continuous mapping
    of prevalence across a geographic area.

    Examples:
        >>> from seroepi.estimators import SpatialPrevalenceEstimator
        >>> estimator = SpatialPrevalenceEstimator(lat_col='lat', lon_col='lon')
        >>> # result = estimator.calculate(agg_df)
    """
    def __init__(self, lat_col: str = 'lat', lon_col: str = 'lon',
                 method: BayesianInferenceMethod = BayesianInferenceMethod.MCMC, num_samples: int = 1500, num_chains: int = 4,
                 num_warmup: int = 1000, svi_steps: int = 3000,
                 target_event: str = 'event', target_n: str = 'n', seed: int = 42):
        """
        Initializes the SpatialPrevalenceEstimator.

        Args:
            lat_col: Column name for latitude. Defaults to 'lat'.
            lon_col: Column name for longitude. Defaults to 'lon'.
            method: Inference method. Defaults to 'mcmc'.
            num_samples: Number of samples. Defaults to 1500.
            num_chains: Number of chains. Defaults to 4.
            num_warmup: Number of warmup steps. Defaults to 1000.
            svi_steps: Number of SVI steps. Defaults to 3000.
            target_event: Column for events. Defaults to 'event'.
            target_n: Column for totals. Defaults to 'n'.
            seed: Random seed. Defaults to 42.
        """
        self._init_bayesian(method, num_samples, num_chains, num_warmup, svi_steps, seed)
        self.lat_col = lat_col
        self.lon_col = lon_col
        self._method_label = f'spatial_gp_{self.method.value}'
        
        self.target_event = target_event
        self.target_n = target_n

        # Fitted attributes
        self.X_train_ = None
        self.loc_mean_ = None
        self.loc_scale_ = None
        self.meta_ = {}

    def _model(self, X, n, event=None):
        """Internal NumPyro GP model."""
        # 1. Global Intercept
        alpha = samp("alpha", dist.Normal(0, 1.5))

        # 2. GP Kernel Parameters (Variance/Amplitude and Length-scale)
        var = samp("var", dist.HalfNormal(1.0))
        length = samp("length", dist.InverseGamma(2.0, 1.0))

        # 3. Spatial Covariance Matrix
        K = _rbf_kernel(X, X, var, length)

        # 4. Latent Spatial Field (f)
        f = samp("f", dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=K))

        # 5. Likelihood
        logit_p = alpha + f
        samp("obs", dist.Binomial(total_count=n, logits=logit_p), obs=event)

    def fit(self, agg_df: pd.DataFrame) -> 'SpatialPrevalenceEstimator':
        """Aggregates to unique locations, normalizes, and fits the GP."""
        self._check_zero_padding(agg_df)

        if self.lat_col not in agg_df.columns or self.lon_col not in agg_df.columns:
            raise KeyError(
                f"Spatial estimator requires '{self.lat_col}' and '{self.lon_col}' "
                "in the aggregated data. Please ensure you included them in the 'Stratify By' dropdown."
            )

        # 1. Ensure we only have ONE row per unique lat/lon to prevent singular matrices
        spatial_df = agg_df.groupby([self.lat_col, self.lon_col], as_index=False).agg({
            self.target_event: 'sum',
            self.target_n: 'sum'
        })

        # 2. Extract and Standardize Coordinates
        raw_coords = spatial_df[[self.lat_col, self.lon_col]].astype(float).values
        self.loc_mean_ = np.mean(raw_coords, axis=0)
        self.loc_scale_ = np.std(raw_coords, axis=0) + 1e-8  # Prevent div by zero

        self.X_train_ = (raw_coords - self.loc_mean_) / self.loc_scale_

        jax_data = {
            "X": jnp.array(self.X_train_),
            "n": jnp.array(spatial_df[self.target_n].values),
            "event": jnp.array(spatial_df[self.target_event].values)
        }

        # 3. Run Inference
        rng_key = random.PRNGKey(self.seed)
        self.samples_ = self._run_inference(jax_data, rng_key)

        self.meta_ = agg_df.attrs.get("metric_meta", {})
        self.is_fitted_ = True
        return self

    def predict(self, df: pd.DataFrame) -> PrevalenceEstimates:
        """
        Calculates the conditional predictive posterior for any set of coordinates.

        Args:
            df: DataFrame containing latitude and longitude columns.

        Returns:
            A PrevalenceEstimates object with predicted values at the locations.
        """
        self.check_is_fitted()

        # 1. Standardize new coordinates using the FITTED scaler
        raw_X_test = df[[self.lat_col, self.lon_col]].astype(float).values
        X_test = jnp.array((raw_X_test - self.loc_mean_) / self.loc_scale_)
        X_train = jnp.array(self.X_train_)

        # 2. Use JAX vmap to instantly vectorize this complex math across all 1500 samples!
        estimate, lower, upper = _compute_spatial_posterior(
            self.samples_['var'],
            self.samples_['length'],
            self.samples_['alpha'],
            self.samples_['f'],
            X_train,
            X_test
        )

        result_df = df.assign(
            estimate=np.array(estimate),
            lower=np.array(lower),
            upper=np.array(upper)
        )

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=[self.lat_col, self.lon_col],
            adjusted_for=self.meta_.get("adjusted_for", 'unknown'),
            method=self._method_label,
            aggregation_type=self.meta_.get("aggregation_type", "unknown"),
            trait=self.meta_.get("trait", "unknown")
        )


class GLMIncidenceEstimator(ModelledMixin, BaseEstimator[IncidenceEstimates]):
    """
    Negative Binomial GLM for time-series incidence estimation.

    Fits a Negative Binomial model to count data over time, optionally
    adjusting for sequencing volume (relative incidence).

    Examples:
        >>> from seroepi.estimators import GLMIncidenceEstimator
        >>> estimator = GLMIncidenceEstimator(use_relative_incidence=True)
        >>> # result = estimator.calculate(inc_df)
    """
    def __init__(self, use_relative_incidence: bool = True, forecast_horizon: int = 0):
        """
        Initializes the GLMIncidenceEstimator.

        Args:
            use_relative_incidence: If True, models cases adjusting for total
                sequencing volume (offset). If False, models absolute counts.
            forecast_horizon: Number of future time steps to project. Defaults to 0.
        """
        self.use_relative_incidence = use_relative_incidence
        self.forecast_horizon = forecast_horizon
        self._method_label = "neg_binomial_glm"

        # Internal state tracking
        self.fit_results_ = {}
        self.strata_ = []
        self.meta_ = {}

    def fit(self, inc_df: pd.DataFrame) -> 'GLMIncidenceEstimator':
        """Fits the Negative Binomial GLM to each stratum."""
        self.meta_ = inc_df.attrs.get("metric_meta", {})
        target_col = self.meta_.get("trait")
        self.freq_ = self.meta_.get("freq")
        self.strata_ = self.meta_.get("stratified_by", [])

        if not target_col or not self.freq_:
            raise ValueError("Incidence metadata missing. Ensure data came from `epi.aggregate_incidence`.")

        # Sort entirely upstream to avoid O(N log N) operations inside the loop
        inc_df_sorted = inc_df.sort_values('date')
        group_cols = self.strata_ + ['target'] if 'target' in inc_df_sorted.columns else self.strata_
        groups = inc_df_sorted.groupby(group_cols, observed=True) if group_cols else [('Global', inc_df_sorted)]

        for name, group in groups:
            # Drop the .sort_values() here. Just copy.
            df_group = group.copy()

            # Filter out periods with ZERO sequencing volume for the GLM fit
            df_model = df_group[df_group['total_sequenced'] > 0].copy()

            if len(df_model) < 3:
                self.fit_results_[name] = None  # Not enough data to model
                continue

            Y = df_model['variant_count']
            
            # Skip if there are zero events (cannot fit a count model)
            if Y.sum() == 0:
                self.fit_results_[name] = None
                continue

            # Create a numeric Time Step for the slope
            min_date = df_model['date'].min()
            df_model['time_step'] = _calculate_time_steps(df_model['date'], min_date, self.freq_)

            X = sm.add_constant(df_model['time_step'])

            offset = np.log(df_model['total_sequenced']) if self.use_relative_incidence else None

            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # alpha=1.0 is a robust starting guess for overdispersion
                    model = sm.GLM(Y, X, family=sm.families.NegativeBinomial(alpha=1.0), offset=offset)
                    self.fit_results_[name] = model.fit()
            except Exception as e:
                warn(f"GLM failed to converge for stratum {name}: {e}")
                self.fit_results_[name] = None

        self.is_fitted_ = True
        return self

    def predict(self, inc_df: pd.DataFrame) -> IncidenceEstimates:
        """Extracts Incidence Rate Ratios (IRR) and forecasts trends."""
        self.check_is_fitted()

        freq_str = self.freq_
        freq = _parse_freq_to_offset(freq_str)

        group_cols = self.strata_ + ['target'] if 'target' in inc_df.columns else self.strata_
        # Sort to ensure chronological order for time-steps
        groups = inc_df.sort_values('date').groupby(group_cols, observed=True) if group_cols else [('Global', inc_df)]
        
        results = []
        all_pred_dfs = []

        for name, group in groups:
            fit = self.fit_results_.get(name)

            row = {k: v for k, v in zip(group_cols, name)} if group_cols and isinstance(name, tuple) else {}
            if group_cols and not isinstance(name, tuple):
                row = {group_cols[0]: name}

            if fit is None:
                row.update({
                    'IRR': np.nan, 'IRR_lower': np.nan, 'IRR_upper': np.nan,
                    'p_value': np.nan, 'status': 'Failed/Insufficient Data'
                })
                
                df_pred = group.copy()
                df_pred['estimate'] = np.nan
                df_pred['lower'] = np.nan
                df_pred['upper'] = np.nan
                all_pred_dfs.append(df_pred)
            else:
                coef = fit.params.get('time_step', 0)
                p_val = fit.pvalues.get('time_step', 1.0)
                ci_lower = fit.conf_int()[0].get('time_step', 0)
                ci_upper = fit.conf_int()[1].get('time_step', 0)

                row.update({
                    'IRR': np.exp(coef),
                    'IRR_lower': np.exp(ci_lower),
                    'IRR_upper': np.exp(ci_upper),
                    'p_value': p_val,
                    'status': 'Converged'
                })
                
                # --- Generate Predictions & Forecasts ---
                df_pred = group.copy()
                min_date = df_pred['date'].min()
                
                # Append future horizon rows if required
                if getattr(self, 'forecast_horizon', 0) > 0:
                    max_date = df_pred['date'].max()
                    future_dates = pd.date_range(start=max_date, periods=self.forecast_horizon + 1, freq=freq)[1:]
                    future_df = pd.DataFrame({'date': future_dates})
                    
                    if group_cols:
                        for col in group_cols:
                            future_df[col] = name[group_cols.index(col)] if isinstance(name, tuple) else name
                    
                    # Use the mean historical sequencing volume for relative adjustments in the future
                    future_df['total_sequenced'] = df_pred['total_sequenced'].mean()
                    future_df['variant_count'] = 0
                    
                    df_pred = pd.concat([df_pred, future_df], ignore_index=True)
                
                # Recalculate the mathematical time_step for ALL rows
                df_pred['time_step'] = _calculate_time_steps(df_pred['date'], min_date, self.freq_)

                # Generate predictions
                X = sm.add_constant(df_pred['time_step'], has_constant='add')
                offset = np.log(df_pred['total_sequenced'].clip(lower=1e-8)) if self.use_relative_incidence else None
                
                pred_res = fit.get_prediction(X, offset=offset).summary_frame(alpha=0.05)
                df_pred['estimate'] = pred_res['mean'].values
                df_pred['lower'] = pred_res['mean_ci_lower'].values
                df_pred['upper'] = pred_res['mean_ci_upper'].values
                
                all_pred_dfs.append(df_pred)
                
            results.append(row)

        final_df = pd.concat(all_pred_dfs, ignore_index=True) if all_pred_dfs else inc_df.copy()

        return IncidenceEstimates(
            data=final_df,
            stratified_by=self.strata_,
            adjusted_for=self.meta_.get("adjusted_for", 'unknown'),
            trait=self.meta_.get("trait", "unknown"),
            freq=self.freq_,
            aggregation_type=self.meta_.get("aggregation_type", "unknown"),
            model_results=pd.DataFrame(results)
        )


class BayesianIncidenceEstimator(ModelledMixin, BayesianMixin, BaseEstimator[IncidenceEstimates]):
    """
    Bayesian Structural Time Series (BSTS) for incidence forecasting.

    Uses a Gaussian Random Walk with drift to model latent log-incidence, 
    and a Negative Binomial likelihood to handle overdispersed clinical count data.
    """

    def __init__(self,
                 forecast_horizon: int = 12,
                 method: BayesianInferenceMethod = BayesianInferenceMethod.MCMC,
                 num_samples: int = 1500,
                 num_chains: int = 4,
                 num_warmup: int = 1000,
                 svi_steps: int = 3000,
                 seed: int = 42):

        self.forecast_horizon = forecast_horizon
        self._init_bayesian(method, num_samples, num_chains, num_warmup, svi_steps, seed)
        self._method_label = f'bsts_forecast_{self.method.value}'

        # Internal state
        self.strata_ = []
        self.meta_ = {}

    def _model(self, T, n_strata, Y=None, forecast_horizon=0):
        """The NumPyro BSTS Model."""
        total_T = T + forecast_horizon
        # Priors for the latent state parameters
        with plate("strata", n_strata, dim=-1):
            # Base log-rate at t=0
            mu_0 = samp("mu_0", dist.Normal(0, 2))
            # The overarching trend (drift)
            drift = samp("drift", dist.Normal(0, 0.5))
            # How volatile the month-to-month changes are
            sigma_rw = samp("sigma_rw", dist.HalfNormal(0.5))
            # Overdispersion for the Negative Binomial count data
            dispersion = samp("dispersion", dist.HalfNormal(2))
            
        # Gaussian Random Walk (Innovations)
        with plate("time", T, dim=-2):
            with plate("strata_inner", n_strata, dim=-1):
                innovations_hist = samp("innovations", dist.Normal(0, 1))
                
        if forecast_horizon > 0:
            with plate("time_future", forecast_horizon, dim=-2):
                with plate("strata_inner_future", n_strata, dim=-1):
                    innovations_future = samp("innovations_future", dist.Normal(0, 1))
            innovations = jnp.concatenate([innovations_hist, innovations_future], axis=-2)
        else:
            innovations = innovations_hist
                
        # Construct the Latent Log-Rate, use JAX's cumulative sum to build the random walk over time
        rw = jnp.cumsum(innovations * sigma_rw, axis=-2)
        time_steps = jnp.arange(total_T)[:, None]
        # Latent state equation: Baseline + Trend + Random Walk
        log_rate = mu_0 + (time_steps * drift) + rw
        
        # Likelihood function
        if Y is not None:  # INFERENCE MODE (Fitting historical data)
            historical_rate = log_rate[:T]
            with plate("obs_time", T, dim=-2):
                with plate("obs_strata", n_strata, dim=-1):
                    samp("obs", dist.NegativeBinomial2(mean=jnp.exp(historical_rate), concentration=dispersion), obs=Y)
        else:  # FORECASTING MODE (Projecting the future)
            with plate("obs_time", total_T, dim=-2):
                with plate("obs_strata", n_strata, dim=-1):
                    samp("obs", dist.NegativeBinomial2(mean=jnp.exp(log_rate), concentration=dispersion))

    def fit(self, inc_df: pd.DataFrame) -> 'BayesianIncidenceEstimator':
        """Pivots the count data into a matrix and fits the BSTS model."""
        self.meta_ = inc_df.attrs.get("metric_meta", {})
        self.strata_ = self.meta_.get("stratified_by", [])
        
        df = inc_df.copy()
        group_cols = self.strata_ + ['target'] if 'target' in df.columns else self.strata_
        
        if not group_cols:
            df['_dummy_group'] = 'Global'
            group_cols = ['_dummy_group']

        # Pivot to get a T x n_strata continuous matrix
        pivot_df = df.pivot_table(
            index='date',
            columns=group_cols,
            values='variant_count',
            fill_value=0
        )
        
        # Filter out empty strata (all 0s) to prevent MCMC gradients from exploding to -inf
        strata_sums = pivot_df.sum(axis=0)
        active_cols = strata_sums[strata_sums > 0].index
        self.inactive_strata_labels_ = strata_sums[strata_sums == 0].index
        
        if len(active_cols) == 0:
            raise ValueError("All strata have zero historical events. Cannot fit the Bayesian model.")
            
        pivot_df = pivot_df[active_cols]
        
        self.dates_ = pivot_df.index
        self.T_ = len(self.dates_)
        self.n_strata_ = pivot_df.shape[1]
        self.strata_labels_ = pivot_df.columns
        
        jax_data = {
            "T": self.T_,
            "n_strata": self.n_strata_,
            "Y": jnp.array(pivot_df.values),
            "forecast_horizon": 0  # 0 during inference
        }
        
        rng_key = random.PRNGKey(self.seed)
        self.samples_ = self._run_inference(jax_data, rng_key)
        self.is_fitted_ = True
        return self

    def predict(self, inc_df: pd.DataFrame) -> IncidenceEstimates:
        """Forecasts future incidence counts using the fitted posterior samples."""
        self.check_is_fitted()
        
        pred = Predictive(self._model, self.samples_)
        pred_key = random.PRNGKey(self.seed + 1)
        
        # Run forward for predictions
        pred_samples = pred(
            pred_key,
            T=self.T_,
            n_strata=self.n_strata_,
            Y=None,
            forecast_horizon=self.forecast_horizon
        )
        
        # obs_draws is shape (num_samples, T + forecast_horizon, n_strata)
        estimate, lower, upper, drifts, p_vals = _summarize_incidence_posterior(
            pred_samples['obs'], 
            self.samples_['drift']
        )
        
        # Reconstruct continuous dates, adding the future horizon
        freq_str = self.meta_.get("freq", TemporalResolution.MONTH.value)
        freq = _parse_freq_to_offset(freq_str)
            
        all_dates = pd.date_range(start=self.dates_[0], periods=self.T_ + self.forecast_horizon, freq=freq)
        
        # Melt the predictions back into long format
        results = []
        group_cols = self.strata_ + ['target'] if 'target' in inc_df.columns else self.strata_
        if not group_cols:
            group_cols = ['_dummy_group']
            
        for i, date in enumerate(all_dates):
            for j, strata_val in enumerate(self.strata_labels_):
                row = {'date': date}
                if group_cols:
                    if isinstance(strata_val, tuple):
                        row.update(zip(group_cols, strata_val))
                    else:
                        row[group_cols[0]] = strata_val
                
                row['estimate'] = float(estimate[i, j])
                row['lower'] = float(lower[i, j])
                row['upper'] = float(upper[i, j])
                results.append(row)
                
            for strata_val in self.inactive_strata_labels_:
                row = {'date': date}
                if group_cols:
                    if isinstance(strata_val, tuple):
                        row.update(zip(group_cols, strata_val))
                    else:
                        row[group_cols[0]] = strata_val
                
                row['estimate'] = 0.0
                row['lower'] = 0.0
                row['upper'] = 0.0
                results.append(row)
                
        res_df = pd.DataFrame(results)
        
        # Merge back the original counts (this ensures that historical rows keep variant_count, total_sequenced)
        merge_cols = ['date'] + (self.strata_ + ['target'] if 'target' in inc_df.columns else self.strata_)
        final_df = res_df.merge(inc_df, on=merge_cols, how='left')
        target = self.meta_.get("trait")
        # Build the model summary with incidence rate ratios (drift exponential)
        
        model_res = []
        for j, strata_val in enumerate(self.strata_labels_):
            row = {}
            if group_cols:
                if isinstance(strata_val, tuple):
                    row.update(zip(group_cols, strata_val))
                else:
                    row[group_cols[0]] = strata_val
            row['IRR'] = float(jnp.exp(drifts[j]))
            row['prob_increasing'] = float(p_vals[j])
            row['status'] = 'Converged'
            model_res.append(row)
            
        for strata_val in self.inactive_strata_labels_:
            row = {}
            if group_cols:
                if isinstance(strata_val, tuple):
                    row.update(zip(group_cols, strata_val))
                else:
                    row[group_cols[0]] = strata_val
            row['IRR'] = np.nan
            row['prob_increasing'] = np.nan
            row['status'] = 'Zero Events'
            model_res.append(row)
            
        return IncidenceEstimates(
            data=final_df,
            stratified_by=self.strata_,
            adjusted_for=self.meta_.get("adjusted_for", "unknown"),
            trait=target or "unknown",
            freq=freq_str,
            aggregation_type=self.meta_.get("aggregation_type", "unknown"),
            model_results=pd.DataFrame(model_res)
        )


# Kernels --------------------------------------------------------------------------------------------------------------
def _rbf_kernel(X, Z, var, length, jitter=1e-5):
    """Calculates the Exponentiated Quadratic (RBF) Kernel matrix."""
    # Compute squared distance between all pairs of points in X and Z
    dist_sq = jnp.sum((X[:, None, :] - Z[None, :, :]) ** 2, axis=-1)
    K = var * jnp.exp(-0.5 * dist_sq / (length ** 2))

    # Add jitter to the diagonal for numerical stability (only if computing symmetric K_XX)
    if X.shape == Z.shape:
        K += jitter * jnp.eye(X.shape[0])
    return K

def _parse_freq_to_offset(freq_str: str) -> str:
    """Safely parses a TemporalResolution string to a pandas offset."""
    try:
        freq = TemporalResolution(freq_str).pandas_offset
        return freq if freq else 'MS'
    except ValueError:
        return 'MS'

def _calculate_time_steps(dates: pd.Series, min_date: pd.Timestamp, freq: str) -> pd.Series:
    """Converts a datetime series into integer time steps based on frequency."""
    if freq == TemporalResolution.MONTH.value or freq.startswith('M'):
        return (dates.dt.year - min_date.year) * 12 + (dates.dt.month - min_date.month)
    elif freq == TemporalResolution.WEEK.value or freq.startswith('W'):
        return (dates - min_date).dt.days // 7
    elif freq == TemporalResolution.YEAR.value or freq.startswith('Y'):
        return dates.dt.year - min_date.year
    return (dates - min_date).dt.days


# Kernels --------------------------------------------------------------------------------------------------------------
@functools.partial(jit)
@functools.partial(vmap, in_axes=(0, 0, 0, 0, None, None))
def _vectorized_gp_predict(var, length, alpha, f, X_train, X_test):
    """Pure function for exact GP conditional predictions. JIT compiled once."""
    K_XX = _rbf_kernel(X_train, X_train, var, length, jitter=1e-5)
    K_Xstar_X = _rbf_kernel(X_test, X_train, var, length, jitter=0.0)
    v = jsp.linalg.solve(K_XX, f, assume_a='pos')
    f_star = K_Xstar_X @ v
    return alpha + f_star

@jit
def _compute_spatial_posterior(var, length, alpha, f, X_train, X_test):
    """Vectorized and JIT-compiled posterior boundaries for GP Spatial prevalence."""
    logit_p_draws = _vectorized_gp_predict(var, length, alpha, f, X_train, X_test)
    p_draws = sigmoid(logit_p_draws)
    estimate = jnp.mean(p_draws, axis=0)
    bounds = jnp.quantile(p_draws, jnp.array([0.025, 0.975]), axis=0)
    return estimate, bounds[0], bounds[1]

@jit
def _compute_prevalence_posterior(alpha, b_target, z_group, sd_group, target_idx, group_idx):
    """Vectorized and JIT-compiled posterior boundaries for hierarchical prevalence."""
    alpha_draws = alpha[:, None]
    r_group_draws = z_group * sd_group[:, None]
    logit_p_draws = alpha_draws + b_target[:, target_idx] + r_group_draws[:, group_idx]
    p_draws = sigmoid(logit_p_draws)
    estimate = jnp.mean(p_draws, axis=0)
    bounds = jnp.quantile(p_draws, jnp.array([0.025, 0.975]), axis=0)
    return estimate, bounds[0], bounds[1]

@jit
def _summarize_incidence_posterior(obs_draws, drifts_draws):
    """Vectorized and JIT-compiled posterior summaries for BSTS Incidence."""
    estimate = jnp.mean(obs_draws, axis=0)
    bounds = jnp.quantile(obs_draws, jnp.array([0.025, 0.975]), axis=0)
    drifts = jnp.mean(drifts_draws, axis=0)
    p_vals = jnp.mean(drifts_draws > 0, axis=0)
    return estimate, bounds[0], bounds[1], drifts, p_vals