from abc import ABC, abstractmethod  # noqa: D100
from datetime import date, datetime, timedelta
from enum import StrEnum
from functools import partial
from os import cpu_count
from pathlib import Path
from typing import Any, ClassVar, cast, Literal, Self, TypeVar
from warnings import catch_warnings, simplefilter, warn

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import polars as pl
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from jax import jit, random, vmap
from jax.nn import sigmoid
from joblib import dump as joblib_dump
from joblib import load as joblib_load
from numpyro import diagnostics as diag
from numpyro import distributions as dist
from numpyro import optim, plate, set_host_device_count
from numpyro import sample as samp
from numpyro.infer import MCMC, NUTS, SVI, Predictive, Trace_ELBO, autoguide
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from seroepi.constants import AggregationType, BayesianInferenceMethod, ConfidenceIntervalMethod, TemporalResolution
from seroepi.domains.base import get_dataframe_metadata
from seroepi.estimators.base import BaseEstimator, IncidenceEstimates, PrevalenceEstimates

def _frequentist_kernel(counts: np.ndarray, denominators: np.ndarray, alpha: float, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    safe_denoms = np.where(denominators == 0, 1, denominators)
    prop = counts / safe_denoms
    prop = np.where(denominators == 0, 0.0, prop)
    lower, upper = sm.stats.proportion_confint(counts, safe_denoms, alpha=alpha, method=method)
    lower = np.where(denominators == 0, 0.0, lower)
    upper = np.where(denominators == 0, 0.0, upper)
    return prop, lower, upper

_FREQUENTIST_KERNELS = {
    "wilson": partial(_frequentist_kernel, method="wilson"),
    "wald": partial(_frequentist_kernel, method="normal"),
    "agresti_coull": partial(_frequentist_kernel, method="agresti_coull"),
    "clopper_pearson": partial(_frequentist_kernel, method="beta"),
    "jeffreys": partial(_frequentist_kernel, method="jeffreys"),
}
# Set-up ---------------------------------------------------------------------------------------------------------------
set_host_device_count(cpu_count() or 1)


# TypeVars -------------------------------------------------------------------------------------------------------------
T_Modelled = TypeVar("T_Modelled", bound="ModelledMixin")


# Mixins ---------------------------------------------------------------------------------------------------------------
from seroepi.estimators.mixins import ModelledMixin, BayesianMixin

@jit
def _compute_prevalence_posterior(
    alpha: jnp.ndarray,
    b_target: jnp.ndarray,
    z_group: jnp.ndarray,
    sd_group: jnp.ndarray,
    target_idx: jnp.ndarray,
    group_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute the prevalence posterior distribution using JAX."""
    # Expand dims for scalar-like parameters over samples
    if alpha.ndim == 1:
        alpha = jnp.expand_dims(alpha, axis=-1)
    if sd_group.ndim == 1:
        sd_group = jnp.expand_dims(sd_group, axis=-1)

    r_group = z_group * sd_group
    logit_p = alpha + b_target[:, target_idx] + r_group[:, group_idx]
    p = sigmoid(logit_p)

    mean_p = jnp.mean(p, axis=0)
    lower_p = jnp.percentile(p, 2.5, axis=0)
    upper_p = jnp.percentile(p, 97.5, axis=0)

    return mean_p, lower_p, upper_p

class UnpooledPrevalenceEstimator(BaseEstimator[PrevalenceEstimates]):  # noqa: D101
    def __init__(self, method: ConfidenceIntervalMethod | str = ConfidenceIntervalMethod.WILSON, alpha: float = 0.05, trait: str | StrEnum | None = None):  # noqa: ANN204, D107
        self.method = method.lower()
        self._method_label = f"unpooled_{self.method}"
        self._method_func = _FREQUENTIST_KERNELS.get(self.method, None)
        if self._method_func is None:
            raise ValueError(f"Unknown method: {self.method}. Choose from: {list(_FREQUENTIST_KERNELS.keys())}")
        self.alpha = alpha
        self.trait = f"{trait.value}" if hasattr(trait, "value") else (f"{trait}" if trait is not None else None)

    def get_params(self) -> dict[str, Any]:
        """Returns parameters for cloning compatibility during Cross-Validation."""
        return {"method": self.method, "alpha": self.alpha, "trait": self.trait}

    def calculate(self, agg_df: Any) -> PrevalenceEstimates:  # type: ignore # noqa: ANN401, D102
        """Expects the output of df.epi.aggregate_prevalence()"""  # noqa: D415
        stratified_by, meta = self._extract_strata(agg_df, exclude_cols=["event", "n", "target"])
        df = agg_df.data if hasattr(agg_df, "data") else agg_df

        # Extract vectors for fast numpy math
        counts = df["event"].to_numpy()
        denominators = df["n"].to_numpy()

        # Route to the selected mathematical method
        prop, lower, upper = self._method_func(counts, denominators, self.alpha)  # type: ignore

        # Fast assignment
        result_df = df.with_columns(
            estimate=pl.Series(np.nan_to_num(prop, nan=0.0)),
            lower=pl.Series(np.nan_to_num(lower, nan=0.0)),
            upper=pl.Series(np.nan_to_num(upper, nan=0.0)),
        )

        # 4-tier trait resolution hierarchy
        # Tier 1: Accessor metadata
        raw_trait = meta.get("trait", None)
        if hasattr(raw_trait, "value"):
            raw_trait = str(raw_trait.value)
        elif raw_trait is not None:
            raw_trait = str(raw_trait)

        # Tier 2: Estimator self.trait attribute
        if (not raw_trait or raw_trait == "unknown") and self.trait is not None:
            raw_trait = self.trait

        # Tier 3: Single unique target value in "target" column
        if (not raw_trait or raw_trait == "unknown") and "target" in df.columns and len(df) > 0:
            if df["target"].n_unique() == 1:
                raw_trait = df["target"][0]

        # Tier 4: Column inference / Fallback
        if not raw_trait or raw_trait == "unknown":
            if "target" in df.columns and len(df) > 0:
                raw_trait = "target"
            else:
                exclude = {"n", "event", "variant_count", "total_sequenced", "date", "latitude", "longitude", "estimate", "lower", "upper"}
                non_count = [c for c in df.columns if c not in exclude and c not in stratified_by]
                if non_count:
                    raw_trait = non_count[0]
                else:
                    raw_trait = "target"

        if not raw_trait:
            raw_trait = "unknown"
        trait_str = f"{raw_trait.value}" if hasattr(raw_trait, "value") else str(raw_trait)

        adj_raw = meta.get("adjusted_for", "unknown")
        adj_str = (
            f"{adj_raw.value}" if hasattr(adj_raw, "value") else (f"{adj_raw}" if adj_raw is not None else "unknown")
        )

        agg_type = meta.get("aggregation_type", AggregationType.TRAIT)
        if hasattr(agg_type, "value"):
            agg_type = f"{agg_type.value}"
        else:
            agg_type = f"{agg_type}"

        method_str = f"{self._method_label.value}" if hasattr(self._method_label, "value") else f"{self._method_label}"

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=[f"{s.value}" if hasattr(s, "value") else f"{s}" for s in stratified_by],
            adjusted_for=adj_str,
            method=method_str,
            aggregation_type=agg_type,
            trait=trait_str,
        )

class BayesianPrevalenceEstimator(ModelledMixin, BayesianMixin, BaseEstimator[PrevalenceEstimates]):
    """Bayesian hierarchical model for prevalence estimation.

    This estimator uses MCMC or SVI to fit a binomial model with random effects
    for groups and fixed effects for targets. It handles overdispersion and
    provides credible intervals.
    """

    def __init__(  # noqa: ANN204, D107
        self,
        method: BayesianInferenceMethod = BayesianInferenceMethod.MCMC,
        num_samples: int = 1500,
        num_chains: int = 4,
        num_warmup: int = 1000,
        svi_steps: int = 3000,
        target_event: str | StrEnum = "event",
        target_n: str | StrEnum = "n",
        seed: int = 42,
    ):
        self._init_bayesian(method, num_samples, num_chains, num_warmup, svi_steps, seed)
        self._method_label = f"bayesian_{self.method.value}"
        self._svi_return_sites = ["alpha", "b_target", "sd_group", "z_group", "obs"]

        self.target_event = str(target_event)
        self.target_n = str(target_n)

        self.encoders_ = {}
        self.strata_ = []
        self.meta_ = {}

    def get_params(self) -> dict[str, Any]:
        """Returns parameters for cloning compatibility."""
        return {
            "method": self.method,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "num_warmup": self.num_warmup,
            "svi_steps": self.svi_steps,
            "target_event": self.target_event,
            "target_n": self.target_n,
            "seed": self.seed,
        }

    def _model(self, target_idx, group_idx, n, n_targets, n_groups, event=None):  # noqa: ANN001, ANN202
        """Internal NumPyro model definition."""
        alpha = samp("alpha", dist.Normal(0, 1.5))
        b_target = samp("b_target", dist.Normal(0, 1).expand([n_targets]))
        sd_group = samp("sd_group", dist.HalfNormal(1))
        z_group = samp("z_group", dist.Normal(0, 1).expand([n_groups]))
        r_group = z_group * sd_group
        logit_p = alpha + b_target[target_idx] + r_group[group_idx]  # type: ignore
        samp("obs", dist.Binomial(total_count=n, logits=logit_p), obs=event)

    def fit(self, agg_df: Any) -> "BayesianPrevalenceEstimator":  # type: ignore # noqa: ANN401, D102
        self._check_zero_padding(agg_df)

        self.strata_, self.meta_ = self._extract_strata(
            agg_df, exclude_cols=[self.target_event, self.target_n, "target"]
        )
        df = agg_df.data if hasattr(agg_df, "data") else agg_df
        raw_trait = self.meta_.get("trait")
        if (not raw_trait or raw_trait == "unknown") and "target" in df.columns and len(df) > 0:
            if df["target"].n_unique() == 1:
                self.meta_["trait"] = df["target"][0]
            else:
                self.meta_["trait"] = "target"

        df_fit = df
        if not self.strata_:
            df_fit = df_fit.with_columns(_dummy_group=pl.lit("Global"))
            group_col = "_dummy_group"
        else:
            group_col = self.strata_[0]

        target_col = "target"

        for col in [group_col, target_col]:
            le = LabelEncoder()
            str_vals = df_fit[col].cast(pl.Utf8).to_numpy()
            encoded_vals = le.fit_transform(str_vals)
            df_fit = df_fit.with_columns(**{f"{col}_idx": pl.Series(encoded_vals)})
            self.encoders_[col] = le

        jax_data = {
            "target_idx": jnp.array(df_fit[f"{target_col}_idx"].cast(pl.Int32).to_numpy()),
            "group_idx": jnp.array(df_fit[f"{group_col}_idx"].cast(pl.Int32).to_numpy()),
            "n": jnp.array(df_fit[self.target_n].cast(pl.Int32).to_numpy()),
            "event": jnp.array(df_fit[self.target_event].cast(pl.Int32).to_numpy()),
            "n_targets": len(self.encoders_[target_col].classes_),
            "n_groups": len(self.encoders_[group_col].classes_),
        }

        rng_key = random.PRNGKey(self.seed)
        self.samples_ = self._run_inference(jax_data, rng_key)

        self.is_fitted_ = True
        return self

    def predict(self, agg_df: Any) -> PrevalenceEstimates:  # type: ignore # noqa: ANN401, D102
        self.check_is_fitted()

        predict_df = agg_df.data if hasattr(agg_df, "data") else agg_df

        if not self.strata_:
            predict_df_mod = predict_df.with_columns(_dummy_group=pl.lit("Global"))
            group_col = "_dummy_group"
        else:
            predict_df_mod = predict_df
            group_col = self.strata_[0]

        target_col = "target"

        target_str_vals = predict_df_mod[target_col].cast(pl.Utf8).to_numpy()
        group_str_vals = predict_df_mod[group_col].cast(pl.Utf8).to_numpy()

        target_idx = jnp.array(self.encoders_[target_col].transform(target_str_vals))
        group_idx = jnp.array(self.encoders_[group_col].transform(group_str_vals))

        assert self.samples_ is not None
        estimate, lower, upper = _compute_prevalence_posterior(
            self.samples_["alpha"],
            self.samples_["b_target"],
            self.samples_["z_group"],
            self.samples_["sd_group"],
            target_idx,
            group_idx,
        )

        result_df = predict_df.with_columns(
            estimate=pl.Series(np.array(estimate)), lower=pl.Series(np.array(lower)), upper=pl.Series(np.array(upper))
        )

        raw_trait = self.meta_.get("trait", "unknown")
        if (not raw_trait or raw_trait == "unknown") and "target" in predict_df.columns and len(predict_df) > 0:
            if predict_df["target"].n_unique() == 1:
                raw_trait = predict_df["target"][0]
            else:
                raw_trait = "target"
        trait_str = (
            f"{raw_trait.value}"
            if hasattr(raw_trait, "value")
            else (f"{raw_trait}" if raw_trait is not None else "unknown")
        )

        adj_raw = self.meta_.get("adjusted_for", "unknown")
        adj_str = (
            f"{adj_raw.value}" if hasattr(adj_raw, "value") else (f"{adj_raw}" if adj_raw is not None else "unknown")
        )

        agg_type = self.meta_.get("aggregation_type", "unknown")
        if hasattr(agg_type, "value"):
            agg_type = f"{agg_type.value}"
        else:
            agg_type = f"{agg_type}"

        method_str = f"{self._method_label.value}" if hasattr(self._method_label, "value") else f"{self._method_label}"

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=[f"{s.value}" if hasattr(s, "value") else f"{s}" for s in self.strata_],
            adjusted_for=adj_str,
            method=method_str,
            aggregation_type=agg_type,
            trait=trait_str,
        )

class GLMPrevalenceEstimator(ModelledMixin, BaseEstimator[PrevalenceEstimates]):
    """Frequentist binomial GLM for prevalence estimation.

    Uses statsmodels to fit a Generalized Linear Model with a binomial family
    and logit link.
    """

    def __init__(
        self,
        target_event: str | StrEnum = "event",
        target_n: str | StrEnum = "n",
        trait: str | StrEnum | None = None,
    ):  # noqa: ANN204, D107
        self.target_event = str(target_event.value) if hasattr(target_event, "value") else str(target_event)
        self.target_n = str(target_n.value) if hasattr(target_n, "value") else str(target_n)
        self.trait = f"{trait.value}" if hasattr(trait, "value") else (f"{trait}" if trait is not None else None)
        self._method_label = "binomial_glm"

    def get_params(self) -> dict[str, Any]:
        """Returns parameters for cloning compatibility."""
        return {
            "target_event": self.target_event,
            "target_n": self.target_n,
            "trait": self.trait,
        }

    def fit(self, agg_df: Any) -> "GLMPrevalenceEstimator":  # type: ignore # noqa: ANN401, D102
        self.strata_, self.meta_ = self._extract_strata(
            agg_df, exclude_cols=[self.target_event, self.target_n, "target"]
        )
        df = agg_df.data if hasattr(agg_df, "data") else agg_df
        raw_trait = self.meta_.get("trait")
        if (not raw_trait or raw_trait == "unknown") and self.trait is not None:
            raw_trait = self.trait
            self.meta_["trait"] = raw_trait
        if (not raw_trait or raw_trait == "unknown") and "target" in df.columns and len(df) > 0:
            if df["target"].n_unique() == 1:
                self.meta_["trait"] = df["target"][0]
            else:
                self.meta_["trait"] = "target"

        feature_cols = self.strata_ + ["target"] if "target" in df.columns else self.strata_

        self.encoder_ = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        X_encoded = self.encoder_.fit_transform(df.select(feature_cols).to_numpy())

        X = sm.add_constant(X_encoded)

        successes = df[self.target_event].to_numpy()
        failures = df[self.target_n].to_numpy() - successes
        Y = np.column_stack((successes, failures))

        with catch_warnings():
            simplefilter("ignore")
            glm_model = sm.GLM(Y, X, family=sm.families.Binomial())
            self.fit_results_ = glm_model.fit()

        self.is_fitted_ = True
        return self

    def predict(self, agg_df: Any) -> PrevalenceEstimates:  # type: ignore # noqa: ANN401, D102
        self.check_is_fitted()

        df = agg_df.data if hasattr(agg_df, "data") else agg_df
        feature_cols = self.strata_ + ["target"] if "target" in df.columns else self.strata_

        X_encoded = self.encoder_.transform(df.select(feature_cols).to_numpy())
        X = sm.add_constant(X_encoded, has_constant="add")

        predictions = self.fit_results_.get_prediction(X).summary_frame(alpha=0.05)

        result_df = df.with_columns(
            estimate=pl.Series(predictions["mean"].values),
            lower=pl.Series(predictions["mean_ci_lower"].values),
            upper=pl.Series(predictions["mean_ci_upper"].values),
        )

        raw_trait = self.meta_.get("trait", "unknown")
        if (not raw_trait or raw_trait == "unknown") and self.trait is not None:
            raw_trait = self.trait
        if (not raw_trait or raw_trait == "unknown") and "target" in df.columns and len(df) > 0:
            if df["target"].n_unique() == 1:
                raw_trait = df["target"][0]
            else:
                raw_trait = "target"
        trait_str = (
            f"{raw_trait.value}"
            if hasattr(raw_trait, "value")
            else (f"{raw_trait}" if raw_trait is not None else "unknown")
        )

        adj_raw = self.meta_.get("adjusted_for", "unknown")
        adj_str = (
            f"{adj_raw.value}" if hasattr(adj_raw, "value") else (f"{adj_raw}" if adj_raw is not None else "unknown")
        )

        agg_type = self.meta_.get("aggregation_type", "unknown")
        if hasattr(agg_type, "value"):
            agg_type = f"{agg_type.value}"
        else:
            agg_type = f"{agg_type}"

        method_str = f"{self._method_label.value}" if hasattr(self._method_label, "value") else f"{self._method_label}"

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=[f"{s.value}" if hasattr(s, "value") else f"{s}" for s in self.strata_],
            adjusted_for=adj_str,
            method=method_str,
            aggregation_type=agg_type,
            trait=trait_str,
        )

def _rbf_kernel(X, Z, var, length):
    deltaXsq = jnp.power((X[:, None] - Z) / length, 2.0)
    sqdist = jnp.sum(deltaXsq, axis=-1)
    return var * jnp.exp(-0.5 * sqdist)

def _compute_spatial_posterior(var_samples, length_samples, alpha_samples, f_samples, X_train, X_test):
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    def _predict_single(v, l, a, f):
        K_train = _rbf_kernel(X_train, X_train, v, l) + jnp.eye(X_train.shape[0]) * 1e-4
        K_cross = _rbf_kernel(X_test, X_train, v, l)
        
        L = jsp.linalg.cho_factor(K_train)
        f_test_mean = K_cross @ jsp.linalg.cho_solve(L, f)
        
        logit_p = a + f_test_mean
        return jsp.special.expit(logit_p)
        
    p_samples = jax.vmap(_predict_single)(var_samples, length_samples, alpha_samples, f_samples)
    return np.mean(p_samples, axis=0), np.percentile(p_samples, 2.5, axis=0), np.percentile(p_samples, 97.5, axis=0)

class SpatialPrevalenceEstimator(ModelledMixin, BayesianMixin, BaseEstimator[PrevalenceEstimates]):
    """Gaussian Process (GP) based spatial prevalence estimator."""

    def __init__(  # noqa: ANN204, D107
        self,
        lat_col: str | StrEnum = "lat",
        lon_col: str | StrEnum = "lon",
        method: BayesianInferenceMethod = BayesianInferenceMethod.MCMC,
        num_samples: int = 1500,
        num_chains: int = 4,
        num_warmup: int = 1000,
        svi_steps: int = 3000,
        target_event: str | StrEnum = "event",
        target_n: str | StrEnum = "n",
        seed: int = 42,
    ):
        self._init_bayesian(method, num_samples, num_chains, num_warmup, svi_steps, seed)
        self.lat_col = str(lat_col.value) if hasattr(lat_col, "value") else str(lat_col)
        self.lon_col = str(lon_col.value) if hasattr(lon_col, "value") else str(lon_col)
        self._method_label = f"spatial_gp_{self.method.value}"
        self._svi_return_sites = ["alpha", "var", "length", "f", "obs"]

        self.target_event = str(target_event.value) if hasattr(target_event, "value") else str(target_event)
        self.target_n = str(target_n.value) if hasattr(target_n, "value") else str(target_n)

        self.X_train_ = None
        self.loc_mean_ = None
        self.loc_scale_ = None
        self.meta_ = {}

    def get_params(self) -> dict[str, Any]:
        """Returns parameters for cloning compatibility."""
        return {
            "lat_col": self.lat_col,
            "lon_col": self.lon_col,
            "method": self.method,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "num_warmup": self.num_warmup,
            "svi_steps": self.svi_steps,
            "target_event": self.target_event,
            "target_n": self.target_n,
            "seed": self.seed,
        }

    def _model(self, X, n, event=None):  # noqa: ANN001, ANN202
        alpha = samp("alpha", dist.Normal(0, 1.5))
        var = samp("var", dist.HalfNormal(1.0))
        length = samp("length", dist.InverseGamma(2.0, 1.0))
        K = _rbf_kernel(X, X, var, length)
        f = samp("f", dist.MultivariateNormal(loc=jnp.zeros(X.shape[0]), covariance_matrix=K))
        logit_p = alpha + f
        samp("obs", dist.Binomial(total_count=n, logits=logit_p), obs=event)

    def fit(self, agg_df: Any) -> "SpatialPrevalenceEstimator":  # type: ignore # noqa: ANN401, D102
        self._check_zero_padding(agg_df)
        df = agg_df.data if hasattr(agg_df, "data") else agg_df

        if self.lat_col not in df.columns or self.lon_col not in df.columns:
            raise KeyError(
                f"Spatial estimator requires '{self.lat_col}' and '{self.lon_col}' "
                "in the aggregated data. Please ensure you included them in the 'Stratify By' dropdown."
            )

        spatial_df = df.group_by([self.lat_col, self.lon_col]).agg(
            [pl.col(self.target_event).sum(), pl.col(self.target_n).sum()]
        )

        raw_coords = spatial_df.select([self.lat_col, self.lon_col]).to_numpy().astype(float)
        self.loc_mean_ = np.mean(raw_coords, axis=0)
        self.loc_scale_ = np.std(raw_coords, axis=0) + 1e-8

        self.X_train_ = (raw_coords - self.loc_mean_) / self.loc_scale_

        jax_data = {
            "X": jnp.array(self.X_train_, dtype=jnp.float32),
            "n": jnp.array(spatial_df[self.target_n].cast(pl.Int32).to_numpy()),
            "event": jnp.array(spatial_df[self.target_event].cast(pl.Int32).to_numpy()),
        }

        rng_key = random.PRNGKey(self.seed)
        self.samples_ = self._run_inference(jax_data, rng_key)

        if hasattr(agg_df, "metadata") and isinstance(agg_df.metadata, dict):
            self.meta_ = agg_df.metadata.get("metric_meta", agg_df.metadata)
        elif isinstance(getattr(agg_df, "metadata", None), dict):
            self.meta_ = getattr(agg_df, "metadata").get("metric_meta", getattr(agg_df, "metadata"))
        else:
            self.meta_ = {}

        raw_trait = self.meta_.get("trait")
        if (not raw_trait or raw_trait == "unknown") and "target" in df.columns and len(df) > 0:
            if df["target"].n_unique() == 1:
                self.meta_["trait"] = df["target"][0]
            else:
                self.meta_["trait"] = "target"

        self.is_fitted_ = True
        return self

    def predict(self, df: Any) -> PrevalenceEstimates:  # noqa: ANN401, D102
        self.check_is_fitted()
        df_in = df.data if hasattr(df, "data") else df

        raw_X_test = df_in.select([self.lat_col, self.lon_col]).to_numpy().astype(float)
        X_test = jnp.array((raw_X_test - self.loc_mean_) / self.loc_scale_, dtype=jnp.float32)
        X_train = jnp.array(self.X_train_, dtype=jnp.float32)

        assert self.samples_ is not None
        var_samples = self.samples_.get("var", self.samples_.get("auto_var"))
        length_samples = self.samples_.get("length", self.samples_.get("auto_length"))
        alpha_samples = self.samples_.get("alpha", self.samples_.get("auto_alpha"))
        f_samples = self.samples_.get("f", self.samples_.get("auto_f"))

        if var_samples is None or length_samples is None or alpha_samples is None or f_samples is None:
            raise KeyError("Spatial posterior prediction requires 'var', 'length', 'alpha', and 'f' samples.")

        estimate, lower, upper = _compute_spatial_posterior(
            var_samples, length_samples, alpha_samples, f_samples, X_train, X_test
        )

        result_df = df_in.with_columns(
            estimate=pl.Series(np.array(estimate)), lower=pl.Series(np.array(lower)), upper=pl.Series(np.array(upper))
        )

        raw_trait = self.meta_.get("trait", "unknown")
        if (not raw_trait or raw_trait == "unknown") and "target" in df_in.columns and len(df_in) > 0:
            if df_in["target"].n_unique() == 1:
                raw_trait = df_in["target"][0]
            else:
                raw_trait = "target"
        trait_str = (
            f"{raw_trait.value}"
            if hasattr(raw_trait, "value")
            else (f"{raw_trait}" if raw_trait is not None else "unknown")
        )

        adj_raw = self.meta_.get("adjusted_for", "unknown")
        adj_str = (
            f"{adj_raw.value}" if hasattr(adj_raw, "value") else (f"{adj_raw}" if adj_raw is not None else "unknown")
        )

        agg_type = self.meta_.get("aggregation_type", "unknown")
        if hasattr(agg_type, "value"):
            agg_type = f"{agg_type.value}"
        else:
            agg_type = f"{agg_type}"

        method_str = f"{self._method_label.value}" if hasattr(self._method_label, "value") else f"{self._method_label}"

        return PrevalenceEstimates(
            data=result_df,
            stratified_by=[
                f"{self.lat_col.value}" if hasattr(self.lat_col, "value") else f"{self.lat_col}",
                f"{self.lon_col.value}" if hasattr(self.lon_col, "value") else f"{self.lon_col}",
            ],
            adjusted_for=adj_str,
            method=method_str,
            aggregation_type=agg_type,
            trait=trait_str,
        )

