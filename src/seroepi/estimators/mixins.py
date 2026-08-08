from abc import ABC, abstractmethod  # noqa: D100
from datetime import date, datetime, timedelta
from enum import StrEnum
from functools import partial
from os import cpu_count
from pathlib import Path
from typing import Any, Self, TypeVar
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

from seroepi.constants import BayesianInferenceMethod, TemporalResolution
from seroepi.domains.base import get_dataframe_metadata
from seroepi.estimators.base import BaseEstimator, IncidenceEstimates, PrevalenceEstimates

# Set-up ---------------------------------------------------------------------------------------------------------------
set_host_device_count(cpu_count() or 1)


# TypeVars -------------------------------------------------------------------------------------------------------------
T_Modelled = TypeVar("T_Modelled", bound="ModelledMixin")


# Mixins ---------------------------------------------------------------------------------------------------------------

class ModelledMixin(ABC):
    """Contract for estimators with an internal fitted state.

    Enforces the scikit-learn fit/predict paradigm and provides universal
    serialization for fitted models.

    Attributes:
        is_fitted_: Boolean indicating if the model has been fitted.
    """

    # State tracking
    is_fitted_: bool = False

    def check_is_fitted(self):  # noqa: ANN201
        """Checks if the model is fitted.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if not self.is_fitted_:
            raise RuntimeError(
                "This estimator instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

    @abstractmethod
    def fit(self, df: Any) -> "ModelledMixin":  # noqa: ANN401
        """Calculates the internal state (e.g., MCMC samples) and saves it to self."""
        pass

    @abstractmethod
    def predict(self, df: Any):  # noqa: ANN201, ANN401
        """Uses the fitted internal state to generate predictions on the dataframe."""
        pass

    def calculate(self, df: Any):  # noqa: ANN201, ANN401
        """One-liner to fit and predict on the same data."""
        return self.fit(df).predict(df)

    def save_model(self, filepath: str | Path) -> None:
        """Universally serializes the fitted estimator instance to disk.

        Args:
            filepath: Path where the model should be saved.
        """
        if not self.is_fitted_:
            warn(f"You are saving a {self.__class__.__name__} that hasn't been fitted yet.")

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib_dump(self, path)

    @classmethod
    def load_model(cls, filepath: str | Path) -> Self:
        """Loads a serialized estimator from disk.

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
    """Shared inference logic for NumPyro-based Bayesian estimators."""

    _model: Any = None

    def _init_bayesian(  # noqa: ANN202
        self,
        method: BayesianInferenceMethod,
        num_samples: int,
        num_chains: int,
        num_warmup: int,
        svi_steps: int,
        seed: int,
    ):
        self.method = BayesianInferenceMethod(method) if isinstance(method, str) else method
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_warmup = num_warmup
        self.svi_steps = svi_steps
        self.seed = seed
        self.samples_ = None
        self.extra_fields_ = None

    def _check_zero_padding(self, df: Any):  # noqa: ANN202, ANN401
        """Ensures the incoming dataframe has been properly padded for Bayesian inference."""
        raw_meta = (
            getattr(df, "metadata", getattr(df, "meta", None))
            or (df.metadata if hasattr(df, "metadata") else {})
            or get_dataframe_metadata(df)
            or {}
        )
        meta = raw_meta.get("metric_meta", raw_meta) if isinstance(raw_meta, dict) else {}

        if "is_zero_padded" in meta and not meta["is_zero_padded"]:
            raise ValueError(
                f"Mathematical Integrity Error: {self.__class__.__name__} requires a strictly rectangular, "
                "zero-padded matrix to construct the posterior geometry. "
                "Please regenerate the dataset using `.epi.aggregate_...(pad_zeros=True)`."
            )

    def _run_inference(self, jax_data: dict[str, Any], rng_key: Any):  # noqa: ANN202, ANN401
        """Routes to the correct inference engine based on self.method."""
        if self.method == BayesianInferenceMethod.MCMC:
            return self._mcmc_inference(jax_data, rng_key)
        elif self.method == BayesianInferenceMethod.SVI:
            return self._svi_inference(jax_data, rng_key)
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose from MCMC or SVI.")

    def _mcmc_inference(self, jax_data: dict[str, Any], rng_key: Any):  # noqa: ANN202, ANN401
        """Runs MCMC inference using NUTS."""
        mcmc = MCMC(
            NUTS(self._model),
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=False,
            chain_method="vectorized",
        )
        mcmc.run(rng_key, **jax_data)
        self.extra_fields_ = mcmc.get_extra_fields()
        return mcmc.get_samples()

    def _svi_inference(self, jax_data: dict[str, Any], rng_key: Any):  # noqa: ANN202, ANN401
        """Runs Stochastic Variational Inference."""
        opt_key, pred_key = random.split(rng_key)
        guide = autoguide.AutoNormal(self._model)
        optimizer = optim.Adam(step_size=0.01)
        svi = SVI(self._model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(opt_key, num_steps=self.svi_steps, **jax_data)
        return_sites = getattr(self, "_svi_return_sites", None)
        predictive = Predictive(
            self._model,
            guide=guide,
            params=svi_result.params,
            num_samples=self.num_samples,
            return_sites=return_sites,
        )
        return predictive(pred_key, **jax_data)

    def diagnostics(self) -> pl.DataFrame:
        """Returns MCMC diagnostics (R-hat, ESS) as a formatted Polars DataFrame."""
        if hasattr(self, "check_is_fitted"):
            self.check_is_fitted()  # type: ignore
        if self.method != BayesianInferenceMethod.MCMC:
            raise TypeError("Diagnostics are only available for MCMC inference.")

        summary_dict = diag.summary(self.samples_, prob=0.95, group_by_chain=False)  # type: ignore

        rows = []
        for param, stats in summary_dict.items():
            param_shape = np.shape(stats["mean"])

            if len(param_shape) == 0:
                row = {"Parameter": param}
                row.update({k: float(v) for k, v in stats.items()})
                rows.append(row)
            else:
                it = np.nditer(np.empty(param_shape), flags=["multi_index"])
                for _ in it:
                    idx = it.multi_index
                    idx_str = ",".join(map(str, idx))
                    row = {"Parameter": f"{param}[{idx_str}]"}
                    row.update({k: float(np.asarray(v)[idx]) for k, v in stats.items()})
                    rows.append(row)

        if getattr(self, "extra_fields_", None) is not None:
            for field, values in (self.extra_fields_ or {}).items():
                val_array = np.asarray(values, dtype=float)
                rows.append(
                    {
                        "Parameter": f"sampler_{field}",
                        "mean": float(np.mean(val_array)),
                        "std": float(np.std(val_array)),
                        "sum": float(np.sum(val_array)),
                    }
                )

        return pl.DataFrame(rows)

