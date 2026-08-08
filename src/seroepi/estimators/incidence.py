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
from seroepi.estimators.base import BaseEstimator, IncidenceEstimates, PrevalenceEstimates, ReproductionEstimates, BaseStatefulEstimator

def _calculate_time_steps(dates: pl.Series, min_date: datetime.date, freq: str) -> np.ndarray:
    freq_str = str(freq.value) if hasattr(freq, "value") else str(freq)
    if freq_str == "year":
        return (dates.dt.year() - min_date.year).to_numpy()
    elif freq_str == "month":
        return ((dates.dt.year() - min_date.year) * 12 + (dates.dt.month() - min_date.month)).to_numpy()
    elif freq_str == "week":
        return ((dates - min_date).dt.total_days() // 7).to_numpy()
    else:
        return (dates - min_date).dt.total_days().to_numpy()

def _generate_future_dates(last_date: datetime.date, steps: int, freq: str) -> list[datetime.date]:
    future_dates = []
    freq_str = str(freq.value) if hasattr(freq, "value") else str(freq)
    for i in range(1, steps + 1):
        if freq_str == "year":
            future_dates.append(last_date + relativedelta(years=i))
        elif freq_str == "month":
            future_dates.append(last_date + relativedelta(months=i))
        elif freq_str == "week":
            future_dates.append(last_date + timedelta(weeks=i))
        else:
            future_dates.append(last_date + timedelta(days=i))
    return future_dates

# Set-up ---------------------------------------------------------------------------------------------------------------
set_host_device_count(cpu_count() or 1)


# TypeVars -------------------------------------------------------------------------------------------------------------
T_Modelled = TypeVar("T_Modelled", bound="ModelledMixin")


# Mixins ---------------------------------------------------------------------------------------------------------------
from seroepi.estimators.mixins import ModelledMixin, BayesianMixin

class GLMIncidenceEstimator(ModelledMixin, BaseEstimator[IncidenceEstimates]):
    """Negative Binomial GLM for time-series incidence estimation."""

    def __init__(self, use_relative_incidence: bool = True, forecast_horizon: int = 0):  # noqa: ANN204, D107
        self.use_relative_incidence = use_relative_incidence
        self.forecast_horizon = forecast_horizon
        self._method_label = "neg_binomial_glm"

        self.fit_results_ = {}
        self.strata_ = []
        self.meta_ = {}

    def get_params(self) -> dict[str, Any]:
        """Returns parameters for cloning compatibility."""
        return {
            "use_relative_incidence": self.use_relative_incidence,
            "forecast_horizon": self.forecast_horizon,
        }

    def fit(self, inc_df: Any) -> "GLMIncidenceEstimator":  # type: ignore # noqa: ANN401, D102
        if hasattr(inc_df, "data") and hasattr(inc_df, "metadata"):
            df = inc_df.data
            self.meta_ = inc_df.metadata.get("metric_meta", {}) if isinstance(inc_df.metadata, dict) else {}
        elif isinstance(inc_df, pl.DataFrame):
            df = inc_df
            self.meta_ = (
                getattr(inc_df, "metadata", {}).get("metric_meta", {})
                if isinstance(getattr(inc_df, "metadata", None), dict)
                else {}
            )
        else:
            df = getattr(inc_df, "data", inc_df)
            self.meta_ = getattr(inc_df, "metadata", getattr(inc_df, "attrs", {}))
            if not isinstance(self.meta_, dict):
                self.meta_ = {}
            self.meta_ = (
                self.meta_.get("metric_meta", {})
                if isinstance(self.meta_.get("metric_meta", None), dict)
                else self.meta_
            )

        target_col = self.meta_.get("trait")
        if not target_col or target_col == "unknown":
            if "target" in df.columns and len(df) > 0:
                if df["target"].n_unique() == 1:
                    target_col = df["target"][0]
                else:
                    target_col = "target"
            else:
                exclude = {"date", "variant_count", "total_sequenced", "n", "event"}
                non_count = [c for c in df.columns if c not in exclude]
                target_col = non_count[0] if non_count else "unknown"
            self.meta_["trait"] = target_col

        self.freq_ = self.meta_.get("freq")
        if not self.freq_:
            self.freq_ = TemporalResolution.MONTH.value
            self.meta_["freq"] = self.freq_

        self.strata_ = self.meta_.get("stratified_by", [])
        if not self.strata_:
            exclude = {"date", "variant_count", "total_sequenced", "target", "n", "event"}
            self.strata_ = [c for c in df.columns if c not in exclude]
            self.meta_["stratified_by"] = self.strata_

        inc_df_sorted = df.sort("date")
        group_cols = self.strata_ + ["target"] if "target" in inc_df_sorted.columns else self.strata_

        if group_cols:
            groups = inc_df_sorted.partition_by(group_cols, as_dict=True)
        else:
            groups = {(("Global",),): inc_df_sorted}

        for name, group in groups.items():
            name_tuple = name if isinstance(name, tuple) else (name,)
            df_group = group
            df_model = df_group.filter(pl.col("total_sequenced") > 0)

            if len(df_model) < 3:
                self.fit_results_[name_tuple] = None
                continue

            Y = df_model["variant_count"].to_numpy()

            if Y.sum() == 0:
                self.fit_results_[name_tuple] = None
                continue

            min_date = df_model["date"].min()
            time_steps = _calculate_time_steps(df_model["date"], min_date, self.freq_)
            df_model = df_model.with_columns(time_step=pl.Series(time_steps))

            X = sm.add_constant(time_steps)
            offset = np.log(df_model["total_sequenced"].to_numpy()) if self.use_relative_incidence else None

            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = sm.GLM(Y, X, family=sm.families.NegativeBinomial(alpha=1.0), offset=offset)
                    self.fit_results_[name_tuple] = model.fit()
            except Exception as e:
                warn(f"GLM failed to converge for stratum {name_tuple}: {e}")
                self.fit_results_[name_tuple] = None

        self.is_fitted_ = True
        return self

    def predict(self, df: Any) -> IncidenceEstimates:  # noqa: ANN401, D102
        self.check_is_fitted()
        df_data = df.data if hasattr(df, "data") else df

        group_cols = self.strata_ + ["target"] if "target" in df_data.columns else self.strata_
        df_sorted = df_data.sort("date")

        if group_cols:
            groups = df_sorted.partition_by(group_cols, as_dict=True)
        else:
            groups = {(("Global",),): df_sorted}

        results = []
        all_pred_dfs = []

        for name_tuple, group in groups.items():
            name_key = name_tuple if isinstance(name_tuple, tuple) else (name_tuple,)
            fit = self.fit_results_.get(name_key)

            row = {}
            if group_cols:
                for col_name, val in zip(group_cols, name_key):
                    row[col_name] = val

            if fit is None:
                row.update(
                    {
                        "IRR": np.nan,
                        "IRR_lower": np.nan,
                        "IRR_upper": np.nan,
                        "p_value": np.nan,
                        "status": "Failed/Insufficient Data",
                    }
                )

                df_pred = group.with_columns(
                    estimate=pl.Series([np.nan] * len(group)),
                    lower=pl.Series([np.nan] * len(group)),
                    upper=pl.Series([np.nan] * len(group)),
                )
                all_pred_dfs.append(df_pred)
            else:
                params = fit.params
                pvalues = fit.pvalues
                ci = fit.conf_int()
                idx = 1 if len(params) > 1 else 0

                coef = float(params[idx])
                p_val = float(pvalues[idx])
                ci_lower = float(ci[idx, 0])
                ci_upper = float(ci[idx, 1])

                row.update(
                    {
                        "IRR": float(np.exp(coef)),
                        "IRR_lower": float(np.exp(ci_lower)),
                        "IRR_upper": float(np.exp(ci_upper)),
                        "p_value": float(p_val),
                        "status": "Converged",
                    }
                )

                df_pred = group
                min_date = df_pred["date"].min()

                if getattr(self, "forecast_horizon", 0) > 0:
                    max_date = df_pred["date"].max()
                    future_dates = _generate_future_dates(max_date, self.forecast_horizon, str(self.freq_))

                    future_dict = {"date": future_dates}
                    if group_cols:
                        for col_name, val in zip(group_cols, name_key):
                            future_dict[col_name] = [val] * len(future_dates)

                    mean_seq = float(df_pred["total_sequenced"].mean()) if "total_sequenced" in df_pred.columns else 1.0
                    seq_dtype = df_pred["total_sequenced"].dtype if "total_sequenced" in df_pred.columns else pl.Float64
                    future_dict["total_sequenced"] = [mean_seq] * len(future_dates)
                    future_dict["variant_count"] = [0] * len(future_dates)

                    future_df = pl.DataFrame(future_dict).with_columns(pl.col("total_sequenced").cast(seq_dtype))
                    df_pred = pl.concat([df_pred, future_df], how="diagonal_relaxed")

                time_steps = _calculate_time_steps(df_pred["date"], min_date, str(self.freq_))
                df_pred = df_pred.with_columns(time_step=pl.Series(time_steps))

                X = sm.add_constant(time_steps, has_constant="add")
                tot_seq = (
                    df_pred["total_sequenced"].to_numpy()
                    if "total_sequenced" in df_pred.columns
                    else np.ones(len(df_pred))
                )
                offset = np.log(np.clip(tot_seq, 1e-8, None)) if self.use_relative_incidence else None

                pred_res = fit.get_prediction(X, offset=offset).summary_frame(alpha=0.05)
                df_pred = df_pred.with_columns(
                    estimate=pl.Series(pred_res["mean"].values),
                    lower=pl.Series(pred_res["mean_ci_lower"].values),
                    upper=pl.Series(pred_res["mean_ci_upper"].values),
                )

                all_pred_dfs.append(df_pred)

            results.append(row)

        final_df = pl.concat(all_pred_dfs, how="diagonal_relaxed") if all_pred_dfs else df

        raw_trait = self.meta_.get("trait", "unknown")
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

        freq_raw = self.freq_ or self.meta_.get("freq", "unknown")
        freq_str = f"{freq_raw.value}" if hasattr(freq_raw, "value") else f"{freq_raw}"

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

        return IncidenceEstimates(
            data=final_df,
            stratified_by=[f"{s.value}" if hasattr(s, "value") else f"{s}" for s in self.strata_],
            adjusted_for=adj_str,
            trait=trait_str,
            freq=freq_str,
            aggregation_type=agg_type,
            model_results=pl.DataFrame(results),
            method=method_str,
        )

class BayesianIncidenceEstimator(ModelledMixin, BayesianMixin, BaseEstimator[IncidenceEstimates]):
    """Bayesian Structural Time Series (BSTS) for incidence forecasting."""

    def __init__(  # noqa: ANN204, D107
        self,
        forecast_horizon: int = 12,
        method: BayesianInferenceMethod = BayesianInferenceMethod.MCMC,
        num_samples: int = 1500,
        num_chains: int = 4,
        num_warmup: int = 1000,
        svi_steps: int = 3000,
        seed: int = 42,
    ):
        self.forecast_horizon = forecast_horizon
        self._init_bayesian(method, num_samples, num_chains, num_warmup, svi_steps, seed)
        self._method_label = f"bsts_forecast_{self.method.value}"
        self._svi_return_sites = ["mu_0", "drift", "sigma_rw", "dispersion", "innovations"]

        self.strata_ = []
        self.meta_ = {}

    def get_params(self) -> dict[str, Any]:
        """Returns parameters for cloning compatibility."""
        return {
            "forecast_horizon": self.forecast_horizon,
            "method": self.method,
            "num_samples": self.num_samples,
            "num_chains": self.num_chains,
            "num_warmup": self.num_warmup,
            "svi_steps": self.svi_steps,
            "seed": self.seed,
        }

    def _model(self, T, n_strata, Y=None, forecast_horizon=0):  # noqa: ANN001, ANN202
        total_T = T + forecast_horizon
        with plate("strata", n_strata, dim=-1):
            mu_0 = samp("mu_0", dist.Normal(0, 2))
            drift = samp("drift", dist.Normal(0, 0.5))
            sigma_rw = samp("sigma_rw", dist.HalfNormal(0.5))
            dispersion = samp("dispersion", dist.HalfNormal(2))

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

        rw = jnp.cumsum(innovations * sigma_rw, axis=-2)
        time_steps = jnp.arange(total_T)[:, None]
        log_rate = mu_0 + (time_steps * drift) + rw

        if Y is not None:
            historical_rate = log_rate[:T]
            with plate("obs_time", T, dim=-2):
                with plate("obs_strata", n_strata, dim=-1):
                    samp("obs", dist.NegativeBinomial2(mean=jnp.exp(historical_rate), concentration=dispersion), obs=Y)
        else:
            with plate("obs_time", total_T, dim=-2):
                with plate("obs_strata", n_strata, dim=-1):
                    samp("obs", dist.NegativeBinomial2(mean=jnp.exp(log_rate), concentration=dispersion))

    def fit(self, inc_df: Any) -> "BayesianIncidenceEstimator":  # type: ignore # noqa: ANN401, D102
        df = inc_df.data if hasattr(inc_df, "data") else inc_df
        if hasattr(inc_df, "metadata"):
            self.meta_ = inc_df.metadata.get("metric_meta", {}) if isinstance(inc_df.metadata, dict) else {}
        else:
            self.meta_ = (
                getattr(inc_df, "metadata", getattr(inc_df, "attrs", {})).get("metric_meta", {})
                if isinstance(getattr(inc_df, "metadata", None), dict)
                else {}
            )

        if not self.meta_.get("trait"):
            if "target" in df.columns and len(df) > 0 and df["target"].n_unique() == 1:
                self.meta_["trait"] = df["target"][0]
            elif "target" in df.columns:
                self.meta_["trait"] = "target"
            else:
                exclude = {"date", "variant_count", "total_sequenced", "n", "event"}
                non_count = [c for c in df.columns if c not in exclude]
                self.meta_["trait"] = non_count[0] if non_count else "unknown"

        if not self.meta_.get("freq"):
            self.meta_["freq"] = TemporalResolution.MONTH.value
        freq_raw = self.meta_["freq"]
        self.freq_ = f"{freq_raw.value}" if hasattr(freq_raw, "value") else f"{freq_raw}"

        self.strata_ = self.meta_.get("stratified_by", [])
        if not self.strata_:
            exclude = {"date", "variant_count", "total_sequenced", "target", "n", "event"}
            self.strata_ = [c for c in df.columns if c not in exclude]
            self.meta_["stratified_by"] = self.strata_

        group_cols = self.strata_ + ["target"] if "target" in df.columns else self.strata_
        if not group_cols:
            df = df.with_columns(_dummy_group=pl.lit("Global"))
            group_cols = ["_dummy_group"]

        pivot_df = (
            df.pivot(on=group_cols, index="date", values="variant_count", aggregate_function="sum")
            .fill_null(0)
            .sort("date")
        )

        target_cols = [c for c in pivot_df.columns if c != "date"]
        strata_sums = {col: pivot_df[col].sum() for col in target_cols}
        active_cols = [col for col, s in strata_sums.items() if s > 0]
        inactive_cols = [col for col, s in strata_sums.items() if s == 0]

        if not active_cols:
            raise ValueError("All strata have zero historical events. Cannot fit the Bayesian model.")

        self.inactive_strata_labels_ = inactive_cols
        self.dates_ = pivot_df["date"].to_list()
        self.T_ = len(self.dates_)
        self.n_strata_ = len(active_cols)
        self.strata_labels_ = active_cols

        jax_data = {
            "T": self.T_,
            "n_strata": self.n_strata_,
            "Y": jnp.array(pivot_df.select(active_cols).to_numpy(dtype=np.int32)),
            "forecast_horizon": 0,
        }

        rng_key = random.PRNGKey(self.seed)
        self.samples_ = self._run_inference(jax_data, rng_key)
        self.is_fitted_ = True
        return self

    def predict(self, df: Any) -> IncidenceEstimates:  # noqa: ANN401, D102
        self.check_is_fitted()
        df_data = df.data if hasattr(df, "data") else df

        assert self.samples_ is not None
        posterior_samples = {k: v for k, v in self.samples_.items() if k != "obs"}
        pred = Predictive(self._model, posterior_samples)
        pred_key = random.PRNGKey(self.seed + 1)

        pred_samples = pred(
            pred_key, T=self.T_, n_strata=self.n_strata_, Y=None, forecast_horizon=self.forecast_horizon
        )

        assert self.samples_ is not None
        estimate, lower, upper, drifts, p_vals = _summarize_incidence_posterior(
            pred_samples["obs"], self.samples_["drift"]
        )

        freq_raw = self.meta_.get("freq", TemporalResolution.MONTH.value)
        freq_str = str(freq_raw.value) if hasattr(freq_raw, "value") else str(freq_raw)

        future_dates = _generate_future_dates(self.dates_[-1], self.forecast_horizon, freq_str)
        all_dates = list(self.dates_) + future_dates

        results = []
        group_cols = self.strata_ + ["target"] if "target" in df_data.columns else self.strata_
        if not group_cols:
            group_cols = ["_dummy_group"]

        for i, dt_val in enumerate(all_dates):
            for j, strata_val in enumerate(self.strata_labels_):
                row = {"date": dt_val}
                row[group_cols[0] if len(group_cols) == 1 else "target"] = strata_val
                row["estimate"] = float(estimate[i, j])
                row["lower"] = float(lower[i, j])
                row["upper"] = float(upper[i, j])
                results.append(row)

            for strata_val in self.inactive_strata_labels_:
                row = {"date": dt_val}
                row[group_cols[0] if len(group_cols) == 1 else "target"] = strata_val
                row["estimate"] = 0.0
                row["lower"] = 0.0
                row["upper"] = 0.0
                results.append(row)

        res_df = pl.DataFrame(results)
        merge_cols = ["date"] + [c for c in group_cols if c in res_df.columns and c in df_data.columns]
        final_df = res_df.join(df_data, on=merge_cols, how="left")

        model_res = []
        for j, strata_val in enumerate(self.strata_labels_):
            row = {group_cols[0] if len(group_cols) == 1 else "target": strata_val}
            row["IRR"] = float(jnp.exp(drifts[j]))
            row["prob_increasing"] = float(p_vals[j])
            row["status"] = "Converged"
            model_res.append(row)

        for strata_val in self.inactive_strata_labels_:
            row = {group_cols[0] if len(group_cols) == 1 else "target": strata_val}
            row["IRR"] = np.nan
            row["prob_increasing"] = np.nan
            row["status"] = "Zero Events"
            model_res.append(row)

        raw_trait = self.meta_.get("trait", "unknown")
        if (not raw_trait or raw_trait == "unknown") and "target" in df_data.columns and len(df_data) > 0:
            if df_data["target"].n_unique() == 1:
                raw_trait = df_data["target"][0]
            else:
                raw_trait = "target"
        trait_str = (
            f"{raw_trait.value}"
            if hasattr(raw_trait, "value")
            else (f"{raw_trait}" if raw_trait is not None else "unknown")
        )

        freq_raw = getattr(self, "freq_", None) or self.meta_.get("freq", "unknown")
        freq_str = f"{freq_raw.value}" if hasattr(freq_raw, "value") else f"{freq_raw}"

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

        return IncidenceEstimates(
            data=final_df,
            stratified_by=[f"{s.value}" if hasattr(s, "value") else f"{s}" for s in self.strata_],
            adjusted_for=adj_str,
            trait=trait_str,
            freq=freq_str,
            aggregation_type=agg_type,
            model_results=pl.DataFrame(model_res),
            method=method_str,
        )


class ReproductionNumberEstimator(ModelledMixin, BayesianMixin, BaseStatefulEstimator[ReproductionEstimates]):
    """Estimates time-varying effective reproduction number R_e(t) from incidence data.
    
    Uses a Bayesian renewal equation approach.
    """
    _result_class = ReproductionEstimates
    
    def __init__(
        self, 
        time_column: str, 
        case_column: str, 
        generation_time_mean: float, 
        generation_time_std: float,
        method: BayesianInferenceMethod | str = BayesianInferenceMethod.MCMC,
        num_samples: int = 1000,
        num_chains: int = 4,
        num_warmup: int = 500,
        svi_steps: int = 2000,
        seed: int = 0,
        **kwargs: Any
    ):
        self.time_column = time_column
        self.case_column = case_column
        self.generation_time_mean = generation_time_mean
        self.generation_time_std = generation_time_std
        self._init_bayesian(
            method=method,
            num_samples=num_samples,
            num_chains=num_chains,
            num_warmup=num_warmup,
            svi_steps=svi_steps,
            seed=seed
        )
        
    def fit(self, *args, **kwargs) -> Self:  # type: ignore
        self.is_fitted_ = True
        return self
        
    def _model(self, cases: jnp.ndarray, weights: jnp.ndarray) -> None:
        import numpyro.distributions as dist
        from numpyro import sample, plate
        
        T = cases.shape[0]
        
        # Random walk for log(R_t)
        sigma = sample("sigma", dist.HalfNormal(0.1))
        log_R = sample("log_R", dist.GaussianRandomWalk(scale=sigma, num_steps=T))
        
        R = jnp.exp(log_R)
        
        # Calculate expected cases: sum_{s=1}^{t-1} I_{t-s} w_s
        idx = jnp.arange(T)
        dt = idx[:, None] - idx[None, :]
        valid = dt > 0
        W_matrix = jnp.where(valid, weights[dt - 1], 0.0)
        
        lambda_t = R * (W_matrix @ cases)
        lambda_t = jnp.where(lambda_t < 1e-5, 1e-5, lambda_t)
        
        # Sample cases
        with plate("obs_plate", T, dim=-1):
            sample("obs", dist.Poisson(lambda_t), obs=cases)

    def _generate_predictions(self, df: pl.DataFrame) -> pl.DataFrame:
        import scipy.stats as sp_stats
        
        df = df.sort(self.time_column)
        # Ensure strict 32-bit casting for JAX/numpyro performance
        cases = df[self.case_column].to_numpy().astype(np.float32)
        T = len(cases)
        
        # 1. Precalculate generation time weights (Gamma distribution)
        shape = (self.generation_time_mean ** 2) / (self.generation_time_std ** 2)
        scale = (self.generation_time_std ** 2) / self.generation_time_mean
        
        lags = np.arange(1, T + 1)
        cdf = sp_stats.gamma.cdf(lags, a=shape, scale=scale)
        cdf_prev = np.concatenate(([0.0], cdf[:-1]))
        weights = (cdf - cdf_prev).astype(np.float32)
        weights = weights / weights.sum() 
        
        # 2. Run Numpyro inference
        self.meta_ = {"trait": self.case_column, "aggregation_type": "incidence"}
        self.strata_ = []
        
        from jax import random
        rng_key = random.PRNGKey(self.seed)
        jax_data = {"cases": jnp.array(cases), "weights": jnp.array(weights)}
        self.samples_ = self._run_inference(jax_data, rng_key)
        
        # 3. Extract R_t
        R_samples = jnp.exp(self.samples_["log_R"]) 
        
        R_mean = np.asarray(np.mean(R_samples, axis=0))
        R_lower = np.asarray(np.percentile(R_samples, 2.5, axis=0))
        R_upper = np.asarray(np.percentile(R_samples, 97.5, axis=0))
        
        return df.with_columns(
            Re_mean=pl.Series(R_mean),
            Re_lower=pl.Series(R_lower),
            Re_upper=pl.Series(R_upper)
        )

    def predict(self, agg_df: Any) -> ReproductionEstimates:
        """Universal predict wrapper that formats metadata automatically."""
        if hasattr(self, "check_is_fitted"):
            self.check_is_fitted()
            
        df = agg_df.data if hasattr(agg_df, "data") else agg_df
        result_df = self._generate_predictions(df)
        meta = getattr(self, "meta_", {})
        strata = getattr(self, "strata_", [])
        
        return ReproductionEstimates(
            data=result_df,
            stratified_by=[str(s) for s in strata],
            adjusted_for=str(meta.get("adjusted_for", "None")),
            trait=str(meta.get("trait", "unknown")),
            aggregation_type=str(meta.get("aggregation_type", "unknown")),
            time_column=self.time_column,
            generation_time_mean=self.generation_time_mean
        )

