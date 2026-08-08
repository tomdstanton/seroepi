"""Epidemiological domain mixin and co-located plotters for temporal, stratification, aggregation, and prevalence visualizations."""  # noqa: E501

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from scipy.stats import norm

from seroepi import estimators
from seroepi.constants import AggregationType, Domain, PlotType, TemporalResolution
from seroepi.dist import TransmissionDistances
from seroepi.domains.base import BaseDomainMixin, BasePlotter, register_plotter
from seroepi.formulation import Formulation
from plotly.subplots import make_subplots
from seroepi.estimators.comparator import ComparisonResult


class EpiMixin(BaseDomainMixin):
    """Mixin class providing epidemiological dataset operations with __slots__ = ()."""

    __slots__ = ()

    @property
    def has_temporal(self) -> bool:
        """Checks if dataset contains valid non-null temporal data."""
        df = self._df
        cols = [
            c
            for c in df.columns
            if c.startswith(f"{Domain.TEMPORAL.value}_") and not c.startswith(f"{Domain.TEMPORAL_RES.value}_")
        ]
        if not cols:
            return False
        return df.select(pl.col(cols[0]).is_not_null().any()).item()

    @property
    def has_spatial(self) -> bool:
        """Checks if dataset contains valid non-null spatial coordinates."""
        df = self._df
        if "latitude" not in df.columns or "longitude" not in df.columns:
            return False
        return df.select(pl.col("latitude").is_not_null().any() & pl.col("longitude").is_not_null().any()).item()

    @property
    def temporal(self) -> pl.DataFrame:
        """Returns a DataFrame of all temporal columns, with prefix removed."""  # noqa: D421
        df = self._df
        prefix = f"{Domain.TEMPORAL.value}_"
        res_prefix = f"{Domain.TEMPORAL_RES.value}_"
        cols = [c for c in df.columns if c.startswith(prefix) and not c.startswith(res_prefix)]
        if not cols:
            return pl.DataFrame()
        return df.select(cols).rename({c: c.replace(prefix, "", 1) for c in cols})

    @property
    def temporal_resolution(self) -> pl.DataFrame:
        """Returns a DataFrame of all temporal resolution columns."""  # noqa: D421
        df = self._df
        prefix = f"{Domain.TEMPORAL_RES.value}_"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            return pl.DataFrame()
        return df.select(cols).rename({c: c.replace(prefix, "", 1) for c in cols})

    @property
    def spatial(self) -> pl.DataFrame:
        """Returns latitude and longitude columns as Float64."""  # noqa: D421
        if not self.has_spatial:
            raise ValueError("Spatial columns ('latitude', 'longitude') are missing.")
        return self._df.select([pl.col("latitude").cast(pl.Float64), pl.col("longitude").cast(pl.Float64)])

    def _resolve_temporal_col(self, col: str | StrEnum | None = None) -> str:
        """Resolves and validates the temporal datetime column."""
        df = self._df
        prefix = f"{Domain.TEMPORAL.value}_"
        res_prefix = f"{Domain.TEMPORAL_RES.value}_"

        if col is None:
            cols = [c for c in df.columns if c.startswith(prefix) and not c.startswith(res_prefix)]
            if not cols:
                if "Collection_Date" in df.columns:
                    col = "Collection_Date"
                else:
                    raise KeyError("A valid temporal datetime column is required.")
            else:
                col = cols[0]
        else:
            col_str = str(col)
            if col_str in df.columns:
                col = col_str
            elif col_str.startswith(prefix) and col_str[len(prefix) :] in df.columns:
                col = col_str[len(prefix) :]
            elif not col_str.startswith(prefix) and f"{prefix}{col_str}" in df.columns:
                col = f"{prefix}{col_str}"
            else:
                col = col_str

        if col not in df.columns:
            raise TypeError(f"Temporal column '{col}' not found. Ensure data is parsed via seroepi.io.")
        return col

    @staticmethod
    def _resolve_freq(freq: str | TemporalResolution) -> str:
        """Resolves frequency inputs into Polars interval strings."""
        if isinstance(freq, TemporalResolution):
            return freq.polars_interval
        if isinstance(freq, str):
            freq_str = freq.lower().strip()
            for member in TemporalResolution:
                if member != TemporalResolution.UNKNOWN and member.value == freq_str:
                    return member.polars_interval
            return freq
        return str(freq)

    def _get_spatiotemporal_arrays(
        self, temporal_col: str | StrEnum | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper to extract coordinates and dates for spatial clustering."""
        temporal_col_str = self._resolve_temporal_col(temporal_col)
        df = self._df
        lats = df.get_column("latitude").to_numpy().astype(float)
        lons = df.get_column("longitude").to_numpy().astype(float)
        coords = np.radians(np.column_stack([lats, lons]))

        dates_series = df.get_column(temporal_col_str)
        if dates_series.dtype in (pl.Datetime, pl.Date):
            null_mask = dates_series.is_null().to_numpy()
            raw_dates = dates_series.dt.date().to_numpy().astype("datetime64[D]").astype(float)
            raw_dates[null_mask] = np.nan
        else:
            raw_dates = np.full(len(df), np.nan)

        valid_mask = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]) | np.isnan(raw_dates))
        return coords, raw_dates, valid_mask

    def epidemic_curve(
        self,
        freq: str | TemporalResolution = TemporalResolution.WEEK,
        stratify_by: str | StrEnum | None = None,
        temporal_col: str | StrEnum | None = None,
    ) -> pl.DataFrame:
        """Generates time-series Polars DataFrame for plotting epidemic curves."""
        if not self.has_temporal:
            raise ValueError("Cannot generate epi curve: No temporal data available.")

        df = self._df
        interval = self._resolve_freq(freq)
        temporal_col_str = self._resolve_temporal_col(temporal_col)

        df_binned = df.with_columns(pl.col(temporal_col_str).dt.truncate(interval).alias("date"))

        stratify_by_str = str(stratify_by) if stratify_by is not None else None

        if stratify_by_str:
            groupers = ["date", stratify_by_str]
            counts = df_binned.group_by(groupers).agg(pl.len().alias("count"))
            curve = counts.pivot(on=stratify_by_str, index="date", values="count").fill_null(0).sort("date")
        else:
            curve = df_binned.group_by("date").agg(pl.len().alias("count")).sort("date")

        return curve

    @property
    def metadata_columns(self) -> list[str]:
        """Returns clinical/metadata column names."""  # noqa: D421
        return [c for c in self._df.columns if c.startswith("meta_")]

    @property
    def ui_metadata_columns(self) -> list[str]:
        """Returns metadata names without 'meta_' prefix for UI display."""  # noqa: D421
        return [c.replace("meta_", "", 1) for c in self.metadata_columns]

    @property
    def genotypes(self) -> list[str]:
        """Compiles a list of all genetic variable columns."""
        prefixes = (
            f"{Domain.GENOTYPE.value}_",
            f"{Domain.PHENOTYPE.value}_",
            f"{Domain.AMR.value}_",
            f"{Domain.VIRULENCE.value}_",
        )
        return [c for c in self._df.columns if c.startswith(prefixes)]

    @property
    def stratify_cols(self) -> list[str]:
        """Returns columns suitable for stratification."""  # noqa: D421
        exclude = {"sample_id", "latitude", "longitude"}
        prefixes = (
            f"{Domain.QC.value}_",
            "meta_",
            f"{Domain.SPATIAL_RES.value}_",
            f"{Domain.TEMPORAL_RES.value}_",
        )
        return [c for c in self._df.columns if c not in exclude and not c.startswith(prefixes)]

    @property
    def cluster_cols(self) -> list[str]:
        """Returns columns suitable for cluster adjustment."""  # noqa: D421
        return [c for c in self._df.columns if c.startswith(f"{Domain.CLUSTER.value}_") or c.endswith("_ST")]

    def aggregate_prevalence(
        self,
        stratify_by: Sequence[str | StrEnum],
        trait_col: str | StrEnum | None = None,
        cluster_col: str | StrEnum | None = None,
        negative_indicator: str | list[str] = "-",
        pad_zeros: bool = False,
    ) -> pl.DataFrame:
        """Aggregates data to calculate event counts and denominators for prevalence."""
        df = self._df
        stratify_by_str = [f"{s.value}" if hasattr(s, "value") else f"{s}" for s in stratify_by]
        trait_col_str = (
            f"{trait_col.value}" if hasattr(trait_col, "value") else (f"{trait_col}" if trait_col is not None else None)
        )
        cluster_col_str = (
            f"{cluster_col.value}"
            if hasattr(cluster_col, "value")
            else (f"{cluster_col}" if cluster_col is not None else None)
        )

        if trait_col_str:
            denom_cols = list(stratify_by_str)
            trait_strata = list(stratify_by_str)
            valid_df = df.filter(pl.col(trait_col_str).is_not_null())

            if valid_df.schema[trait_col_str] == pl.Boolean:
                is_event = pl.col(trait_col_str)
            else:
                neg_list = [negative_indicator] if isinstance(negative_indicator, str) else list(negative_indicator)
                is_event = ~pl.col(trait_col_str).is_in(neg_list)
            valid_df = valid_df.with_columns(is_event.alias("_is_event"))
        else:
            if len(stratify_by_str) < 1:
                raise ValueError("Compositional prevalence requires at least 1 stratify_by column.")
            denom_cols = list(stratify_by_str[:-1])
            trait_strata = list(stratify_by_str)
            valid_df = df.filter(pl.col(trait_strata[-1]).is_not_null())

        if trait_col_str:
            if cluster_col_str:
                events_df = (
                    valid_df.filter(pl.col("_is_event"))
                    .group_by(trait_strata)
                    .agg(pl.col(cluster_col_str).n_unique().alias("event"))
                    if trait_strata
                    else valid_df.filter(pl.col("_is_event")).select(pl.col(cluster_col_str).n_unique().alias("event"))
                )
                denoms_df = (
                    valid_df.group_by(denom_cols).agg(pl.col(cluster_col_str).n_unique().alias("n"))
                    if denom_cols
                    else valid_df.select(pl.col(cluster_col_str).n_unique().alias("n"))
                )
            else:
                events_df = (
                    valid_df.filter(pl.col("_is_event")).group_by(trait_strata).agg(pl.len().alias("event"))
                    if trait_strata
                    else valid_df.filter(pl.col("_is_event")).select(pl.len().alias("event"))
                )
                denoms_df = (
                    valid_df.group_by(denom_cols).agg(pl.len().alias("n"))
                    if denom_cols
                    else valid_df.select(pl.len().alias("n"))
                )
        else:
            if cluster_col_str:
                events_df = valid_df.group_by(trait_strata).agg(pl.col(cluster_col_str).n_unique().alias("event"))
                denoms_df = (
                    valid_df.group_by(denom_cols).agg(pl.col(cluster_col_str).n_unique().alias("n"))
                    if denom_cols
                    else valid_df.select(pl.col(cluster_col_str).n_unique().alias("n"))
                )
            else:
                events_df = valid_df.group_by(trait_strata).agg(pl.len().alias("event"))
                denoms_df = (
                    valid_df.group_by(denom_cols).agg(pl.len().alias("n"))
                    if denom_cols
                    else valid_df.select(pl.len().alias("n"))
                )

        if pad_zeros and trait_strata:
            unique_dfs = [df.select(pl.col(c).drop_nulls().unique()) for c in trait_strata]
            grid = unique_dfs[0]
            for udf in unique_dfs[1:]:
                grid = grid.join(udf, how="cross")
            agg_df = grid.join(events_df, on=trait_strata, how="left").with_columns(pl.col("event").fill_null(0))
        else:
            agg_df = events_df

        if denom_cols:
            agg_df = agg_df.join(denoms_df, on=denom_cols, how="left").with_columns(pl.col("n").fill_null(0))
        else:
            n_val = denoms_df.get_column("n")[0] if len(denoms_df) > 0 else 0
            agg_df = agg_df.with_columns(pl.lit(n_val).alias("n"))

        if not pad_zeros:
            agg_df = agg_df.filter(pl.col("n") > 0)

        if trait_col_str:
            agg_df = agg_df.with_columns(pl.lit(trait_col_str).alias("target"))
        else:
            agg_df = agg_df.rename({trait_strata[-1]: "target"})

        agg_df = agg_df.with_columns(pl.col("event").cast(pl.Int64), pl.col("n").cast(pl.Int64))

        meta_dict = {
            "metric_meta": {
                "trait": f"{trait_col_str}"
                if trait_col_str
                else (f"{stratify_by_str[-1]}" if stratify_by_str else "unknown"),
                "stratified_by": [
                    f"{s.value}" if hasattr(s, "value") else f"{s}"
                    for s in (stratify_by_str if trait_col_str else stratify_by_str[:-1])
                ],
                "adjusted_for": f"{cluster_col_str}" if cluster_col_str else "unknown",
                "aggregation_type": AggregationType.TRAIT if trait_col_str else AggregationType.COMPOSITIONAL,
                "is_zero_padded": pad_zeros,
            }
        }
        return self._wrap_result(agg_df, meta_dict)

    def aggregate_diversity(
        self,
        stratify_by: Sequence[str | StrEnum],
        trait_col: str | StrEnum | None = None,
        cluster_col: str | StrEnum | None = None,
        negative_indicator: str | list[str] = "-",
        pad_zeros: bool = False,
    ) -> pl.DataFrame:
        """Aggregates data to calculate counts for diversity analysis."""
        df = self._df
        stratify_by_str = [f"{s.value}" if hasattr(s, "value") else f"{s}" for s in stratify_by]
        trait_col_str = (
            f"{trait_col.value}" if hasattr(trait_col, "value") else (f"{trait_col}" if trait_col is not None else None)
        )
        cluster_col_str = (
            f"{cluster_col.value}"
            if hasattr(cluster_col, "value")
            else (f"{cluster_col}" if cluster_col is not None else None)
        )

        is_trait = bool(trait_col_str)
        if trait_col_str:
            groupers = list(stratify_by_str)
            trait_strata = list(stratify_by_str) + [trait_col_str]
            valid_df = df.filter(pl.col(trait_col_str).is_not_null())
            if valid_df.schema[trait_col_str] != pl.Boolean:
                neg_list = [negative_indicator] if isinstance(negative_indicator, str) else list(negative_indicator)
                valid_df = valid_df.filter(~pl.col(trait_col_str).is_in(neg_list))
        else:
            if not stratify_by_str:
                raise ValueError("Compositional diversity requires at least 1 stratify_by column.")
            groupers = list(stratify_by_str[:-1])
            trait_col_str = stratify_by_str[-1]
            trait_strata = list(stratify_by_str)
            valid_df = df.filter(pl.col(trait_col_str).is_not_null())

        if cluster_col_str:
            div_df = valid_df.group_by(trait_strata).agg(pl.col(cluster_col_str).n_unique().alias("variant_count"))
        else:
            div_df = valid_df.group_by(trait_strata).agg(pl.len().alias("variant_count"))

        if pad_zeros and trait_strata:
            unique_dfs = [df.select(pl.col(c).drop_nulls().unique()) for c in trait_strata]
            grid = unique_dfs[0]
            for udf in unique_dfs[1:]:
                grid = grid.join(udf, how="cross")
            div_df = grid.join(div_df, on=trait_strata, how="left").with_columns(pl.col("variant_count").fill_null(0))

        if groupers:
            div_df = div_df.with_columns(pl.col("variant_count").sum().over(groupers).alias("n_total"))
        else:
            div_df = div_df.with_columns(pl.col("variant_count").sum().alias("n_total"))

        div_df = div_df.rename({trait_col_str: "target"})
        if is_trait:
            div_df = div_df.with_columns(pl.lit(trait_col_str).alias("target"))

        div_df = div_df.with_columns(pl.col("variant_count").cast(pl.Int64), pl.col("n_total").cast(pl.Int64))

        meta_dict = {
            "metric_meta": {
                "trait": f"{trait_col_str}"
                if is_trait
                else (f"{stratify_by_str[-1]}" if stratify_by_str else "unknown"),
                "stratified_by": [
                    f"{s.value}" if hasattr(s, "value") else f"{s}"
                    for s in (stratify_by_str if is_trait else stratify_by_str[:-1])
                ],
                "aggregation_type": AggregationType.TRAIT if is_trait else AggregationType.COMPOSITIONAL,
                "is_zero_padded": pad_zeros,
            }
        }
        return self._wrap_result(div_df, meta_dict)

    def aggregate_incidence(
        self,
        stratify_by: Sequence[str | StrEnum],
        trait_col: str | StrEnum | None = None,
        freq: str | TemporalResolution = TemporalResolution.MONTH,
        cluster_col: str | StrEnum | None = None,
        negative_indicator: str | list[str] = "-",
        pad_zeros: bool = False,
        temporal_col: str | StrEnum | None = None,
    ) -> pl.DataFrame:
        """Aggregates data for time-series incidence analysis."""
        df = self._df
        stratify_by_str = [f"{s.value}" if hasattr(s, "value") else f"{s}" for s in stratify_by]
        trait_col_str = (
            f"{trait_col.value}" if hasattr(trait_col, "value") else (f"{trait_col}" if trait_col is not None else None)
        )
        cluster_col_str = (
            f"{cluster_col.value}"
            if hasattr(cluster_col, "value")
            else (f"{cluster_col}" if cluster_col is not None else None)
        )

        temporal_col_str = self._resolve_temporal_col(temporal_col)
        interval = self._resolve_freq(freq)

        df_binned = df.with_columns(pl.col(temporal_col_str).dt.truncate(interval).alias("date_bin"))

        if trait_col_str:
            denom_cols = ["date_bin"] + list(stratify_by_str)
            trait_strata = ["date_bin"] + list(stratify_by_str)
        else:
            if not stratify_by_str:
                raise ValueError("Compositional incidence requires at least 1 stratify_by column.")
            denom_cols = ["date_bin"] + list(stratify_by_str[:-1])
            trait_strata = ["date_bin"] + list(stratify_by_str)

        if trait_col_str:
            valid_df = df_binned.filter(pl.col(trait_col_str).is_not_null())
            if valid_df.schema[trait_col_str] == pl.Boolean:
                is_event = pl.col(trait_col_str)
            else:
                neg_list = [negative_indicator] if isinstance(negative_indicator, str) else list(negative_indicator)
                is_event = ~pl.col(trait_col_str).is_in(neg_list)
            valid_df = valid_df.with_columns(is_event.alias("_is_event"))

            if cluster_col_str:
                events_df = (
                    valid_df.filter(pl.col("_is_event"))
                    .group_by(trait_strata)
                    .agg(pl.col(cluster_col_str).n_unique().alias("variant_count"))
                )
                denoms_df = valid_df.group_by(denom_cols).agg(
                    pl.col(cluster_col_str).n_unique().alias("total_sequenced")
                )
            else:
                events_df = (
                    valid_df.filter(pl.col("_is_event")).group_by(trait_strata).agg(pl.len().alias("variant_count"))
                )
                denoms_df = valid_df.group_by(denom_cols).agg(pl.len().alias("total_sequenced"))
        else:
            valid_df = df_binned.filter(pl.col(trait_strata[-1]).is_not_null())
            if cluster_col_str:
                events_df = valid_df.group_by(trait_strata).agg(
                    pl.col(cluster_col_str).n_unique().alias("variant_count")
                )
                denoms_df = valid_df.group_by(denom_cols).agg(
                    pl.col(cluster_col_str).n_unique().alias("total_sequenced")
                )
            else:
                events_df = valid_df.group_by(trait_strata).agg(pl.len().alias("variant_count"))
                denoms_df = valid_df.group_by(denom_cols).agg(pl.len().alias("total_sequenced"))

        min_date = df_binned.select(pl.col("date_bin").min()).item()
        max_date = df_binned.select(pl.col("date_bin").max()).item()

        if min_date is not None and max_date is not None:
            date_range = pl.date_range(min_date, max_date, interval, eager=True).alias("date_bin")
            dates_df = pl.DataFrame([date_range]).with_columns(pl.col("date_bin").cast(df_binned.schema["date_bin"]))
        else:
            dates_df = pl.DataFrame(schema={"date_bin": df_binned.schema["date_bin"]})

        if pad_zeros:
            strata_cols = trait_strata[1:]
            grid = dates_df
            for c in strata_cols:
                udf = df_binned.select(pl.col(c).drop_nulls().unique())
                grid = grid.join(udf, how="cross")
        else:
            if len(trait_strata) > 1:
                strata_cols = trait_strata[1:]
                obs_strata = df_binned.select(strata_cols).drop_nulls().unique()
                grid = dates_df.join(obs_strata, how="cross")
            else:
                grid = dates_df

        inc_df = grid.join(events_df, on=trait_strata, how="left").with_columns(pl.col("variant_count").fill_null(0))
        inc_df = inc_df.join(denoms_df, on=denom_cols, how="left").with_columns(pl.col("total_sequenced").fill_null(0))
        inc_df = inc_df.rename({"date_bin": "date"})

        if trait_col_str:
            inc_df = inc_df.with_columns(pl.lit(trait_col_str).alias("target"))
        else:
            inc_df = inc_df.rename({trait_strata[-1]: "target"})

        inc_df = inc_df.with_columns(pl.col("variant_count").cast(pl.Int64), pl.col("total_sequenced").cast(pl.Int64))

        freq_str = f"{interval.value}" if hasattr(interval, "value") else f"{interval}"
        meta_dict = {
            "metric_meta": {
                "trait": f"{trait_col_str}"
                if trait_col_str
                else (f"{stratify_by_str[-1]}" if stratify_by_str else "unknown"),
                "freq": freq_str,
                "stratified_by": [
                    f"{s.value}" if hasattr(s, "value") else f"{s}"
                    for s in (stratify_by_str if trait_col_str else stratify_by_str[:-1])
                ],
                "aggregation_type": AggregationType.TRAIT if trait_col_str else AggregationType.COMPOSITIONAL,
                "is_zero_padded": pad_zeros,
            }
        }
        return self._wrap_result(inc_df, meta_dict)

    def transmission_network(
        self,
        clone_col: str | StrEnum,
        spatial_threshold_km: float = 10.0,
        temporal_threshold_days: int = 20,
        temporal_col: str | StrEnum | None = None,
    ) -> TransmissionDistances:
        """Builds a sparse adjacency graph of transmission links."""
        df = self._df
        clone_col_str = str(clone_col)
        if clone_col_str not in df.columns:
            raise KeyError(f"Clone column '{clone_col_str}' not found in DataFrame.")
        if not self.has_spatial:
            raise KeyError("Spatial clustering requires 'latitude' and 'longitude' columns.")

        temporal_col_str = self._resolve_temporal_col(temporal_col)
        coords, raw_dates, _ = self._get_spatiotemporal_arrays(temporal_col_str)

        return TransmissionDistances.from_spatiotemporal(
            sample_ids=df.get_column("sample_id").to_list(),
            coords=coords,
            dates=raw_dates,
            clones=df.get_column(clone_col_str).to_numpy(),
            spatial_threshold_km=spatial_threshold_km,
            temporal_threshold_days=temporal_threshold_days,
        )

    def transmission_clusters(
        self,
        clone_col: str | StrEnum,
        spatial_threshold_km: float = 10.0,
        temporal_threshold_days: int = 20,
        temporal_col: str | StrEnum | None = None,
        network: TransmissionDistances | None = None,
    ) -> pl.Series:
        """Extracts categorical cluster labels from the transmission network."""
        df = self._df
        clone_col_str = str(clone_col)
        temporal_col_str = self._resolve_temporal_col(temporal_col)

        if network is None:
            network = self.transmission_network(
                clone_col_str, spatial_threshold_km, temporal_threshold_days, temporal_col_str
            )

        labels = network.get_clusters()
        if hasattr(labels, "to_numpy"):
            labels_array = labels.to_numpy().astype(float)
        else:
            labels_array = np.array(labels, dtype=float)

        _, _, valid_mask = self._get_spatiotemporal_arrays(temporal_col_str)
        clone_mask = df.get_column(clone_col_str).is_not_null().to_numpy()

        labels_array[~valid_mask] = np.nan
        labels_array[~clone_mask] = np.nan

        col_name = f"{Domain.CLUSTER.value}_transmission_{spatial_threshold_km}km_{temporal_threshold_days}days"
        str_labels = [str(int(x)) if not np.isnan(x) else None for x in labels_array]
        return pl.Series(col_name, str_labels).cast(pl.Categorical)


@pl.api.register_dataframe_namespace("epi")
class EpiAccessor(EpiMixin):
    """Polars DataFrame namespace for epidemiological analysis."""

    def __init__(self, polars_obj: pl.DataFrame):  # noqa: ANN204, D107
        self._df_obj = polars_obj


class CompositionBarPlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(cls, result: estimators.PrevalenceEstimates, **kwargs) -> go.Figure:  # type: ignore # noqa: ANN003, D102
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        is_comp = result.aggregation_type == AggregationType.COMPOSITIONAL
        if not is_comp:
            raise ValueError("Composition Bar Plot strictly requires Compositional aggregation mode.")

        df = result.data
        target_col = "target"
        group_cols = result.stratified_by
        strata_label = ", ".join(result.stratified_by) if result.stratified_by else "Global"
        title_prefix = "Sample Composition"

        fig = go.Figure()

        target_ranks = (
            df.group_by(target_col)
            .agg(pl.col("estimate").sum())
            .sort("estimate", descending=True)[target_col]
            .to_list()
        )
        grouped_df = df.partition_by(target_col, as_dict=True)

        if not group_cols:
            for t in target_ranks:
                key = (t,) if isinstance(t, tuple) else (t,)
                t_df = grouped_df.get(key)
                if t_df is None or len(t_df) == 0:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=["Global Formulation"],
                        y=t_df["estimate"].to_numpy(),
                        name=str(t),
                        hovertemplate=f"<b>{cls._clean_label(t)}</b><br>Prevalence: %{{y:.1%}}<extra></extra>",
                    )
                )
        else:
            group_col = group_cols[0]
            for t in target_ranks:
                key = (t,) if isinstance(t, tuple) else (t,)
                t_df = grouped_df.get(key)
                if t_df is None or len(t_df) == 0:
                    continue
                fig.add_trace(
                    go.Bar(
                        x=t_df[group_col].to_list(),
                        y=t_df["estimate"].to_numpy(),
                        name=str(t),
                        hovertemplate=f"<b>{cls._clean_label(t)}</b><br>{cls._clean_label(group_col)}: %{{x}}<br>Prevalence: %{{y:.1%}}<extra></extra>",  # noqa: E501
                    )
                )

        return cls.apply_theme(
            fig.update_layout(
                barmode="stack",
                title=f"<b>{title_prefix}</b><br><sup>Stratified by {strata_label}</sup>",
                yaxis_title="Cumulative Prevalence",
                yaxis=dict(tickformat=".0%"),
            )
        )


class CompositionHeatmapPlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(cls, result: estimators.PrevalenceEstimates, **kwargs) -> go.Figure:  # type: ignore # noqa: ANN003, D102
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        df = result.data
        is_comp = result.aggregation_type == AggregationType.COMPOSITIONAL

        if not is_comp:
            if len(result.stratified_by) != 2:
                raise ValueError(
                    f"Trait Heatmap strictly requires exactly 2 stratification variables. Found: {result.stratified_by}"
                )
            y_col = result.stratified_by[0]
            x_col = result.stratified_by[1]
            title_prefix = "Prevalence Matrix"
        else:
            y_col = "target"
            group_cols = result.stratified_by
            if len(group_cols) != 1:
                raise ValueError(
                    f"Composition Heatmap strictly requires exactly 1 grouping variable alongside the target. "
                    f"Found target '{result.trait}' and groups: {group_cols}"
                )
            x_col = group_cols[0]
            title_prefix = "Density Matrix"

        pivot_df = df.pivot(on=x_col, index=y_col, values="estimate", aggregate_function="sum").fill_null(0)
        x_cols = [c for c in pivot_df.columns if c != y_col]

        sum_series = pivot_df.select(x_cols).to_numpy().sum(axis=1)
        pivot_df = pivot_df.with_columns(total_burden=pl.Series(sum_series)).sort("total_burden", descending=False)

        z_matrix = pivot_df.select(x_cols).to_numpy()
        y_labels = pivot_df[y_col].to_list()

        fig = go.Figure(
            data=go.Heatmap(
                z=z_matrix,
                x=x_cols,
                y=y_labels,
                colorscale=cls.get_colorscale(transparent=True),
                showscale=True,
                colorbar=dict(tickformat=".0%"),
                xgap=1,
                ygap=1,
                hovertemplate=f"<b>%{{y}}</b><br>{x_col}: %{{x}}<br>Prevalence: %{{z:.1%}}<extra></extra>",
            )
        )

        return cls.apply_theme(
            fig.update_layout(
                title=f"<b>{title_prefix}</b><br><sup>{y_col} vs {x_col}</sup>",
                xaxis_title=x_col.replace("_", " ").title(),
                yaxis_title=y_col.replace("_", " ").title(),
            )
        )


class ForestPlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(cls, result: estimators.PrevalenceEstimates, sort_by: str = "estimate", **kwargs) -> go.Figure:  # type: ignore # noqa: ANN003, D102
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        df = result.data
        is_comp = result.aggregation_type == AggregationType.COMPOSITIONAL
        top_n = 0

        if is_comp:
            y_col = "target"
            group_cols = result.stratified_by
            color_col = group_cols[0] if group_cols else None

            top_n = kwargs.get("top_n", 20)
            target_totals = df.group_by(y_col).agg(pl.col("estimate").sum()).sort("estimate", descending=True)

            is_truncated = len(target_totals) > top_n
            if is_truncated:
                top_targets = target_totals.head(top_n)[y_col].to_list()
                df = df.filter(pl.col(y_col).is_in(top_targets))

        else:
            y_col = result.stratified_by[0] if result.stratified_by else "target"
            color_col = result.stratified_by[1] if len(result.stratified_by) > 1 else None
            is_truncated = False

        if sort_by in df.columns:
            df = df.sort(sort_by, descending=True)

        y_order = df[y_col].unique().to_list()

        fig = go.Figure()

        if color_col:
            for group_key, group_df in df.partition_by(color_col, as_dict=True).items():
                group_name = group_key[0] if isinstance(group_key, tuple) else group_key
                est = group_df["estimate"].to_numpy()
                upper = group_df["upper"].to_numpy()
                lower = group_df["lower"].to_numpy()
                y_vals = group_df[y_col].to_list()
                c_vals = group_df[color_col].to_list()

                customdata = np.column_stack((c_vals, lower, upper))

                fig.add_trace(
                    go.Scatter(
                        x=est,
                        y=y_vals,
                        name=str(group_name),
                        mode="markers",
                        marker=dict(size=10, symbol="square"),
                        error_x=dict(
                            type="data",
                            symmetric=False,
                            array=upper - est,
                            arrayminus=est - lower,
                            width=0,
                            thickness=2,
                        ),
                        customdata=customdata,
                        hovertemplate=(
                            f"<b>{cls._clean_label(y_col)}:</b> %{{y}}<br>"
                            f"<b>{cls._clean_label(color_col)}:</b> %{{customdata[0]}}<br>"
                            "<b>Prevalence:</b> %{x:.1%}<br>"
                            "<b>95% CI:</b> %{customdata[1]:.1%} - %{customdata[2]:.1%}<extra></extra>"
                        ),
                    )
                )

            fig.update_layout(scattermode="group")

        else:
            est = df["estimate"].to_numpy()
            upper = df["upper"].to_numpy()
            lower = df["lower"].to_numpy()
            customdata = np.column_stack((lower, upper))

            fig.add_trace(
                go.Scatter(
                    x=est,
                    y=df[y_col].to_list(),
                    mode="markers",
                    marker=dict(size=10, color=cls._MAIN_COLOUR, symbol="square"),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=upper - est,
                        arrayminus=est - lower,
                        color=cls._CI_COLOUR,
                        width=0,
                        thickness=2,
                    ),
                    customdata=customdata,
                    hovertemplate=(
                        f"<b>{cls._clean_label(y_col)}:</b> %{{y}}<br>"
                        "<b>Prevalence:</b> %{x:.1%}<br>"
                        "<b>95% CI:</b> %{customdata[0]:.1%} - %{customdata[1]:.1%}<extra></extra>"
                    ),
                )
            )

        title_prefix = "Prevalence Estimates"
        if is_truncated:
            title_prefix += f" (Top {top_n})"

        yaxis_kwargs = dict(
            autorange="reversed",
            title=cls._clean_label(y_col),
            type="category",
            categoryorder="array",
            categoryarray=y_order,
        )

        return cls.apply_theme(
            fig.update_layout(
                title=f"<b>{title_prefix}</b><br><sup>Trait: {cls._clean_label(result.trait)} | Stratified by {', '.join(result.stratified_by)}</sup>",  # noqa: E501
                xaxis_title="Prevalence (%)",
                yaxis=yaxis_kwargs,
                xaxis=dict(tickformat=".0%"),
            )
        )


class PyramidPlotter(BasePlotter):
    """Generates a side-by-side Pyramid Plot (Butterfly Chart) from a ComparisonResult."""

    @classmethod
    def can_render(cls, result: Any) -> bool:  # noqa: ANN401, D102
        return isinstance(result, ComparisonResult)

    @classmethod
    def render(
        cls,
        result: ComparisonResult,
        df: Any = None,  # noqa: ANN401
        sort_by: str = "right",
        **kwargs,  # noqa: ANN003
    ) -> go.Figure:
        """Renders a pyramid plot comparing two estimation results.

        Args:
            result: The ComparisonResult object containing the merged data.
            df: Optional dataset (not typically used here).
            sort_by: Column to sort the y-axis by ('left', 'right', or 'name').
            **kwargs: Additional arguments to pass to the figure.
        """
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")

        comp_df = result.df

        # Determine the y-axis labels. Usually the target or trait.
        if "target" in comp_df.columns:
            y_col = "target"
        elif "trait" in comp_df.columns:
            y_col = "trait"
        else:
            y_col = [c for c, dtype in comp_df.schema.items() if dtype == pl.Utf8][0]

        if sort_by == "left":
            comp_df = comp_df.sort("estimate_left", descending=False)
        elif sort_by == "right":
            comp_df = comp_df.sort("estimate_right", descending=False)
        else:
            comp_df = comp_df.sort(y_col, descending=True)

        y_vals = comp_df[y_col].to_list()

        # Create subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            shared_yaxes=True,
            horizontal_spacing=0.02,
            subplot_titles=(result.left_name, result.right_name),
        )

        # Left side
        fig.add_trace(
            go.Bar(
                y=y_vals,
                x=comp_df["estimate_left"].to_list(),
                orientation="h",
                name=result.left_name,
                marker_color="#1f77b4",
                marker_line_color="black",
                marker_line_width=1,
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(comp_df["upper_left"] - comp_df["estimate_left"]).to_list(),
                    arrayminus=(comp_df["estimate_left"] - comp_df["lower_left"]).to_list(),
                    color="black",
                    thickness=1,
                    width=4,
                ),
            ),
            row=1,
            col=1,
        )

        # Right side
        fig.add_trace(
            go.Bar(
                y=y_vals,
                x=comp_df["estimate_right"].to_list(),
                orientation="h",
                name=result.right_name,
                marker_color="#f2a900",
                marker_line_color="black",
                marker_line_width=1,
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(comp_df["upper_right"] - comp_df["estimate_right"]).to_list(),
                    arrayminus=(comp_df["estimate_right"] - comp_df["lower_right"]).to_list(),
                    color="black",
                    thickness=1,
                    width=4,
                ),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            barmode="overlay",
            showlegend=False,
            height=max(400, len(y_vals) * 30),
            plot_bgcolor="white",
            **kwargs,
        )

        fig.update_xaxes(autorange="reversed", row=1, col=1)
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGray")

        return fig


class EpicurvePlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (estimators.IncidenceEstimates,)

    @classmethod
    def render(cls, result: estimators.IncidenceEstimates, **kwargs) -> go.Figure:  # type: ignore # noqa: ANN003, D102
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")

        data = result.data.sort("date")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=data["date"].to_list(),
                y=data["total_sequenced"].to_numpy(),
                mode="lines",
                fill="tozeroy",
                fillcolor="rgba(51, 65, 85, 0.3)",
                line=dict(color="#475569", width=1, dash="dot"),
                name="Total Sequenced Volume",
                hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Total Sequenced</b>: %{y}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Bar(
                x=data["date"].to_list(),
                y=data["variant_count"].to_numpy(),
                marker_color=cls._MAIN_COLOUR,
                name=f"{cls._clean_label(result.trait)} Cases",
                hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Cases</b>: %{y}<extra></extra>",
            )
        )

        subtitle = ""
        if (
            result.model_results is not None
            and not result.model_results.is_empty()
            and "IRR" in result.model_results.columns
        ):
            irr = result.model_results["IRR"][0]
            if irr is not None and not np.isnan(irr):
                direction = "Increasing" if irr > 1 else "Decreasing"
                subtitle = f"<br><sup>Trend: {direction} (IRR: {irr:.2f} per time step)</sup>"

        return cls.apply_theme(
            fig.update_layout(
                title=f"<b>Epidemic Curve: {cls._clean_label(result.trait)}</b>{subtitle}",
                xaxis=dict(title="Date", type="date"),
                yaxis=dict(title="Count"),
                barmode="overlay",
                hovermode="x unified",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
            )
        )


class LongitudinalPrevalencePlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(cls, result: estimators.PrevalenceEstimates, **kwargs) -> go.Figure:  # type: ignore # noqa: ANN003, D102
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")

        time_cols = [
            col
            for col in result.stratified_by
            if "date" in col.lower() or "year" in col.lower() or "month" in col.lower()
        ]

        if not time_cols:
            raise ValueError(
                "No temporal column detected in the strata. Ensure you stratified by a Date, Year, or Month column."
            )

        time_col = time_cols[0]
        data = result.data.sort(time_col)

        x_vals = data[time_col].to_list()
        upper_vals = data["upper"].to_list()
        lower_vals = data["lower"].to_list()

        fig = go.Figure()

        fig.add_trace(
            go.Scattergl(
                x=x_vals + x_vals[::-1],
                y=upper_vals + lower_vals[::-1],
                fill="toself",
                fillcolor=cls._CI_COLOUR,
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                legendgroup="Prevalence",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=data["estimate"].to_numpy(),
                mode="lines+markers",
                line=dict(color=cls._MAIN_COLOUR, width=3),
                marker=dict(size=8, color=cls._MAIN_COLOUR),
                name="Prevalence",
                hovertemplate="<b>Date</b>: %{x}<br><b>Prevalence</b>: %{y:.2%}<extra></extra>",
                legendgroup="Prevalence",
            )
        )

        return cls.apply_theme(
            fig.update_layout(
                title=f"<b>Longitudinal Prevalence of {cls._clean_label(result.trait)}</b>",
                xaxis=dict(title=time_col.title()),
                yaxis=dict(title="Prevalence", tickformat=".0%", range=[0, 1.05]),
            )
        )


class CumulativeCoveragePlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates, dict)

    @classmethod
    def render(  # type: ignore # noqa: D102
        cls,
        result: estimators.PrevalenceEstimates | dict[str, Any],
        max_valencies: int | None = None,
        **kwargs,  # noqa: ANN003
    ) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")

        if isinstance(result, dict):
            res = result.get("res")
            formulation = result.get("formulation")
        else:
            res = result
            formulation = None

        if getattr(res, "aggregation_type", None) != AggregationType.COMPOSITIONAL:
            raise ValueError("Cumulative coverage strictly requires compositional prevalence estimates.")

        data = res.data if hasattr(res, "data") else res
        assert data is not None
        z_score = norm.ppf(0.975)

        upper = data["upper"].to_numpy()
        lower = data["lower"].to_numpy()
        se = (upper - lower) / (2 * z_score)
        var = se**2
        data = data.with_columns(var=pl.Series(var))

        if formulation:
            target_order = formulation.get_formulation()
        else:
            target_order = (
                data.group_by("target").agg(pl.col("event").sum()).sort("event", descending=True)["target"].to_list()
            )
            if max_valencies:
                target_order = target_order[:max_valencies]

        fig = go.Figure()
        group_cols = getattr(res, "stratified_by", [])

        colors = px.colors.qualitative.Plotly

        if not group_cols:
            grouped = data.group_by("target").agg([pl.col("estimate").sum(), pl.col("var").sum()])
            target_dict = {row["target"]: row for row in grouped.iter_rows(named=True)}

            estimates_ordered = np.array([target_dict.get(t, {}).get("estimate", 0.0) for t in target_order])
            vars_ordered = np.array([target_dict.get(t, {}).get("var", 0.0) for t in target_order])

            cum_prop = np.clip(np.cumsum(estimates_ordered), 0, 1)
            cum_se = np.sqrt(np.cumsum(vars_ordered))

            cum_lower = np.clip(cum_prop - z_score * cum_se, 0, 1)
            cum_upper = np.clip(cum_prop + z_score * cum_se, 0, 1)

            fig.add_trace(
                go.Scatter(
                    x=target_order + target_order[::-1],
                    y=cum_upper.tolist() + cum_lower.tolist()[::-1],
                    fill="toself",
                    fillcolor=cls._MAIN_COLOUR,
                    opacity=0.2,
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup="Cumulative Population Coverage",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=target_order,
                    y=cum_prop,
                    mode="lines+markers",
                    name="Cumulative Population Coverage",
                    line=dict(color=cls._MAIN_COLOUR, width=3),
                    marker=dict(size=8, color=cls._MAIN_COLOUR),
                    customdata=np.column_stack((cum_lower, cum_upper)),
                    hovertemplate="<b>%{x}</b><br>Cumulative Coverage: %{y:.1%}<br>95% CI: %{customdata[0]:.1%} - %{customdata[1]:.1%}<extra></extra>",  # noqa: E501
                    legendgroup="Cumulative Population Coverage",
                )
            )
            strata_label = "Baseline"
        else:
            color_col = group_cols[0]
            strata_label = f"Stratified by {cls._clean_label(color_col)}"

            for i, (stratum_key, group_df) in enumerate(data.partition_by(color_col, as_dict=True).items()):
                stratum = stratum_key[0] if isinstance(stratum_key, tuple) else stratum_key
                grouped = group_df.group_by("target").agg([pl.col("estimate").sum(), pl.col("var").sum()])
                target_dict = {row["target"]: row for row in grouped.iter_rows(named=True)}

                estimates_ordered = np.array([target_dict.get(t, {}).get("estimate", 0.0) for t in target_order])
                vars_ordered = np.array([target_dict.get(t, {}).get("var", 0.0) for t in target_order])

                cum_prop = np.clip(np.cumsum(estimates_ordered), 0, 1)
                cum_se = np.sqrt(np.cumsum(vars_ordered))

                cum_lower = np.clip(cum_prop - z_score * cum_se, 0, 1)
                cum_upper = np.clip(cum_prop + z_score * cum_se, 0, 1)

                color = colors[i % len(colors)]

                fig.add_trace(
                    go.Scatter(
                        x=target_order + target_order[::-1],
                        y=cum_upper.tolist() + cum_lower.tolist()[::-1],
                        fill="toself",
                        fillcolor=color,
                        opacity=0.2,
                        line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup=str(stratum),
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=target_order,
                        y=cum_prop,
                        mode="lines+markers",
                        name=str(stratum),
                        line=dict(color=color, width=2),
                        marker=dict(size=6, color=color),
                        customdata=np.column_stack((cum_lower, cum_upper)),
                        hovertemplate=f"<b>%{{x}}</b><br>{cls._clean_label(color_col)}: {stratum}<br>Cumulative Coverage: %{{y:.1%}}<br>95% CI: %{{customdata[0]:.1%}} - %{{customdata[1]:.1%}}<extra></extra>",  # noqa: E501
                        legendgroup=str(stratum),
                    )
                )

        return cls.apply_theme(
            fig.update_layout(
                title=f"<b>Cumulative Coverage</b><br><sup>Targeting top {len(target_order)} {cls._clean_label(getattr(res, 'trait', 'unknown'))} variants | {strata_label}</sup>",  # noqa: E501
                xaxis=dict(title="Variant added to formulation", tickangle=45),
                yaxis=dict(title="Cumulative Population Coverage", tickformat=".0%", range=[0, 1.05]),
                hovermode="x unified",
            )
        )


class StabilityBumpPlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (Formulation,)

    @classmethod
    def render(cls, formulation: Formulation, **kwargs) -> go.Figure:  # type: ignore # noqa: ANN003, D102
        if not cls.can_render(formulation):
            raise TypeError(f"{cls.__name__} does not support {type(formulation).__name__}.")

        history = formulation.permutation_history
        baseline = formulation.rankings
        valency = formulation.max_valency

        fig = go.Figure()

        top_targets = (
            baseline.head(valency)["target"].to_list() if "target" in baseline.columns and len(baseline) > 0 else []
        )

        x_categories = ["Baseline"]
        if "holdout_group" in history.columns and len(history) > 0:
            x_categories += history["holdout_group"].unique().to_list()

        baseline_ranks = (
            dict(zip(baseline["target"].to_list(), baseline["baseline_rank"].to_list()))
            if "target" in baseline.columns and "baseline_rank" in baseline.columns
            else {}
        )

        history_lookup = {}
        if len(history) > 0 and "target" in history.columns and "holdout_group" in history.columns:
            for row in history.iter_rows(named=True):
                history_lookup[(row["target"], str(row["holdout_group"]))] = row["loo_rank"]

        for target in top_targets:
            y_ranks = [baseline_ranks.get(target)]
            for group in x_categories[1:]:
                rank = history_lookup.get((target, str(group)))
                y_ranks.append(rank)

            fig.add_trace(
                go.Scatter(
                    x=x_categories,
                    y=y_ranks,
                    mode="lines+markers",
                    name=str(target),
                    line=dict(width=2),
                    marker=dict(size=8),
                    hovertemplate="<b>Holdout: %{x}</b><br>Rank: %{y}<extra></extra>",
                )
            )

        fig.add_hline(
            y=valency + 0.5,
            line_dash="dot",
            line_color="#EF4444",
            annotation_text="Valency Cutoff",
            annotation_position="bottom right",
        )

        return (
            cls.apply_theme(
                fig.update_layout(
                    title=f"<b>Trait Priority Stability (LOO)</b><br><sup>Targeting Top {valency} {cls._clean_label(formulation.trait)} Variants</sup>",  # noqa: E501
                    hovermode="x unified",
                )
            )
            .update_xaxes(title="Excluded Group (Leave-One-Out)")
            .update_yaxes(title="Priority Rank", autorange="reversed", dtick=1)
        )


class LongevityPlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (Formulation,)

    @classmethod
    def render(cls, formulation: Formulation, **kwargs) -> go.Figure:  # type: ignore # noqa: ANN003, D102
        if not cls.can_render(formulation):
            raise TypeError(f"{cls.__name__} does not support {type(formulation).__name__}.")

        forecast = kwargs.get("forecast")
        if forecast is None:
            raise ValueError("LongevityPlotter requires a 'forecast' (IncidenceEstimates) in kwargs.")

        df = formulation.evaluate_longevity(forecast)

        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        x_vals = df["date"].to_list()
        total_cases = df["total_cases"].to_numpy()
        covered_cases = df["covered_cases"].to_numpy()
        coverage_pct = df["coverage_pct"].to_numpy()

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=total_cases,
                name="Total Projected Cases",
                mode="lines",
                line=dict(color=cls._FONT_COLOUR, width=0),
                fill="tozeroy",
                fillcolor="rgba(71, 85, 105, 0.4)",
                hovertemplate="%{y:.1f} Total Cases<extra></extra>",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=covered_cases,
                name="Covered by Vaccine",
                mode="lines",
                line=dict(color=cls._MAIN_COLOUR, width=2),
                fill="tozeroy",
                fillcolor=cls._CI_COLOUR,
                hovertemplate="%{y:.1f} Covered<extra></extra>",
            ),
            secondary_y=False,
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=coverage_pct,
                name="Coverage (%)",
                mode="lines+markers",
                line=dict(color=cls._ACCENT_COLOUR, width=2, dash="dot"),
                marker=dict(size=6, symbol="circle"),
                hovertemplate="%{y:.1f}% Coverage<extra></extra>",
            ),
            secondary_y=True,
        )

        fig.add_hline(
            y=70,
            secondary_y=True,
            line_dash="dash",
            line_color="#EF4444",
            annotation_text="70% Efficacy Target",
            annotation_position="bottom right",
            annotation_font=dict(color="#EF4444"),
        )

        fig.update_yaxes(title_text="Absolute Case Burden", rangemode="tozero", secondary_y=False)
        fig.update_yaxes(title_text="Coverage (%)", range=[0, 105], showgrid=False, secondary_y=True)

        return cls.apply_theme(
            fig.update_layout(
                title=f"<b>Longevity Forecast</b><br><sup>Vaccine Trait: {cls._clean_label(formulation.trait)}</sup>",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
                margin=dict(t=60, b=40, l=40, r=40),
            )
        )


register_plotter(PlotType.COMPOSITION_BAR, CompositionBarPlotter)
register_plotter(PlotType.COMPOSITION_HEATMAP, CompositionHeatmapPlotter)
register_plotter(PlotType.FOREST, ForestPlotter)
register_plotter(PlotType.EPICURVE, EpicurvePlotter)
register_plotter(PlotType.LONGITUDINAL_PREVALENCE, LongitudinalPrevalencePlotter)
register_plotter(PlotType.CUMULATIVE_COVERAGE, CumulativeCoveragePlotter)
register_plotter(PlotType.STABILITY_BUMP, StabilityBumpPlotter)
register_plotter(PlotType.LONGEVITY, LongevityPlotter)
register_plotter(PlotType.PYRAMID, PyramidPlotter)
