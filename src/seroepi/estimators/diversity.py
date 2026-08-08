"""Core statistical estimators for unpooled prevalence, alpha diversity, and beta diversity."""

from enum import StrEnum
from typing import Any, Literal

import numpy as np
import polars as pl
from scipy.spatial.distance import pdist, squareform
from scipy.stats import beta, entropy, norm

from seroepi.constants import AggregationType, AlphaDiversityMetric, BetaDiversityMetric
from seroepi.domains.base import get_dataframe_metadata
from seroepi.estimators.base import AlphaDiversityEstimates, BaseEstimator, BetaDiversityEstimates, PrevalenceEstimates


# Classes --------------------------------------------------------------------------------------------------------------

class AlphaDiversityEstimator(BaseEstimator[AlphaDiversityEstimates]):  # noqa: D101
    _DEFAULT_METRICS = [AlphaDiversityMetric.SHANNON, AlphaDiversityMetric.SIMPSON, AlphaDiversityMetric.RICHNESS]

    def __init__(self, target: str | StrEnum | None = None, metrics: list[AlphaDiversityMetric | str] | None = None):  # noqa: ANN204, D107
        self.target = f"{target.value}" if hasattr(target, "value") else (f"{target}" if target is not None else None)
        self.metrics = [f"{m.value}" if hasattr(m, "value") else f"{m}" for m in (metrics or self._DEFAULT_METRICS)]
        self._method_label = "alpha_diversity"

    def get_params(self) -> dict[str, Any]:
        """Returns parameters for cloning compatibility."""
        return {"target": self.target, "metrics": self.metrics}

    def calculate(self, div_df: Any) -> AlphaDiversityEstimates:  # type: ignore # noqa: ANN401, D102
        raw_meta = get_dataframe_metadata(div_df)
        meta = raw_meta.get("metric_meta", raw_meta) if isinstance(raw_meta, dict) else {}
        df = div_df.data if hasattr(div_df, "data") else div_df

        raw_target = meta.get("trait", self.target)
        if (not raw_target or raw_target == "unknown") and "target" in df.columns and len(df) > 0:
            if df["target"].n_unique() == 1:
                raw_target = df["target"][0]
            else:
                raw_target = "target"
        if not raw_target:
            raw_target = self.target

        target_col = (
            f"{raw_target.value}"
            if hasattr(raw_target, "value")
            else (f"{raw_target}" if raw_target is not None else None)
        )

        strata = [f"{s.value}" if hasattr(s, "value") else f"{s}" for s in meta.get("stratified_by", [])]
        if not strata:
            strata = [
                f"{c.value}" if hasattr(c, "value") else f"{c}"
                for c in df.columns
                if c not in {"target", "variant_count", "n_total"}
            ]

        if not target_col:
            raise ValueError("Target trait must be defined either in init or via accessor metadata.")

        results = []

        # If stratified, group by the strata. Otherwise, treat as one global group.
        if strata:
            groups = df.group_by(strata)
        else:
            groups = [(("Global",), df)]

        for name, group in groups:
            counts = group["variant_count"].to_numpy()

            # Filter out true zeroes (important for richness)
            counts = counts[counts > 0]
            if len(counts) == 0:
                continue

            p = counts / counts.sum()

            row = {}
            if strata:
                tuple_name = name if isinstance(name, tuple) else (name,)
                for s_col, s_val in zip(strata, tuple_name):
                    row[s_col] = s_val

            if "shannon" in self.metrics:
                row["shannon"] = float(entropy(p, base=np.e))
            if "simpson" in self.metrics:
                row["simpson"] = float(1.0 - np.sum(p**2))
            if "richness" in self.metrics:
                row["richness"] = int(len(counts))

            row["n_samples"] = int(counts.sum())
            results.append(row)

        res_df = pl.DataFrame(results) if results else pl.DataFrame()

        adj_raw = meta.get("adjusted_for", "unknown")
        adj_str = (
            f"{adj_raw.value}" if hasattr(adj_raw, "value") else (f"{adj_raw}" if adj_raw is not None else "unknown")
        )

        agg_type = meta.get("aggregation_type", AggregationType.TRAIT)
        if hasattr(agg_type, "value"):
            agg_type = f"{agg_type.value}"
        else:
            agg_type = f"{agg_type}"

        return AlphaDiversityEstimates(
            data=res_df,
            stratified_by=[f"{s.value}" if hasattr(s, "value") else f"{s}" for s in strata],
            adjusted_for=adj_str,
            trait=target_col,
            aggregation_type=agg_type,
            metrics=[f"{m.value}" if hasattr(m, "value") else f"{m}" for m in self.metrics],
        )

class BetaDiversityEstimator(BaseEstimator[BetaDiversityEstimates]):  # noqa: D101
    def __init__(self, target: str | StrEnum | None = None, metric: BetaDiversityMetric | str = BetaDiversityMetric.BRAYCURTIS):  # noqa: ANN204
        """Calculates between-group dissimilarity.
        Common metrics: 'braycurtis' (abundance-weighted), 'jaccard' (presence/absence).
        """  # noqa: D205
        self.target = f"{target.value}" if hasattr(target, "value") else (f"{target}" if target is not None else None)
        self.metric = f"{metric.value}" if hasattr(metric, "value") else f"{metric}"
        self._method_label = f"beta_diversity_{self.metric}"

    def get_params(self) -> dict[str, Any]:
        """Returns parameters for cloning compatibility."""
        return {"target": self.target, "metric": self.metric}

    def calculate(self, div_df: Any) -> BetaDiversityEstimates:  # type: ignore # noqa: ANN401, D102
        raw_meta = get_dataframe_metadata(div_df)
        meta = raw_meta.get("metric_meta", raw_meta) if isinstance(raw_meta, dict) else {}
        df = div_df.data if hasattr(div_df, "data") else div_df

        raw_target = meta.get("trait", self.target)
        if (not raw_target or raw_target == "unknown") and "target" in df.columns and len(df) > 0:
            if df["target"].n_unique() == 1:
                raw_target = df["target"][0]
            else:
                raw_target = "target"
        if not raw_target:
            raw_target = self.target

        target_col = (
            f"{raw_target.value}"
            if hasattr(raw_target, "value")
            else (f"{raw_target}" if raw_target is not None else None)
        )

        strata = [f"{s.value}" if hasattr(s, "value") else f"{s}" for s in meta.get("stratified_by", [])]
        if not strata:
            strata = [
                f"{c.value}" if hasattr(c, "value") else f"{c}"
                for c in df.columns
                if c not in {"target", "variant_count", "n_total"}
            ]

        if not target_col:
            raise ValueError("Target trait must be defined either in init or via accessor metadata.")
        if not strata:
            raise ValueError("Beta diversity requires at least one stratification level to compare groups.")

        # 2. Pivot the data into a Wide Matrix
        pivot_df = df.pivot(on="target", index=strata, values="variant_count", aggregate_function="sum").fill_null(0)

        target_cols = [c for c in pivot_df.columns if c not in strata]
        matrix_vals = pivot_df.select(target_cols).to_numpy()

        # 3. Calculate Pairwise Distances
        distances = pdist(matrix_vals, metric=self.metric)  # type: ignore
        dist_matrix = squareform(distances)
        dist_matrix = np.nan_to_num(dist_matrix, nan=0.0)

        # 4. Format the row/column names for the UI
        if len(strata) == 1:
            strata_names = [f"{x}" for x in pivot_df[strata[0]].to_list()]
        else:
            strata_names = [
                " | ".join(f"{row[col]}" for col in strata) for row in pivot_df.select(strata).iter_rows(named=True)
            ]

        # 5. Wrap back into an explicitly labeled Polars DataFrame
        result_matrix = pl.DataFrame(dist_matrix, schema=strata_names)

        adj_raw = meta.get("adjusted_for", "unknown")
        adj_str = (
            f"{adj_raw.value}" if hasattr(adj_raw, "value") else (f"{adj_raw}" if adj_raw is not None else "unknown")
        )

        agg_type = meta.get("aggregation_type", AggregationType.TRAIT)
        if hasattr(agg_type, "value"):
            agg_type = f"{agg_type.value}"
        else:
            agg_type = f"{agg_type}"

        return BetaDiversityEstimates(
            data=result_matrix,
            stratified_by=[f"{s.value}" if hasattr(s, "value") else f"{s}" for s in strata],
            adjusted_for=adj_str,
            trait=target_col,
            aggregation_type=agg_type,
            metric=f"{self.metric.value}" if hasattr(self.metric, "value") else f"{self.metric}",
        )

