"""Genomic and phenotypic domain mixin and co-located plotters for genetic diversity, traits, and distance networks."""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl

from seroepi import estimators
from seroepi.constants import DistanceEpidemiologicalDomain, Domain, PlotType
from seroepi.dist import DistancesBase
from seroepi.domains.base import BaseDomainMixin, BasePlotter, register_plotter


class GenoMixin(BaseDomainMixin):
    """Mixin class providing genomic, phenotypic, and AMR dataset operations with __slots__ = ()."""

    __slots__ = ()

    @property
    def genotype(self) -> pl.DataFrame:
        """Returns Core Genotype matrix with prefix removed."""  # noqa: D421
        df = self._df
        prefix = f"{Domain.GENOTYPE.value}_"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            return pl.DataFrame()
        return df.select(cols).rename({c: c.replace(prefix, "", 1) for c in cols})

    @property
    def phenotype(self) -> pl.DataFrame:
        """Returns Phenotype matrix with prefix removed."""  # noqa: D421
        df = self._df
        prefix = f"{Domain.PHENOTYPE.value}_"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            return pl.DataFrame()
        return df.select(cols).rename({c: c.replace(prefix, "", 1) for c in cols})

    @property
    def amr(self) -> pl.DataFrame:
        """Returns AMR determinant matrix with prefix removed."""  # noqa: D421
        df = self._df
        prefix = f"{Domain.AMR.value}_"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            return pl.DataFrame()
        return df.select(cols).rename({c: c.replace(prefix, "", 1) for c in cols})

    @property
    def virulence(self) -> pl.DataFrame:
        """Returns Virulence marker matrix with prefix removed."""  # noqa: D421
        df = self._df
        prefix = f"{Domain.VIRULENCE.value}_"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            return pl.DataFrame()
        return df.select(cols).rename({c: c.replace(prefix, "", 1) for c in cols})

    def _resolve_trait_col(self, t: str | StrEnum, domain_val: str, df_cols: set[str]) -> str | None:
        """Resolves a target trait name against DataFrame columns by checking exact, domain-preferred, and multi-domain matches."""
        t_str = str(t.value if hasattr(t, "value") else t)
        if t_str in df_cols:
            return t_str

        prefix = f"{domain_val}_"
        if not t_str.startswith(prefix):
            prepended = f"{prefix}{t_str}"
            if prepended in df_cols:
                return prepended

        resolved = self._resolve_col(t_str, list(df_cols))
        if resolved in df_cols:
            return resolved

        return None

    def has_any(self, traits: Sequence[str | StrEnum] | str | StrEnum, domain: str | Domain = Domain.AMR) -> pl.Series:
        """Checks if isolates possess ANY of the specified traits."""
        if isinstance(traits, (str, StrEnum)):
            traits = [traits]
        df = self._df
        domain_val = str(domain.value if hasattr(domain, "value") else domain)
        df_cols = set(df.columns)
        trait_cols: list[str] = []
        for t in traits:
            col = self._resolve_trait_col(t, domain_val, df_cols)
            if col:
                trait_cols.append(col)

        if not trait_cols:
            return pl.Series("has_any", [False] * len(df), dtype=pl.Boolean)
        return df.select(pl.any_horizontal([pl.col(c) for c in trait_cols]).alias("has_any")).get_column("has_any")

    def has_all(self, traits: Sequence[str | StrEnum] | str | StrEnum, domain: str | Domain = Domain.VIRULENCE) -> pl.Series:
        """Checks if isolates possess ALL of the specified traits."""
        if isinstance(traits, (str, StrEnum)):
            traits = [traits]
        df = self._df
        domain_val = str(domain.value if hasattr(domain, "value") else domain)
        df_cols = set(df.columns)
        trait_cols: list[str] = []
        for t in traits:
            col = self._resolve_trait_col(t, domain_val, df_cols)
            if col:
                trait_cols.append(col)
            else:
                return pl.Series("has_all", [False] * len(df), dtype=pl.Boolean)

        if not trait_cols:
            return pl.Series("has_all", [True] * len(df), dtype=pl.Boolean)
        return df.select(pl.all_horizontal([pl.col(c) for c in trait_cols]).alias("has_all")).get_column("has_all")

    def has_gene(self, gene_col: str | StrEnum, gene_name: str) -> pl.Series:
        """Searches for a specific gene within a column."""
        return self._df.get_column(str(gene_col)).cast(pl.Utf8).str.contains(gene_name, literal=True)

    def sort_loci(self, locus_col: str | StrEnum) -> Any:  # noqa: ANN401
        """Sorts DataFrame numerically by locus (e.g. K2 before K10)."""
        res_df = (
            self._df.with_columns(
                pl.col(str(locus_col)).str.extract(r"(\d+)").cast(pl.Float64, strict=False).alias("_sort_key")
            )
            .sort("_sort_key")
            .drop("_sort_key")
        )
        return self._wrap_result(res_df)


@pl.api.register_dataframe_namespace("geno")
class GenoAccessor(GenoMixin):
    """Polars DataFrame namespace for genetic and trait-based operations."""

    def __init__(self, polars_obj: pl.DataFrame):  # noqa: ANN204, D107
        self._df_obj = polars_obj


class AlphaDiversityPlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (estimators.AlphaDiversityEstimates,)

    @classmethod
    def render(  # type: ignore # noqa: D102
        cls,
        result: estimators.AlphaDiversityEstimates,
        metric: str = "shannon",
        sort_ascending: bool = False,
        **kwargs,  # noqa: ANN003
    ) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")

        if metric not in result.metrics:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {result.metrics}")

        data = result.data

        group_col = result.stratified_by[0] if result.stratified_by else "Global"

        if sort_ascending is not None and metric in data.columns:
            data = data.sort(metric, descending=not sort_ascending)

        x_vals = data[group_col].to_list() if group_col in data.columns else ["Global"] * len(data)
        y_vals = data[metric].to_numpy() if metric in data.columns else np.zeros(len(data))
        n_samples = data["n_samples"].to_numpy() if "n_samples" in data.columns else np.zeros(len(data))

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=y_vals,
                width=0.02,
                marker_color="rgba(14, 165, 233, 0.4)",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(size=12, color=cls._MAIN_COLOUR),
                name=metric.title(),
                hovertemplate=f"<b>%{{x}}</b><br>{metric.title()}: %{{y:.3f}}<br>Sequenced (n): %{{customdata}}<extra></extra>",  # noqa: E501
                customdata=n_samples,
            )
        )

        return cls.apply_theme(
            fig.update_layout(
                title=f"<b>Alpha Diversity ({metric.title()})</b><br><sup>Trait: {cls._clean_label(result.trait)}</sup>",  # noqa: E501
                xaxis=dict(title=group_col.title() if group_col != "Global" else ""),
                yaxis=dict(title=f"{metric.title()} Index", rangemode="tozero"),
                showlegend=False,
            )
        )


class BetaHeatmapPlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (estimators.BetaDiversityEstimates,)

    @classmethod
    def render(cls, result: estimators.BetaDiversityEstimates, mask_upper: bool = True, **kwargs) -> go.Figure:  # type: ignore # noqa: ANN003, D102
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")

        dist_matrix = result.data
        labels = dist_matrix.columns
        matrix_vals = dist_matrix.to_numpy()

        if mask_upper:
            mask = np.triu(np.ones(matrix_vals.shape, dtype=bool))
            matrix_vals = np.where(mask, np.nan, matrix_vals)

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix_vals,
                x=labels,
                y=labels,
                colorscale=cls.get_colorscale(transparent=True),
                zmin=0,
                zmax=1,
                xgap=1,
                ygap=1,
                hoverongaps=False,
                hovertemplate="<b>Group 1:</b> %{y}<br><b>Group 2:</b> %{x}<br><b>Dissimilarity:</b> %{z:.3f}<extra></extra>",  # noqa: E501
                colorbar=dict(title=f"{result.metric.title()}<br>Distance"),
            )
        )

        return cls.apply_theme(
            fig.update_layout(
                title=f"<b>Beta Diversity ({result.metric.title()})</b><br><sup>Trait: {cls._clean_label(result.trait)}</sup>",  # noqa: E501
                xaxis=dict(title="", tickangle=45),
                yaxis=dict(title="", autorange="reversed"),
                margin=dict(l=20, r=20, t=60, b=80),
                width=700,
                height=700,
            )
        )


class NetworkPlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (DistancesBase,)

    @staticmethod
    def _build_edges(rows: np.ndarray, cols: np.ndarray, pos: np.ndarray) -> tuple[list[Any], list[Any]]:
        if len(rows) == 0:
            return [], []
        ex = np.full(len(rows) * 3, None, dtype=object)
        ex[0::3], ex[1::3] = pos[rows, 0], pos[cols, 0]

        ey = np.full(len(rows) * 3, None, dtype=object)
        ey[0::3], ey[1::3] = pos[rows, 1], pos[cols, 1]
        return ex.tolist(), ey.tolist()

    @classmethod
    def render(  # type: ignore # noqa: D102
        cls,
        result: DistancesBase,
        df: Any = None,  # noqa: ANN401
        pos: np.ndarray | None = None,
        edge_type: str = "snp",
        threshold: int = 20,
        color_col: str | None = None,
        trans_network: DistancesBase | None = None,
        **kwargs,  # noqa: ANN003
    ) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")

        dense_dist = result.matrix.toarray().astype(float)

        if pos is None:
            pos = result.layout(threshold=threshold if edge_type == "snp" else None)

        sample_ids = result.index.to_list()
        df_in = df.data if hasattr(df, "data") else df

        if df_in is not None and "sample_id" in df_in.columns:
            aligned_df = pl.DataFrame({"sample_id": sample_ids}).join(df_in, on="sample_id", how="left")
        else:
            aligned_df = pl.DataFrame({"sample_id": sample_ids})

        edge_x, edge_y = [], []
        title = "<b>Isolate Network</b><br><sup>Nodes positioned by force-directed layout (Spring)</sup>"

        if edge_type == "snp" and getattr(result, "metric_type", None) in [
            DistanceEpidemiologicalDomain.ABSOLUTE_DISTANCE,
            DistanceEpidemiologicalDomain.RELATIVE_DISTANCE,
        ]:
            adj = dense_dist <= threshold
            rows, cols = np.where(np.triu(adj, k=1))
            edge_x, edge_y = cls._build_edges(rows, cols, pos)
            title = f"<b>Genomic SNP Network</b><br><sup>Edges connect isolates ≤ {threshold} SNPs apart</sup>"

        elif edge_type == "trans":
            net = trans_network if trans_network is not None else result
            if getattr(net, "metric_type", None) in [
                DistanceEpidemiologicalDomain.ABSOLUTE_SIMILARITY,
                DistanceEpidemiologicalDomain.RELATIVE_SIMILARITY,
            ]:
                from scipy.sparse import triu

                upper_adj = triu(net.matrix, k=1).tocoo()
                edge_x, edge_y = cls._build_edges(upper_adj.row, upper_adj.col, pos)
                title = "<b>Transmission Network</b><br><sup>Edges connect isolates based on spatial/temporal proximity</sup>"  # noqa: E501

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.4, color="#888"), hoverinfo="none", mode="lines")

        if color_col and color_col in aligned_df.columns:
            color_vals = [str(x) if x is not None else "Unknown" for x in aligned_df[color_col].to_list()]
            unique_colors = list(dict.fromkeys(color_vals))
            color_map = {c: i for i, c in enumerate(unique_colors)}
            colors = [color_map[c] for c in color_vals]
            hover_text = [f"ID: {idx}<br>{color_col}: {c}" for idx, c in zip(sample_ids, color_vals)]
            marker_dict = dict(
                showscale=False, color=colors, colorscale="Turbo", size=10, line=dict(width=1, color="white")
            )
        else:
            hover_text = [f"ID: {idx}" for idx in sample_ids]
            marker_dict = dict(color=cls._MAIN_COLOUR, size=10, line=dict(width=1, color="white"))

        node_trace = go.Scatter(
            x=pos[:, 0], y=pos[:, 1], mode="markers", hovertext=hover_text, hoverinfo="text", marker=marker_dict
        )

        return cls.apply_theme(
            go.Figure(data=[edge_trace, node_trace]).update_layout(
                title=title,
                showlegend=False,
                hovermode="closest",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(b=20, l=5, r=5, t=60),
            )
        )


register_plotter(PlotType.ALPHA_DIVERSITY, AlphaDiversityPlotter)
register_plotter(PlotType.BETA_HEATMAP, BetaHeatmapPlotter)
register_plotter(PlotType.NETWORK, NetworkPlotter)
