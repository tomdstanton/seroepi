"""Quality control domain mixin for dataset filtering and summary metrics."""

from enum import StrEnum
from typing import Any

import polars as pl

from seroepi.constants import Domain
from seroepi.domains.base import BaseDomainMixin


class QcMixin(BaseDomainMixin):
    """Mixin class providing quality control dataset operations with __slots__ = ()."""

    __slots__ = ()

    @property
    def metrics(self) -> pl.DataFrame:
        """Returns QC metrics matrix with prefix removed."""  # noqa: D421
        df = self._df
        prefix = f"{Domain.QC.value}_"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            return pl.DataFrame()
        return df.select(cols).rename({c: c.replace(prefix, "", 1) for c in cols})

    def filter_assemblies(
        self,
        min_n50: int = 10000,
        max_contigs: int = 500,
        require_species: str | StrEnum | None = None,
    ) -> Any:  # noqa: ANN401
        """Filters genomes based on quality thresholds."""
        df = self._df
        exprs = []

        if f"{Domain.QC.value}_N50" in df.columns:
            n50 = pl.col(f"{Domain.QC.value}_N50").cast(pl.Float64, strict=False)
            exprs.append((n50 >= min_n50) | n50.is_null())

        if f"{Domain.QC.value}_contig_count" in df.columns:
            contigs = pl.col(f"{Domain.QC.value}_contig_count").cast(pl.Float64, strict=False)
            exprs.append((contigs <= max_contigs) | contigs.is_null())

        if require_species and f"{Domain.QC.value}_species" in df.columns:
            req_species_str = str(require_species)
            exprs.append(pl.col(f"{Domain.QC.value}_species").str.to_lowercase().str.contains(req_species_str.lower()))

        if not exprs:
            return self._wrap_result(df)

        combined = exprs[0]
        for e in exprs[1:]:
            combined = combined & e

        return self._wrap_result(df.filter(combined))

    def report(self) -> dict[str, Any]:
        """Generates summary dictionary of dataset quality."""
        metrics_df = self.metrics
        rep = {}
        if "QC_warnings" in metrics_df.columns:
            rep["Total Warnings"] = metrics_df.filter(pl.col("QC_warnings") != "-").height
        if "N50" in metrics_df.columns:
            rep["Median N50"] = metrics_df.select(pl.col("N50").cast(pl.Float64, strict=False).median()).item()
        return rep


@pl.api.register_dataframe_namespace("qc")
class QCAccessor(QcMixin):
    """Polars DataFrame namespace for quality control operations."""

    def __init__(self, polars_obj: pl.DataFrame):  # noqa: ANN204, D107
        self._df_obj = polars_obj
