"""Core SeroEpi dataset container wrapping Polars DataFrames with Patito validation."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from seroepi.domains import EpiMixin, GenoMixin, GeoMixin, QcMixin


@dataclass(frozen=True, slots=True)
class SeroEpiDataset(GeoMixin, EpiMixin, GenoMixin, QcMixin):
    """Frozen, slotted dataclass wrapping a Polars DataFrame validated with Patito schemas.

    Attributes:
        data: The underlying Polars DataFrame containing isolate records.
        name: Name identifier for the dataset.
        metadata: Dictionary containing dataset-level metadata.
    """

    data: pl.DataFrame | pl.LazyFrame
    name: str = "SeroEpi Dataset"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:  # noqa: D105
        if not isinstance(self.data, (pl.DataFrame, pl.LazyFrame)):
            raise TypeError(f"SeroEpiDataset expected pl.DataFrame or pl.LazyFrame for 'data', got {type(self.data).__name__}")

        if self.metadata is None:
            object.__setattr__(self, "metadata", {})
        else:
            object.__setattr__(self, "metadata", dict(self.metadata))
        if not isinstance(self.name, str):
            object.__setattr__(self, "name", str(self.name))

        if isinstance(self.data, pl.LazyFrame):
            return

        from seroepi.domains.base import MetaDataFrame

        # Attempt Patito schema validation if SampleModel is available
        try:
            from patito.exceptions import DataFrameValidationError
            from seroepi.io import SampleModel

            validated_df = SampleModel.validate(self.data)
            meta_df = MetaDataFrame(validated_df, metadata=self.metadata)
            object.__setattr__(self, "data", meta_df)
        except DataFrameValidationError:
            raise
        except Exception:
            # If SampleModel validation fails for non-validation reasons (e.g. missing optional columns), wrap original data
            meta_df = MetaDataFrame(self.data, metadata=self.metadata)
            object.__setattr__(self, "data", meta_df)

    @property
    def metric_meta(self) -> dict[str, Any]:
        """Returns the inner metric_meta dictionary or metadata dictionary."""
        if isinstance(self.metadata, dict):
            return self.metadata.get("metric_meta", self.metadata)
        return {}

    @property
    def meta(self) -> dict[str, Any]:
        """Metric metadata dictionary."""
        return self.metric_meta

    @classmethod
    def from_kaptive(
        cls,
        results: "Iterable[Any]",
        name: str = "Kaptive Dataset",
        metadata: dict[str, Any] | None = None,
    ) -> "SeroEpiDataset":
        """Instantiates a SeroEpiDataset from an iterable of Kaptive SerotypingResult objects."""
        from seroepi.io import KaptiveNativeParser

        df = KaptiveNativeParser.from_results(results, dataset_name=name)
        return cls(data=df, name=name, metadata=metadata or {})

    # --- Domain Accessor Properties (for backward compatibility) ---
    @property
    def geo(self) -> "SeroEpiDataset":
        """Geo domain accessor forwarding to self."""
        return self

    @property
    def epi(self) -> "SeroEpiDataset":
        """Epi domain accessor forwarding to self."""
        return self

    @property
    def geno(self) -> "SeroEpiDataset":
        """Geno domain accessor forwarding to self."""
        return self

    @property
    def qc(self) -> "SeroEpiDataset":
        """QC domain accessor forwarding to self."""
        return self

    # --- DataFrame Container Delegation ---
    def __len__(self) -> int:  # noqa: D105
        if isinstance(self.data, pl.LazyFrame):
            raise TypeError("Cannot determine length of a LazyFrame. Call .collect() first.")
        return len(self.data)

    def __getitem__(self, item: Any) -> Any:  # noqa: ANN401, D105
        if isinstance(self.data, pl.LazyFrame):
            raise TypeError("Cannot use __getitem__ on a LazyFrame. Call .collect() first.")
        return self.data[item]

    @property
    def shape(self) -> tuple[int, int]:  # noqa: D102
        if isinstance(self.data, pl.LazyFrame):
            raise TypeError("Cannot determine shape of a LazyFrame. Call .collect() first.")
        return self.data.shape

    @property
    def columns(self) -> list[str]:  # noqa: D102
        return self.data.columns

    @property
    def schema(self) -> pl.Schema:  # noqa: D102
        return self.data.schema

    # --- Immutability-Preserving Transformation Methods ---
    def filter(self, *predicates: Any, **constraints: Any) -> "SeroEpiDataset":  # noqa: ANN401
        """Filters the underlying Polars DataFrame and returns a new SeroEpiDataset."""
        new_df = self.data.filter(*predicates, **constraints)
        return SeroEpiDataset(data=new_df, name=self.name, metadata=dict(self.metadata))

    def select(self, *exprs: Any, **named_exprs: Any) -> "SeroEpiDataset":  # noqa: ANN401
        """Selects columns from the underlying Polars DataFrame and returns a new SeroEpiDataset."""
        new_df = self.data.select(*exprs, **named_exprs)
        return SeroEpiDataset(data=new_df, name=self.name, metadata=dict(self.metadata))

    def with_columns(self, *exprs: Any, **named_exprs: Any) -> "SeroEpiDataset":  # noqa: ANN401
        """Adds or updates columns in the underlying Polars DataFrame and returns a new SeroEpiDataset."""
        new_df = self.data.with_columns(*exprs, **named_exprs)
        return SeroEpiDataset(data=new_df, name=self.name, metadata=dict(self.metadata))

    def head(self, n: int = 5) -> "SeroEpiDataset":
        """Returns a new SeroEpiDataset containing the first n rows."""
        return SeroEpiDataset(data=self.data.head(n), name=self.name, metadata=dict(self.metadata))

    def tail(self, n: int = 5) -> "SeroEpiDataset":
        """Returns a new SeroEpiDataset containing the last n rows."""
        return SeroEpiDataset(data=self.data.tail(n), name=self.name, metadata=dict(self.metadata))

    def pipe(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Pipes the underlying Polars DataFrame through a function."""
        res = self.data.pipe(func, *args, **kwargs)
        if isinstance(res, (pl.DataFrame, pl.LazyFrame)):
            return SeroEpiDataset(data=res, name=self.name, metadata=dict(self.metadata))
        return res

    def collect(self) -> "SeroEpiDataset":
        """Executes the lazy frame and returns a new eager dataset."""
        if isinstance(self.data, pl.LazyFrame):
            return SeroEpiDataset(data=self.data.collect(), name=self.name, metadata=dict(self.metadata))
        return self

    def lazy(self) -> "SeroEpiDataset":
        """Returns a new lazy dataset for out-of-core execution."""
        if isinstance(self.data, pl.DataFrame):
            return SeroEpiDataset(data=self.data.lazy(), name=self.name, metadata=dict(self.metadata))
        return self

    # --- Validation & Serialization Helpers ---
    def validate(self, schema: Any = None) -> "SeroEpiDataset":  # noqa: ANN401
        """Validates dataset using Patito schema model."""
        if schema is None:
            from seroepi.io import SampleModel

            schema = SampleModel
        validated_df = schema.validate(self.data)
        return SeroEpiDataset(data=pl.DataFrame(validated_df), name=self.name, metadata=dict(self.metadata))

    def to_polars(self) -> pl.DataFrame:
        """Returns the underlying Polars DataFrame."""
        return self.data

    def to_dicts(self) -> list[dict[str, Any]]:
        """Returns dataset records as a list of dictionaries."""
        return self.data.to_dicts()

    def to_dict(self) -> dict[str, Any]:
        """Serializes dataset into dictionary format."""
        return {
            "dataset_name": self.name,
            "records": self.data.to_dicts(),
            "metadata": self.metadata,
        }
