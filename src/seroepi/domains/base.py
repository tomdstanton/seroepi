"""Base domain module providing base domain mixin, base plotter, and render_plot router."""

import weakref
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from importlib.resources import files
from json import load as json_load
from typing import Any
from warnings import warn

import polars as pl
from plotly.graph_objects import Figure

from seroepi.constants import Domain, PlotType


class WeakDataFrameRegistry(weakref.WeakKeyDictionary[Any, Any]):
    """A weak-reference registry mapping DataFrames and datasets to metadata dicts.

    Subclasses weakref.WeakKeyDictionary for type compatibility, using object IDs
    and weakref.finalize to automatically clean up entries when DataFrame instances
    are garbage collected.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, D107
        super().__init__(*args, **kwargs)
        self._data_by_id: dict[int, dict[str, Any]] = {}

    def register(self, df: Any, metadata: dict[str, Any]) -> None:  # noqa: ANN401
        """Registers a DataFrame or dataset instance with its metadata dictionary."""
        if not isinstance(metadata, dict):
            return
        if hasattr(df, "data") and isinstance(getattr(df, "data"), pl.DataFrame):
            df_obj = getattr(df, "data")
            df_id = id(df_obj)
            self._data_by_id[df_id] = metadata
            try:
                weakref.finalize(df_obj, self._data_by_id.pop, df_id, None)
            except (TypeError, AttributeError):
                pass

        if isinstance(df, pl.DataFrame):
            df_id = id(df)
            self._data_by_id[df_id] = metadata
            try:
                weakref.finalize(df, self._data_by_id.pop, df_id, None)
            except (TypeError, AttributeError):
                pass
        else:
            try:
                super().__setitem__(df, metadata)
            except Exception:
                pass
            df_id = id(df)
            self._data_by_id[df_id] = metadata

    def get(self, key: Any, default: Any = None) -> Any:  # noqa: ANN401
        """Gets metadata dict for a DataFrame, dataset, or object."""
        if key is None:
            return default
        if hasattr(key, "data") and isinstance(getattr(key, "data"), pl.DataFrame):
            data_meta = self._data_by_id.get(id(getattr(key, "data")), None)
            if data_meta is not None:
                return data_meta
        if isinstance(key, pl.DataFrame):
            return self._data_by_id.get(id(key), default)
        try:
            val = super().get(key, None)
            if val is not None:
                return val
        except Exception:
            pass
        return self._data_by_id.get(id(key), default)

    def __setitem__(self, key: Any, value: dict[str, Any]) -> None:  # noqa: ANN401, D105
        self.register(key, value)

    def __getitem__(self, key: Any) -> Any:  # noqa: ANN401, D105
        res = self.get(key, None)
        if res is None:
            raise KeyError(key)
        return res

    def __contains__(self, key: Any) -> bool:  # noqa: ANN401, D105
        if key is None:
            return False
        if hasattr(key, "data") and isinstance(getattr(key, "data"), pl.DataFrame):
            if id(getattr(key, "data")) in self._data_by_id:
                return True
        if isinstance(key, pl.DataFrame):
            return id(key) in self._data_by_id
        try:
            if super().__contains__(key):
                return True
        except Exception:
            pass
        return id(key) in self._data_by_id

    def clear(self) -> None:
        """Clears all registered metadata."""
        self._data_by_id.clear()
        try:
            super().clear()
        except Exception:
            pass


_DATAFRAME_METADATA_REGISTRY = WeakDataFrameRegistry()


def register_dataframe_metadata(df: Any, metadata: dict[str, Any]) -> None:  # noqa: ANN401
    """Associates a metadata dictionary with a Polars DataFrame or SeroEpiDataset instance."""
    if not isinstance(metadata, dict):
        return
    _DATAFRAME_METADATA_REGISTRY.register(df, metadata)
    if isinstance(df, pl.DataFrame) and hasattr(df, "__dict__"):
        try:
            df.__dict__["metadata"] = metadata
            df.__dict__["meta"] = metadata.get("metric_meta", metadata) if isinstance(metadata, dict) else metadata
        except Exception:
            pass
    elif hasattr(df, "data") and isinstance(getattr(df, "data"), pl.DataFrame):
        try:
            register_dataframe_metadata(getattr(df, "data"), metadata)
        except Exception:
            pass


def get_dataframe_metadata(obj: object) -> dict[str, Any]:
    """Retrieves metadata associated with a SeroEpiDataset, Polars DataFrame, or generic container."""
    if obj is None:
        return {}

    # For Polars DataFrame (or MetaDataFrame), look up directly in instance attributes or registry
    if isinstance(obj, pl.DataFrame):
        if hasattr(obj, "_metadata") and isinstance(getattr(obj, "_metadata"), dict):
            return getattr(obj, "_metadata")
        try:
            obj_dict = getattr(obj, "__dict__", {})
            if isinstance(obj_dict, dict) and "metadata" in obj_dict and isinstance(obj_dict["metadata"], dict):
                return obj_dict["metadata"]
        except Exception:
            pass
        reg_meta = _DATAFRAME_METADATA_REGISTRY.get(obj, {})
        return reg_meta if isinstance(reg_meta, dict) else {}

    # For SeroEpiDataset or other wrappers, check underlying data DataFrame or metadata attribute
    try:
        if hasattr(obj, "data") and isinstance(getattr(obj, "data"), pl.DataFrame):
            inner_meta = get_dataframe_metadata(getattr(obj, "data"))
            if inner_meta:
                return inner_meta
        if hasattr(obj, "metadata") and isinstance(getattr(obj, "metadata"), dict):
            meta_val = getattr(obj, "metadata")
            if isinstance(meta_val, dict) and meta_val:
                return meta_val
    except Exception:
        pass

    reg_meta = _DATAFRAME_METADATA_REGISTRY.get(obj, {})
    return reg_meta if isinstance(reg_meta, dict) else {}


def get_dataframe_meta(obj: object) -> dict[str, Any]:
    """Retrieves the metric_meta dictionary associated with a container or DataFrame."""
    if isinstance(obj, pl.DataFrame) and hasattr(obj, "_meta") and isinstance(getattr(obj, "_meta"), dict):
        return getattr(obj, "_meta")
    raw = get_dataframe_metadata(obj)
    if isinstance(raw, dict):
        return raw.get("metric_meta", raw)
    return {}


def set_dataframe_meta(obj: pl.DataFrame, meta: dict[str, Any]) -> None:
    """Sets inner meta dictionary on a Polars DataFrame."""
    if hasattr(obj, "__dict__"):
        try:
            obj.__dict__["meta"] = meta
        except Exception:
            pass


try:
    setattr(pl.DataFrame, "metadata", property(fget=get_dataframe_metadata, fset=register_dataframe_metadata))
    setattr(pl.DataFrame, "meta", property(fget=get_dataframe_meta, fset=set_dataframe_meta))
except Exception:
    pass


class MetaDataFrame(pl.DataFrame):
    """Polars DataFrame subclass supporting dynamic .metadata and .meta attributes."""

    def __init__(self, data: Any = None, metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:  # noqa: ANN401
        """Initializes a MetaDataFrame with associated metadata."""
        super().__init__(data, **kwargs)
        meta = metadata if metadata is not None else {}
        object.__setattr__(self, "_metadata", meta)
        object.__setattr__(self, "_meta", meta.get("metric_meta", meta) if isinstance(meta, dict) else meta)
        register_dataframe_metadata(self, meta)

    @property
    def metadata(self) -> dict[str, Any]:
        """The metadata dictionary."""
        return getattr(self, "_metadata", {})

    @metadata.setter
    def metadata(self, value: dict[str, Any]) -> None:
        """Sets the metadata dictionary."""
        object.__setattr__(self, "_metadata", value)
        object.__setattr__(self, "_meta", value.get("metric_meta", value) if isinstance(value, dict) else value)
        register_dataframe_metadata(self, value)

    @property
    def meta(self) -> dict[str, Any]:
        """The metric_meta dictionary."""
        return getattr(self, "_meta", {})

    @meta.setter
    def meta(self, value: dict[str, Any]) -> None:
        """Sets the metric_meta dictionary."""
        object.__setattr__(self, "_meta", value)


class BaseDomainMixin:
    """Base domain mixin with __slots__ = () providing DataFrame access and dataset wrapping."""

    __slots__ = ()

    @property
    def _df(self) -> pl.DataFrame:
        if hasattr(self, "_df_obj") and isinstance(getattr(self, "_df_obj"), pl.DataFrame):
            return getattr(self, "_df_obj")
        if hasattr(self, "data") and isinstance(getattr(self, "data"), pl.DataFrame):
            return getattr(self, "data")
        if isinstance(self, pl.DataFrame):
            return self
        raise AttributeError("Domain mixin expected 'data' attribute (SeroEpiDataset) or pl.DataFrame instance.")

    def _resolve_col(self, col: str | StrEnum, df_cols: Sequence[str] | None = None) -> str:
        """Resolves exact, unprefixed, or prepended column names against DataFrame columns."""
        col_str = col.value if hasattr(col, "value") else str(col)
        available_cols = list(df_cols) if df_cols is not None else list(self._df.columns)

        if col_str in available_cols:
            return col_str

        prefixes = [
            f"{Domain.GENOTYPE.value}_",
            f"{Domain.PHENOTYPE.value}_",
            f"{Domain.AMR.value}_",
            f"{Domain.VIRULENCE.value}_",
            f"{Domain.SPATIAL.value}_",
            f"{Domain.SPATIAL_RES.value}_",
            f"{Domain.TEMPORAL.value}_",
            f"{Domain.QC.value}_",
        ]

        for prefix in prefixes:
            if col_str.startswith(prefix):
                unprefixed = col_str[len(prefix) :]
                if unprefixed in available_cols:
                    return unprefixed

        for prefix in prefixes:
            prepended = f"{prefix}{col_str}"
            if prepended in available_cols:
                return prepended

        for c in available_cols:
            for prefix in prefixes:
                if c.startswith(prefix) and c[len(prefix) :] == col_str:
                    return c

        return col_str

    def _wrap_result(self, res_df: pl.DataFrame, meta_dict: dict[str, Any] | None = None) -> Any:  # noqa: ANN401
        from seroepi.dataset import SeroEpiDataset

        meta = meta_dict or {}
        parent_meta = {}

        if hasattr(self, "metadata") and isinstance(getattr(self, "metadata"), dict):
            parent_meta = dict(getattr(self, "metadata"))
        elif hasattr(self, "_df_obj"):
            parent_meta = dict(get_dataframe_metadata(getattr(self, "_df_obj")))
        elif hasattr(self, "data"):
            parent_meta = dict(get_dataframe_metadata(getattr(self, "data")))

        final_meta = dict(parent_meta) if parent_meta else {}
        final_meta.update(meta)

        if isinstance(res_df, MetaDataFrame):
            res_df.metadata = final_meta
            meta_df = res_df
        else:
            meta_df = MetaDataFrame(res_df, metadata=final_meta)

        register_dataframe_metadata(meta_df, final_meta)
        register_dataframe_metadata(res_df, final_meta)

        if hasattr(self, "data") and hasattr(self, "name") and hasattr(self, "metadata"):
            return SeroEpiDataset(
                data=meta_df,
                name=getattr(self, "name"),
                metadata=final_meta,
            )

        return meta_df


class BasePlotter(ABC):
    """Stateless base class for all plotting engines in seroepi."""

    _MAIN_COLOUR = "#0EA5E9"
    _CI_COLOUR = "rgba(14, 165, 233, 0.2)"
    _ACCENT_COLOUR = "#EC4899"  # Vibrant Neon Pink
    _FONT_COLOUR = "#94A3B8"  # Slate 400
    _GRID_COLOUR = "rgba(148, 163, 184, 0.2)"  # Subtle translucent grid lines
    _WORLD_GEOJSON = None

    SUPPORTED_TYPES = ()

    @classmethod
    def _get_world_geojson(cls) -> dict[str, Any]:
        """Lazily loads and caches the internal world boundaries GeoJSON."""
        if cls._WORLD_GEOJSON is None:
            try:
                geojson_path = files("seroepi.data").joinpath("world_boundaries.geojson")
                with geojson_path.open(mode="r", encoding="utf-8") as f:
                    cls._WORLD_GEOJSON = json_load(f)
            except Exception as e:
                warn(f"Could not load internal world boundaries. Ensure the file exists: {e}")
                cls._WORLD_GEOJSON = {}
        return cls._WORLD_GEOJSON

    @classmethod
    def can_render(cls, result_obj: Any) -> bool:  # noqa: ANN401
        """Checks if the incoming result object is supported by this plotter."""
        return isinstance(result_obj, cls.SUPPORTED_TYPES)

    @classmethod
    def _clean_label(cls, col_name: str) -> str:
        """Strips domain prefixes for clean UI rendering."""
        if not isinstance(col_name, str):
            return str(col_name)
        for domain in [Domain.GENOTYPE.value, Domain.PHENOTYPE.value, Domain.AMR.value, Domain.VIRULENCE.value]:
            prefix = f"{domain}_"
            if col_name.startswith(prefix):
                return col_name.replace(prefix, "").replace("_", " ")
        return col_name.replace("_", " ")

    @classmethod
    def get_colorscale(cls, transparent: bool = True) -> list[Any]:
        """Returns the standard Cyberpunk continuous color scale."""
        base_color = "rgba(0,0,0,0)" if transparent else "rgba(15, 23, 42, 0.4)"
        return [
            [0.0, base_color],
            [0.4, "#8B5CF6"],  # Deep Purple
            [0.7, cls._MAIN_COLOUR],  # Electric Cyan
            [1.0, cls._ACCENT_COLOUR],  # Neon Pink
        ]

    @classmethod
    def apply_theme(cls, fig: Figure) -> Figure:
        """Applies a universal transparent theme optimized for both light and dark web app modes."""
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=cls._FONT_COLOUR),
            hoverlabel=dict(bgcolor="black", bordercolor=cls._MAIN_COLOUR, font_size=14, font_color="white"),
        )
        fig.update_xaxes(gridcolor=cls._GRID_COLOUR, zerolinecolor=cls._GRID_COLOUR, linecolor=cls._GRID_COLOUR)
        fig.update_yaxes(gridcolor=cls._GRID_COLOUR, zerolinecolor=cls._GRID_COLOUR, linecolor=cls._GRID_COLOUR)
        return fig

    @classmethod
    @abstractmethod
    def render(cls, result_obj: Any, **kwargs: Any) -> Figure:  # noqa: ANN401, D102
        pass


_PLOTTER_MAP: dict[PlotType, type[BasePlotter]] = {}


def register_plotter(plot_type: PlotType, plotter_cls: type[BasePlotter]) -> None:
    """Registers a concrete plotter class for a specific PlotType."""
    _PLOTTER_MAP[plot_type] = plotter_cls


def render_plot(result_obj: Any, plot_type: PlotType, **kwargs) -> Figure:  # noqa: ANN003, ANN401
    """A central router that invokes the correct plotter for the desired plot type."""
    if not _PLOTTER_MAP:
        import seroepi.domains.epi  # noqa: F401
        import seroepi.domains.geno  # noqa: F401
        import seroepi.domains.geo  # noqa: F401

    if plotter := _PLOTTER_MAP.get(plot_type, None):
        return plotter.render(result_obj, **kwargs)
    available = list(_PLOTTER_MAP.keys())
    raise ValueError(f"Plot type '{plot_type}' is not registered. Available: {available}")
