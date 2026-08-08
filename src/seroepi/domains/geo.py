"""Geospatial domain mixin and co-located plotters for spatial operations and visualization."""

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from shapely import from_geojson, points
from shapely.strtree import STRtree

from seroepi import estimators
from seroepi.constants import AggregationType, Domain, PlotType, SpatialResolution
from seroepi.data.gazetteer_data import GAZETTEER_DICT
from seroepi.domains.base import BaseDomainMixin, BasePlotter, register_plotter


def _parse_year(val: Any) -> int | None:
    """Helper to parse arbitrary year inputs (int, float, str, date, datetime) to integer year."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        import math

        if math.isnan(val):
            return None
        return int(val)
    if hasattr(val, "year"):
        return getattr(val, "year")
    val_str = str(val).strip()
    if not val_str:
        return None
    try:
        return int(float(val_str))
    except (ValueError, TypeError):
        return None


class GeoMixin(BaseDomainMixin):
    """Mixin class providing geographical dataset operations with __slots__ = ()."""

    __slots__ = ()

    _gazetteer_df: pl.DataFrame | None = None

    @property
    def gazetteer(self) -> pl.DataFrame:
        """Returns the internal gazetteer used for coordinate imputation as a Polars DataFrame."""  # noqa: D421
        if GeoMixin._gazetteer_df is None:
            records = [{"country": k, **v} for k, v in GAZETTEER_DICT.items()]
            GeoMixin._gazetteer_df = pl.DataFrame(records)
        return GeoMixin._gazetteer_df

    @property
    def spatial(self) -> pl.DataFrame:
        """Returns a DataFrame of all spatial columns, with the prefix removed."""  # noqa: D421
        df = self._df
        prefix = f"{Domain.SPATIAL.value}_"
        res_prefix = f"{Domain.SPATIAL_RES.value}_"
        cols = [c for c in df.columns if c.startswith(prefix) and not c.startswith(res_prefix)]
        if not cols:
            return pl.DataFrame()
        return df.select(cols).rename({c: c.replace(prefix, "", 1) for c in cols})

    @property
    def spatial_resolution(self) -> pl.DataFrame:
        """Returns a DataFrame of all spatial resolution columns."""  # noqa: D421
        df = self._df
        prefix = f"{Domain.SPATIAL_RES.value}_"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            return pl.DataFrame()
        return df.select(cols).rename({c: c.replace(prefix, "", 1) for c in cols})

    def standardize_and_impute(self, spatial_col: str | StrEnum | None = None) -> Any:  # noqa: ANN401
        """Standardizes spatial names and imputes missing coordinates.

        Uses the internal gazetteer to find centroids for countries when exact
        latitude and longitude are missing.
        """
        df = self._df
        prefix = f"{Domain.SPATIAL.value}_"
        res_prefix = f"{Domain.SPATIAL_RES.value}_"
        spatial_cols = [c for c in df.columns if c.startswith(prefix) and not c.startswith(res_prefix)]
        if not spatial_cols:
            return self._wrap_result(df)

        if spatial_col is None:
            spatial_col = spatial_cols[0]
        else:
            spatial_col_str = str(spatial_col)
            if spatial_col_str in df.columns:
                spatial_col = spatial_col_str
            elif f"{prefix}{spatial_col_str}" in df.columns:
                spatial_col = f"{prefix}{spatial_col_str}"
            else:
                spatial_col = spatial_col_str

        res_col = spatial_col.replace(prefix, res_prefix, 1)

        # Ensure latitude and longitude exist
        df_exprs = []
        if "latitude" not in df.columns:
            df_exprs.append(pl.lit(None).cast(pl.Float64).alias("latitude"))
        if "longitude" not in df.columns:
            df_exprs.append(pl.lit(None).cast(pl.Float64).alias("longitude"))
        if df_exprs:
            df = df.with_columns(df_exprs)

        gaz = self.gazetteer
        df_clean = df.with_columns(pl.col(spatial_col).cast(pl.Utf8).str.strip_chars().alias("_clean_spatial"))
        joined = df_clean.join(gaz, left_on="_clean_spatial", right_on="country", how="left")

        lat_expr = pl.coalesce([pl.col("latitude"), pl.col("centroid_lat")]).cast(pl.Float64)
        lon_expr = pl.coalesce([pl.col("longitude"), pl.col("centroid_lon")]).cast(pl.Float64)

        existing_res = pl.col(res_col) if res_col in df.columns else pl.lit(SpatialResolution.UNKNOWN.value)
        res_expr = (
            pl.when(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
            .then(pl.lit(SpatialResolution.EXACT.value))
            .when(pl.col("_clean_spatial").is_not_null() & pl.col("centroid_lat").is_not_null())
            .then(pl.col("spatial_resolution"))
            .otherwise(existing_res)
            .cast(pl.Categorical)
            .alias(res_col)
        )

        res_df = joined.with_columns(
            [
                lat_expr.alias("latitude"),
                lon_expr.alias("longitude"),
                res_expr,
            ]
        )

        drop_cols = [c for c in gaz.columns if c != "country" and c in res_df.columns]
        drop_cols.append("_clean_spatial")
        return self._wrap_result(res_df.drop(drop_cols))

    def reverse_geocode(
        self,
        geojson_path: str | Path | None = None,
        target_spatial_name: str | StrEnum = "Country",
    ) -> Any:  # noqa: ANN401
        """Performs reverse geocoding to determine spatial locality from coordinates."""
        df = self._df
        if geojson_path is None:
            geojson_path = Path(__file__).resolve().parent.parent / "data" / "world_boundaries.geojson"

        if not Path(geojson_path).exists():
            return self._wrap_result(df)

        with open(geojson_path, encoding="utf-8") as f:
            feature_collection = json.load(f)

        features = [f for f in feature_collection.get("features", []) if f.get("geometry")]
        if not features:
            return self._wrap_result(df)

        geom_strings = [json.dumps(f["geometry"]) for f in features]
        polygons = from_geojson(geom_strings)
        country_names = np.array([f["properties"].get("ADMIN", "Unknown") for f in features])

        if "latitude" not in df.columns or "longitude" not in df.columns:
            return self._wrap_result(df)

        valid_coords_df = (
            df.filter(
                pl.col("latitude").is_not_null()
                & pl.col("longitude").is_not_null()
                & ~pl.col("latitude").is_nan()
                & ~pl.col("longitude").is_nan()
            )
            .select(["latitude", "longitude"])
            .unique()
        )
        if len(valid_coords_df) == 0:
            return self._wrap_result(df)

        lons = valid_coords_df.get_column("longitude").to_numpy().astype(float)
        lats = valid_coords_df.get_column("latitude").to_numpy().astype(float)
        pts = points(lons, lats)

        tree = STRtree(polygons)
        pt_idx, poly_idx = tree.query(pts, predicate="intersects")

        unique_pt_idx, unique_indices = np.unique(pt_idx, return_index=True)
        country_results = np.full(len(valid_coords_df), None, dtype=object)
        country_results[unique_pt_idx] = country_names[poly_idx[unique_indices]]

        valid_coords_df = valid_coords_df.with_columns(
            pl.Series("country_name", country_results.tolist(), dtype=pl.Utf8)
        )

        target_str = str(target_spatial_name)
        spatial_prefix = f"{Domain.SPATIAL.value}_"
        if target_str.startswith(spatial_prefix):
            target_str = target_str[len(spatial_prefix) :]
        new_col = f"{spatial_prefix}{target_str}"
        res_col = f"{Domain.SPATIAL_RES.value}_{target_str}"

        res_df = df.join(valid_coords_df, on=["latitude", "longitude"], how="left")
        res_df = res_df.with_columns(
            [
                pl.coalesce(
                    [
                        pl.col("country_name"),
                        pl.col(new_col).cast(pl.Utf8) if new_col in df.columns else pl.lit(None, dtype=pl.Utf8),
                    ]
                ).alias(new_col),
                pl.when(pl.col("latitude").is_not_null() & pl.col("longitude").is_not_null())
                .then(pl.lit(SpatialResolution.EXACT.value))
                .otherwise(pl.col(res_col) if res_col in df.columns else pl.lit(SpatialResolution.UNKNOWN.value))
                .cast(pl.Categorical)
                .alias(res_col),
                pl.col("latitude").cast(pl.Float64),
                pl.col("longitude").cast(pl.Float64),
            ]
        )
        if "country_name" in res_df.columns:
            res_df = res_df.drop("country_name")
        return self._wrap_result(res_df)

    def with_metadata(
        self,
        indicator: str,
        country_col: str | StrEnum = "country",
        year_col: str | StrEnum | None = None,
    ) -> Any:  # noqa: ANN401
        """Fetches World Bank indicator metadata for countries in dataset and appends as Float64 column.

        Args:
            indicator: World Bank indicator code (e.g., 'SP.DYN.LE00.IN').
            country_col: Column name or StrEnum in dataset containing country names/codes.
            year_col: Optional column name or StrEnum in dataset containing observation years.

        Returns:
            Updated SeroEpiDataset (or MetaDataFrame) with indicator column attached.
        """
        from seroepi.client import WorldBankClient

        df = self._df
        resolved_country = self._resolve_col(country_col)
        if resolved_country not in df.columns:
            raise ValueError(f"Country column '{country_col}' not found in dataset.")

        resolved_year = None
        if year_col is not None:
            resolved_year = self._resolve_col(year_col)
            if resolved_year not in df.columns:
                raise ValueError(f"Year column '{year_col}' not found in dataset.")

        unique_countries_raw = df.get_column(resolved_country).drop_nulls().unique().to_list()
        raw_countries = [str(c) for c in unique_countries_raw if c is not None and str(c).strip()]

        if not raw_countries:
            df_base = df.drop(indicator) if indicator in df.columns else df
            res_df = df_base.with_columns(pl.lit(None).cast(pl.Float64).alias(indicator))
            return self._wrap_result(res_df)

        client = WorldBankClient()
        wb_payload = client.fetch_indicator(indicator, raw_countries)

        if resolved_year is not None:
            unique_combos = df.select([resolved_country, resolved_year]).unique()
            c_vals = unique_combos.get_column(resolved_country).to_list()
            y_vals = unique_combos.get_column(resolved_year).to_list()

            indicator_vals: list[float | None] = []
            for raw_c, raw_y in zip(c_vals, y_vals):
                if raw_c is None or not str(raw_c).strip():
                    indicator_vals.append(None)
                    continue

                iso3 = client.resolve_iso3(str(raw_c))
                country_data = wb_payload.get(iso3, {}) if iso3 else {}

                if not country_data:
                    indicator_vals.append(None)
                    continue

                parsed_yr = _parse_year(raw_y)
                sorted_years = sorted(country_data.keys())

                if parsed_yr is not None:
                    if parsed_yr in country_data:
                        indicator_vals.append(country_data[parsed_yr])
                    else:
                        past_years = [y for y in sorted_years if y <= parsed_yr]
                        fallback_year = max(past_years) if past_years else max(sorted_years)
                        indicator_vals.append(country_data[fallback_year])
                else:
                    fallback_year = max(sorted_years)
                    indicator_vals.append(country_data[fallback_year])

            lookup_df = pl.DataFrame(
                {
                    resolved_country: c_vals,
                    resolved_year: y_vals,
                    indicator: indicator_vals,
                },
                schema={
                    resolved_country: df.schema[resolved_country],
                    resolved_year: df.schema[resolved_year],
                    indicator: pl.Float64,
                },
            )
            join_cols = [resolved_country, resolved_year]

        else:
            unique_combos = df.select([resolved_country]).unique()
            c_vals = unique_combos.get_column(resolved_country).to_list()

            indicator_vals = []
            for raw_c in c_vals:
                if raw_c is None or not str(raw_c).strip():
                    indicator_vals.append(None)
                    continue

                iso3 = client.resolve_iso3(str(raw_c))
                country_data = wb_payload.get(iso3, {}) if iso3 else {}

                if not country_data:
                    indicator_vals.append(None)
                    continue

                sorted_years = sorted(country_data.keys())
                latest_year = max(sorted_years)
                indicator_vals.append(country_data[latest_year])

            lookup_df = pl.DataFrame(
                {
                    resolved_country: c_vals,
                    indicator: indicator_vals,
                },
                schema={
                    resolved_country: df.schema[resolved_country],
                    indicator: pl.Float64,
                },
            )
            join_cols = [resolved_country]

        df_base = df.drop(indicator) if indicator in df.columns else df
        try:
            res_df = df_base.join(lookup_df, on=join_cols, how="left", nulls_equal=True)
        except (TypeError, ValueError):
            try:
                res_df = df_base.join(lookup_df, on=join_cols, how="left", join_nulls=True)
            except (TypeError, ValueError):
                res_df = df_base.join(lookup_df, on=join_cols, how="left")
        return self._wrap_result(res_df)


@pl.api.register_dataframe_namespace("geo")
class GeoAccessor(GeoMixin):
    """Polars DataFrame namespace for geographical operations."""

    def __init__(self, polars_obj: pl.DataFrame):  # noqa: ANN204, D107
        self._df_obj = polars_obj


class ChoroplethPlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(  # type: ignore # noqa: D102
        cls,
        result: estimators.PrevalenceEstimates,
        geo_col: str | None = None,
        target_variant: str | None = None,
        feature_id_key: str = "properties.ADMIN",
        **kwargs,  # noqa: ANN003
    ) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        data = result.data

        if geo_col is None:
            spatial_candidates = [c for c in result.stratified_by if c.startswith(f"{Domain.SPATIAL.value}_")]
            if spatial_candidates:
                geo_col = spatial_candidates[0]
            elif len(result.stratified_by) == 1:
                geo_col = result.stratified_by[0]
            else:
                raise ValueError("Cannot plot choropleth: Please stratify by a spatial column.")
        elif geo_col not in result.stratified_by:
            raise ValueError(f"Cannot plot choropleth: '{geo_col}' was not used as a stratification variable.")

        target_name = result.trait
        if result.aggregation_type == AggregationType.COMPOSITIONAL:
            if target_variant is None:
                top_row = data.group_by("target").agg(pl.col("event").sum()).sort("event", descending=True)
                target_variant = top_row["target"][0]

            data = data.filter(pl.col("target") == target_variant)
            target_name = f"{cls._clean_label(result.trait)}: {target_variant}"

        geojson_data = cls._get_world_geojson()

        estimates = data["estimate"].to_numpy()
        lowers = data["lower"].to_numpy()
        uppers = data["upper"].to_numpy()
        geo_vals = data[geo_col].to_list()

        hover_text = [
            f"<b>{geo_col}</b>: {g}<br><b>Prevalence</b>: {e:.2%}<br><b>95% CI</b>: [{l:.2%}, {u:.2%}]"
            for g, e, l, u in zip(geo_vals, estimates, lowers, uppers)  # noqa: E741
        ]

        fig = go.Figure(
            go.Choroplethmapbox(
                geojson=geojson_data,
                locations=geo_vals,
                featureidkey=feature_id_key,
                z=estimates,
                colorscale=cls.get_colorscale(transparent=False),
                zmin=0,
                zmax=1,
                marker_opacity=0.8,
                marker_line_width=1.5,
                marker_line_color="white",
                colorbar_title="Prevalence",
                text=hover_text,
                hoverinfo="text",
            )
        )

        title_prefix = (
            "Regional Composition of"
            if result.aggregation_type == AggregationType.COMPOSITIONAL
            else "Regional Prevalence of"
        )

        return cls.apply_theme(
            fig.update_layout(
                title=f"<b>{title_prefix} {target_name}</b>",
                mapbox_style="carto-positron",
                mapbox_zoom=1,
                mapbox_center={"lat": 0, "lon": 0},
                margin={"r": 0, "t": 60, "l": 0, "b": 0},
            )
        )


class SpatialSurfacePlotter(BasePlotter):  # noqa: D101
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(  # type: ignore # noqa: D102
        cls,
        result: estimators.PrevalenceEstimates,
        lat_col: str = "lat",
        lon_col: str = "lon",
        **kwargs,  # noqa: ANN003
    ) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        data = result.data

        if lat_col not in data.columns or lon_col not in data.columns:
            raise ValueError(f"Spatial surface requires '{lat_col}' and '{lon_col}' columns in the data.")

        lats = data[lat_col].to_numpy()
        lons = data[lon_col].to_numpy()
        estimates = data["estimate"].to_numpy()

        hover_text = [
            f"<b>Lat</b>: {la:.3f} | <b>Lon</b>: {lo:.3f}<br><b>Predicted Prevalence</b>: {e:.2%}"
            for la, lo, e in zip(lats, lons, estimates)
        ]

        fig = go.Figure(
            go.Densitymapbox(
                lat=lats,
                lon=lons,
                z=estimates,
                radius=25,
                colorscale=cls.get_colorscale(transparent=True),
                zmin=0,
                zmax=1,
                opacity=0.85,
                text=hover_text,
                hoverinfo="text",
                colorbar_title="Predicted<br>Prevalence",
            )
        )

        title_prefix = (
            "Spatial Composition of"
            if result.aggregation_type == AggregationType.COMPOSITIONAL
            else "Spatial Surface for"
        )

        return cls.apply_theme(
            fig.update_layout(
                title=f"<b>{title_prefix} {cls._clean_label(result.trait)}</b><br><sup>Gaussian Process Prediction Surface</sup>",  # noqa: E501
                mapbox_style="carto-positron",
                mapbox_zoom=3,
                mapbox_center={"lat": float(lats.mean()), "lon": float(lons.mean())},
                margin={"r": 0, "t": 60, "l": 0, "b": 0},
            )
        )


register_plotter(PlotType.CHOROPLETH, ChoroplethPlotter)
register_plotter(PlotType.SPATIAL_SURFACE, SpatialSurfacePlotter)
