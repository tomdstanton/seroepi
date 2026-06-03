"""
Module to handle epidemiological, geospatial and genotypic operations on isolate datasets in the form of Pandas
DataFrames.
"""
from pathlib import Path
from typing import Union
import pandas as pd
import numpy as np
import json
from shapely import from_geojson, points
from shapely.strtree import STRtree
from seroepi.data.gazetteer_data import GAZETTEER_DICT
from seroepi.constants import MetricType, TemporalResolution, SpatialResolution, AggregationType, Domain
from seroepi.dist import TransmissionDistances


# Accessors ------------------------------------------------------------------------------------------------------------
@pd.api.extensions.register_dataframe_accessor("geo")
class GeoAccessor:
    """
    Pandas accessor for geographical operations on isolate datasets.

    Provides methods for standardizing location names, imputing missing
    coordinates using a gazetteer, and performing reverse geocoding.

    Attributes:
        gazetteer (pd.DataFrame): A DataFrame containing centroid coordinates and
            metadata for countries.
    """
    
    # Class-level cache to prevent expensive dataframe recreation on every accessor call
    _gazetteer_df = None
    
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    @property
    def gazetteer(self) -> pd.DataFrame:
        """Returns the internal gazetteer used for coordinate imputation."""
        if GeoAccessor._gazetteer_df is None:
            GeoAccessor._gazetteer_df = pd.DataFrame.from_dict(GAZETTEER_DICT, orient='index')
        return GeoAccessor._gazetteer_df

    @property
    def spatial(self) -> pd.DataFrame:
        """Returns a DataFrame of all spatial columns, with the prefix removed."""
        return self._obj.filter(regex=f"^{Domain.SPATIAL.value}_(?!res_)").rename(columns=lambda c: c.replace(f'{Domain.SPATIAL.value}_', '', 1))

    @property
    def spatial_resolution(self) -> pd.DataFrame:
        """Returns a DataFrame of all spatial resolution columns."""
        return self._obj.filter(regex=f"^{Domain.SPATIAL_RES.value}_").rename(columns=lambda c: c.replace(f'{Domain.SPATIAL_RES.value}_', '', 1))

    def standardize_and_impute(self, spatial_col: str = None) -> pd.DataFrame:
        """
        Standardizes spatial names and imputes missing coordinates.

        Uses the internal gazetteer to find centroids for countries when exact
        latitude and longitude are missing.
        
        Args:
            spatial_col: Optional specific spatial column to impute by. Defaults to the first mapped spatial column.

        Returns:
            A new DataFrame with imputed coordinates and spatial resolution metadata.
        """
        df = self._obj.copy()
        ref_data = self.gazetteer
        
        spatial_cols = df.filter(regex=f"^{Domain.SPATIAL.value}_(?!res_)").columns.tolist()
        if not spatial_cols:
            return df
            
        if spatial_col is None:
            spatial_col = spatial_cols[0]
        elif spatial_col not in df.columns and f"{Domain.SPATIAL.value}_{spatial_col}" in df.columns:
            spatial_col = f"{Domain.SPATIAL.value}_{spatial_col}"
            
        res_col = spatial_col.replace(f"{Domain.SPATIAL.value}_", f"{Domain.SPATIAL_RES.value}_")

        # Initialize tracking
        if res_col not in df.columns:
            df[res_col] = SpatialResolution.UNKNOWN.value

        # If it's a category, we need to ensure all possible enum values are in the categories before assigning
        if isinstance(df[res_col].dtype, pd.CategoricalDtype):
            missing_cats = [c for c in SpatialResolution.choices() if c not in df[res_col].cat.categories]
            if missing_cats:
                df[res_col] = df[res_col].cat.add_categories(missing_cats)

        exact_mask = df['latitude'].notna() & df['longitude'].notna()
        df.loc[exact_mask, res_col] = SpatialResolution.EXACT.value

        # Impute using the instant dictionary-backed dataframe
        clean_spatial = df[spatial_col].str.strip()
        needs_imputation = (~exact_mask) & clean_spatial.notna()

        # OPTIMIZATION: Only map the rows that actually need imputation
        impute_spatial = clean_spatial[needs_imputation]
        df.loc[needs_imputation, 'latitude'] = impute_spatial.map(ref_data['centroid_lat'])
        df.loc[needs_imputation, 'longitude'] = impute_spatial.map(ref_data['centroid_lon'])

        imputed_mask = needs_imputation & df['latitude'].notna()
        df.loc[imputed_mask, res_col] = clean_spatial[imputed_mask].map(ref_data['spatial_resolution'])

        # Remove unused categories if it was a CategoricalDtype
        if isinstance(df[res_col].dtype, pd.CategoricalDtype):
             df[res_col] = df[res_col].cat.remove_unused_categories()

        df['latitude'] = df['latitude'].astype("Float64")
        df['longitude'] = df['longitude'].astype("Float64")

        return df

    def reverse_geocode(self, geojson_path: Union[str, Path] = None, target_spatial_name: str = 'Country') -> pd.DataFrame:
        """
        Performs reverse geocoding to determine spatial locality from coordinates.

        Args:
            geojson_path: Optional path to a GeoJSON file containing boundary polygons.
                Defaults to the built-in world_boundaries.geojson.
            target_spatial_name: The name to append to the spatial domain prefix.

        Returns:
            A new DataFrame with updated 'spatial' information.
        """
        if geojson_path is None:
            geojson_path = Path(__file__).parent / "data" / "world_boundaries.geojson"
            
        if not Path(geojson_path).exists():
            return self._obj.copy()

        with open(geojson_path, 'r', encoding='utf-8') as f:
            feature_collection = json.load(f)

        features = [f for f in feature_collection.get('features', []) if f.get('geometry')]
        
        # 1. Fast C-level GeoJSON geometry parsing
        geom_strings = [json.dumps(f['geometry']) for f in features]
        polygons = from_geojson(geom_strings)
        country_names = np.array([f['properties'].get('ADMIN', 'Unknown') for f in features])

        df = self._obj.copy()
        exact_mask = df['latitude'].notna() & df['longitude'].notna()

        if not exact_mask.any():
            return df

        # OPTIMIZATION: Reverse geocode unique coordinates using an STRtree Spatial Index
        unique_coords = df.loc[exact_mask, ['latitude', 'longitude']].drop_duplicates()
        
        # 2. Vectorized Point creation (astype(float) prevents Pandas Float64 extension errors)
        pts = points(unique_coords['longitude'].astype(float).values, unique_coords['latitude'].astype(float).values)

        # 3. R-Tree spatial index for lightning-fast Point-in-Polygon queries
        tree = STRtree(polygons)
        pt_idx, poly_idx = tree.query(pts, predicate='intersects')

        # 4. Resolve border overlaps by keeping the first matched polygon per point
        unique_pt_idx, unique_indices = np.unique(pt_idx, return_index=True)
        
        # Map back the names
        country_results = np.full(len(unique_coords), pd.NA, dtype=object)
        country_results[unique_pt_idx] = country_names[poly_idx[unique_indices]]
        unique_coords['country_name'] = country_results

        # Merge back into the main DataFrame
        df = df.merge(unique_coords, on=['latitude', 'longitude'], how='left')
        
        new_col = f"{Domain.SPATIAL.value}_{target_spatial_name}"
        res_col = f"{Domain.SPATIAL_RES.value}_{target_spatial_name}"
        
        df['latitude'] = df['latitude'].astype("Float64")
        df['longitude'] = df['longitude'].astype("Float64")
        
        return df


@pd.api.extensions.register_dataframe_accessor("epi")
class EpiAccessor:
    """
    Pandas accessor for epidemiological analysis on isolate datasets.

    Provides methods for generating epidemic curves, calculating prevalence,
    diversity, and incidence, and identifying transmission clusters.
    """
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    # --- Shiny UI State Checkers ---

    @property
    def has_temporal(self) -> bool:
        """Checks if the dataset contains valid temporal data (any 'temporal_' column)."""
        cols = self._obj.filter(regex=f"^{Domain.TEMPORAL.value}_(?!res_)").columns
        return bool(len(cols) > 0 and self._obj[cols[0]].notna().any())

    @property
    def has_spatial(self) -> bool:
        """Checks if the dataset contains valid spatial coordinates (latitude and longitude)."""
        return 'latitude' in self._obj.columns and 'longitude' in self._obj.columns

    # --- Spatiotemporal Helpers ---

    @property
    def temporal(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all temporal columns, with the prefix removed.
        """
        return self._obj.filter(regex=f"^{Domain.TEMPORAL.value}_(?!res_)").rename(columns=lambda c: c.replace(f'{Domain.TEMPORAL.value}_', '', 1))
        
    @property
    def temporal_resolution(self) -> pd.DataFrame:
        """
        Returns a DataFrame of all temporal resolution columns.
        """
        return self._obj.filter(regex=f"^{Domain.TEMPORAL_RES.value}_").rename(columns=lambda c: c.replace(f'{Domain.TEMPORAL_RES.value}_', '', 1))

    @property
    def spatial(self) -> pd.DataFrame:
        """
        Returns the core spatial coordinates (latitude, longitude).

        Raises:
            ValueError: If spatial columns are missing.
        """
        if not self.has_spatial:
            raise ValueError("Spatial columns ('latitude', 'longitude') are missing.")
        return self._obj[['latitude', 'longitude']].astype("Float64")

    # --- Internal Kernel Helpers ---

    def _resolve_temporal_col(self, col: str = None) -> str:
        """Resolves and validates the temporal datetime column."""
        df = self._obj
        if col is None:
            cols = df.filter(regex=f"^{Domain.TEMPORAL.value}_(?!res_)").columns
            if not len(cols):
                raise KeyError("A valid temporal datetime column is required.")
            col = cols[0]
        elif not col.startswith(f"{Domain.TEMPORAL.value}_"):
            col = f"{Domain.TEMPORAL.value}_{col}"

        if col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[col]):
            raise TypeError(f"Temporal column '{col}' must be a datetime64 type. Ensure data is parsed via seroepi.io.")
        return col

    @staticmethod
    def _resolve_freq(freq: Union[str, TemporalResolution]) -> tuple[str, str, str]:
        """Resolves frequency inputs into (pandas_offset, pandas_period, raw_value)."""
        if isinstance(freq, str):
            try: freq = TemporalResolution(freq)
            except ValueError: pass
            
        if isinstance(freq, TemporalResolution):
            return freq.pandas_offset, freq.pandas_period, freq.value
            
        # Fallback for raw pandas strings
        return freq, freq.replace('ME', 'M').replace('YE', 'Y'), str(freq)

    def _attach_metadata(self, df: pd.DataFrame, metric_type: MetricType, stratify_by: list[str], 
                         trait_col: str = None, cluster_col: str = None, pad_zeros: bool = False, extra: dict = None) -> pd.DataFrame:
        """Standardizes metric metadata attachment."""
        df.attrs = self._obj.attrs.copy()
        meta = {
            "metric_type": metric_type,
            "stratified_by": stratify_by if trait_col else stratify_by[:-1],
            "trait": trait_col if trait_col else stratify_by[-1],
            "aggregation_type": AggregationType.TRAIT if trait_col else AggregationType.COMPOSITIONAL,
            "adjusted_for": cluster_col,
            "is_zero_padded": pad_zeros
        }
        if extra: meta.update(extra)
        df.attrs['metric_meta'] = meta
        return df

    # --- Time Series / Epidemic Curve Methods ---

    def _get_spatiotemporal_arrays(self, temporal_col: str = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Helper to extract and format coordinates and dates for spatial clustering."""
        temporal_col = self._resolve_temporal_col(temporal_col)
        df = self._obj
        
        # Schema guarantees datetime64, so we just safely strip timezones if present
        date_series = df[temporal_col].dt.tz_localize(None)
        # Safely pull coordinates using the internal spatial property
        coords = np.radians(self.spatial.astype(float).values)
        
        # Convert dates to raw days (use float to allow NaNs for missing dates)
        raw_dates = np.full(len(df), np.nan)
        date_mask = date_series.notna().values
        raw_dates[date_mask] = date_series[date_mask].values.astype('datetime64[D]').astype(float)
        
        # Create a boolean mask of rows that have all required spatiotemporal data
        valid_mask = ~(np.isnan(coords[:, 0]) | np.isnan(coords[:, 1]) | np.isnan(raw_dates))
        return coords, raw_dates, valid_mask

    def epidemic_curve(self, freq: Union[str, TemporalResolution] = TemporalResolution.WEEK,
                       stratify_by: str = None, temporal_col: str = None) -> pd.DataFrame:
        """
        Generates a time-series DataFrame for plotting epidemic curves.

        Args:
            freq: Time frequency for resampling (e.g., TimeResolution.MONTH, 'ME', 'YE').
                Defaults to TimeResolution.WEEK.
            stratify_by: Column name to group by before resampling.

        Returns:
            A DataFrame with counts of isolates per time interval.

        Raises:
            ValueError: If temporal data is not available.
        """
        if not self.has_temporal:
            raise ValueError("Cannot generate epi curve: No temporal data available.")

        df = self._obj.copy()
        
        freq_val, _, _ = self._resolve_freq(freq)
        temporal_col = self._resolve_temporal_col(temporal_col)

        # Set time as index for resampling
        df = df.set_index(temporal_col)

        if stratify_by:
            curve = df.groupby(stratify_by, observed=True).resample(freq_val).size().unstack(level=0, fill_value=0)
        else:
            curve = df.resample(freq_val).size().to_frame(name='count')

        return curve.reset_index()

    @property
    def metadata_columns(self) -> list[str]:
        """Returns the raw names of user-uploaded clinical/metadata columns."""
        return self._obj.filter(regex="^meta_").columns.tolist()

    @property
    def ui_metadata_columns(self) -> list[str]:
        """Returns clean metadata names (without 'meta_' prefix) for UI display."""
        return [c.replace('meta_', '', 1) for c in self.metadata_columns]

    @property
    def genotypes(self) -> list[str]:
        """Compiles a list of all genetic variables (Core + Accessory traits)."""
        # Grab all the dynamically prefixed traits (genotype, phenotype, amr, virulence)
        return self._obj.filter(regex=f"^({Domain.GENOTYPE.value}|{Domain.PHENOTYPE.value}|{Domain.AMR.value}|{Domain.VIRULENCE.value})_").columns.tolist()

    @property
    def stratify_cols(self) -> list[str]:
        """Returns columns suitable for stratification (excluding QC, metadata, and high-cardinality/internal cols)."""
        exclude_strat = ['sample_id', 'latitude', 'longitude']
        return [c for c in self._obj.columns if not c.startswith((f'{Domain.QC.value}_', 'meta_', f'{Domain.SPATIAL_RES.value}_', f'{Domain.TEMPORAL_RES.value}_')) and c not in exclude_strat]

    @property
    def cluster_cols(self) -> list[str]:
        """Returns columns suitable for cluster adjustment (e.g., transmission clusters and ST)."""
        return [c for c in self._obj.columns if c.startswith(f"{Domain.CLUSTER.value}_") or c.endswith('_ST')]

    @staticmethod
    def _calculate_events_and_denoms(
        df: pd.DataFrame,
        trait_col: str,
        denom_cols: list[str],
        trait_strata: list[str],
        cluster_col: str,
        negative_indicator: Union[str, list[str]],
        event_name: str,
        denom_name: str
    ) -> tuple[Union[pd.Series, int], Union[pd.Series, int]]:
        """Helper to calculate numerators and denominators for aggregated metrics."""
        if trait_col:
            valid_df = df.dropna(subset=[trait_col]).copy()

            if pd.api.types.is_bool_dtype(valid_df[trait_col]):
                valid_df['_trait_bool'] = valid_df[trait_col]
            else:
                neg_list = [negative_indicator] if isinstance(negative_indicator, str) else negative_indicator
                valid_df['_trait_bool'] = ~valid_df[trait_col].isin(neg_list)

            if cluster_col:
                denoms = valid_df.groupby(denom_cols, observed=True)[cluster_col].nunique().rename(denom_name) if denom_cols else valid_df[cluster_col].nunique()
                events = valid_df[valid_df['_trait_bool']].groupby(trait_strata, observed=True)[cluster_col].nunique().rename(event_name) if trait_strata else valid_df[valid_df['_trait_bool']][cluster_col].nunique()
            else:
                denoms = valid_df.groupby(denom_cols, observed=True).size().rename(denom_name) if denom_cols else len(valid_df)
                events = valid_df.groupby(trait_strata, observed=True)['_trait_bool'].sum().rename(event_name) if trait_strata else valid_df['_trait_bool'].sum()

        else:
            valid_df = df.dropna(subset=[trait_strata[-1]]).copy()

            if cluster_col:
                denoms = valid_df.groupby(denom_cols, observed=True)[cluster_col].nunique().rename(denom_name) if denom_cols else valid_df[cluster_col].nunique()
                events = valid_df.groupby(trait_strata, observed=True)[cluster_col].nunique().rename(event_name)
            else:
                denoms = valid_df.groupby(denom_cols, observed=True).size().rename(denom_name) if denom_cols else len(valid_df)
                events = valid_df.groupby(trait_strata, observed=True).size().rename(event_name)

        return events, denoms

    def aggregate_prevalence(self, stratify_by: list[str], trait_col: str = None,
                             cluster_col: str = None, negative_indicator: Union[str, list[str]] = '-',
                             pad_zeros: bool = False) -> pd.DataFrame:
        """
        Aggregates data to calculate event counts and denominators for prevalence.

        Supports both trait prevalence (presence/absence of a marker) and
        compositional prevalence (distribution of variants within a locus).

        Args:
            stratify_by: Columns to group by (e.g., ['spatial']).
            trait_col: The column containing the trait/marker to measure.
                If None, compositional prevalence is calculated for the last
                column in `stratify_by`.
            cluster_col: Column containing cluster IDs to adjust for (e.g., nosocomial outbreaks).
            negative_indicator: Value(s) representing the absence of a trait.
                Defaults to '-'.
            pad_zeros: If True, pads missing combinations of strata with zero counts.
                Essential for spatial/hierarchical models. If False, only includes
                observed combinations for efficiency. Defaults to False.

        Returns:
            An aggregated DataFrame with 'event' and 'n' columns.

        Raises:
            ValueError: If compositional prevalence is requested without enough strata.
        """
        df = self._obj.copy()

        if trait_col:
            denom_cols = stratify_by
            trait_strata = stratify_by
        else:
            if len(stratify_by) < 1:
                raise ValueError("Compositional prevalence requires at least 1 stratify_by column.")
            denom_cols = stratify_by[:-1]
            trait_strata = stratify_by

        events, denoms = self._calculate_events_and_denoms(
            df=df,
            trait_col=trait_col,
            denom_cols=denom_cols,
            trait_strata=trait_strata,
            cluster_col=cluster_col,
            negative_indicator=negative_indicator,
            event_name='event',
            denom_name='n'
        )

        # Expand Grid
        if len(stratify_by) > 0:
            if pad_zeros:
                # Full Cartesian expansion (padding zeroes)
                unique_levels = [df[col].dropna().unique() for col in stratify_by]
                base_index = pd.MultiIndex.from_product(unique_levels, names=stratify_by)
            else:
                # Efficiently use only the observed strata combinations
                base_index = denoms.index if trait_col else events.index

            agg_df = pd.DataFrame(index=base_index).join(events, how='left').fillna({'event': 0}).reset_index()

            # Safely map denominators
            if len(denom_cols) == 0:
                agg_df['n'] = denoms
            else:
                agg_df = agg_df.set_index(denom_cols)
                agg_df['n'] = denoms
                agg_df = agg_df.reset_index().fillna({'n': 0})
                
            if not pad_zeros:
                agg_df = agg_df[agg_df['n'] > 0].copy()
        else:
            agg_df = pd.DataFrame({'event': [events], 'n': [denoms]})

        # Standardize to 'target' for uniform downstream handling
        if trait_col:
            agg_df['target'] = trait_col
        else:
            agg_df = agg_df.rename(columns={stratify_by[-1]: 'target'})

        return self._attach_metadata(agg_df, MetricType.PREVALENCE, stratify_by, trait_col, cluster_col, pad_zeros)

    def aggregate_diversity(self, stratify_by: list[str], trait_col: str = None,
                            cluster_col: str = None, negative_indicator: Union[str, list[str]] = '-',
                            pad_zeros: bool = False) -> pd.DataFrame:
        """
        Aggregates data to calculate counts for diversity analysis (e.g., Shannon index).

        Args:
            stratify_by: Columns to group by.
            trait_col: The locus or trait to measure diversity for.
            cluster_col: Optional cluster column to adjust for.
            negative_indicator: Value(s) to exclude from diversity counts.
            pad_zeros: If True, pads missing combinations of strata with zero counts.
                Defaults to False.

        Returns:
            A DataFrame with 'variant_count' and 'n_total'.

        Raises:
            ValueError: If compositional diversity is requested without enough strata.
        """
        df = self._obj.copy()

        is_trait = True if trait_col else False
        if trait_col:
            groupers = stratify_by
            trait_strata = stratify_by + [trait_col]
            valid_df = df.dropna(subset=[trait_col]).copy()

            # For trait diversity, we often want to strip out the "absence" indicators
            # so they don't count as a diversity variant mathematically
            if not pd.api.types.is_bool_dtype(valid_df[trait_col]):
                neg_list = [negative_indicator] if isinstance(negative_indicator, str) else negative_indicator
                valid_df = valid_df[~valid_df[trait_col].isin(neg_list)]
        else:
            if not stratify_by:
                raise ValueError("Compositional diversity requires at least 1 stratify_by column.")
            groupers = stratify_by[:-1]
            trait_col = stratify_by[-1]
            trait_strata = stratify_by
            valid_df = df.dropna(subset=[trait_col]).copy()

        if cluster_col:
            div_df = valid_df.groupby(trait_strata, observed=True)[cluster_col].nunique().reset_index()
            div_df = div_df.rename(columns={cluster_col: 'variant_count'})
        else:
            div_df = valid_df.groupby(trait_strata, observed=True).size().reset_index(name='variant_count')

        if pad_zeros and trait_strata:
            unique_levels = [df[col].dropna().unique() for col in trait_strata]
            expanded_index = pd.MultiIndex.from_product(unique_levels, names=trait_strata)
            div_df = div_df.set_index(trait_strata).reindex(expanded_index, fill_value=0).reset_index()

        if groupers:
            div_df['n_total'] = div_df.groupby(groupers, observed=True)['variant_count'].transform('sum')
        else:
            div_df['n_total'] = div_df['variant_count'].sum()

        # Standardize to 'target'
        div_df = div_df.rename(columns={trait_col: 'target'})
        if is_trait:
            div_df['target'] = trait_col

        return self._attach_metadata(div_df, MetricType.DIVERSITY, stratify_by, trait_col, cluster_col, pad_zeros)

    def aggregate_incidence(self, stratify_by: list[str], trait_col: str = None, freq: Union[str, TemporalResolution] = TemporalResolution.MONTH,
                            cluster_col: str = None, negative_indicator: Union[str, list[str]] = '-',
                            pad_zeros: bool = False, temporal_col: str = None) -> pd.DataFrame:
        """
        Aggregates data for time-series incidence analysis.

        Args:
            stratify_by: Columns to group by.
            trait_col: The marker to measure incidence for.
            freq: Time frequency for binning (e.g., TimeResolution.MONTH, 'ME'). Defaults to TimeResolution.MONTH.
            cluster_col: Optional cluster column to adjust for.
            negative_indicator: Value(s) representing absence.
            pad_zeros: If True, pads missing combinations of strata. If False,
                maintains unbroken time grids only for observed strata combinations.

        Returns:
            A DataFrame with 'variant_count', 'total_sequenced', and binned dates.

        Raises:
            ValueError: If temporal data is missing or inappropriate strata provided.
        """
        df = self._obj.copy()

        temporal_col = self._resolve_temporal_col(temporal_col)
        _, period_freq, stored_freq = self._resolve_freq(freq)

        # Snap dates to the requested frequency bin using the safe string
        df['date_bin'] = df[temporal_col].dt.to_period(period_freq).dt.to_timestamp()

        if trait_col:
            denom_cols = ['date_bin'] + stratify_by
            trait_strata = ['date_bin'] + stratify_by
        else:
            if not stratify_by:
                raise ValueError("Compositional incidence requires at least 1 stratify_by column.")

            denom_cols = ['date_bin'] + stratify_by[:-1]
            trait_strata = ['date_bin'] + stratify_by

        events, denoms = self._calculate_events_and_denoms(
            df=df,
            trait_col=trait_col,
            denom_cols=denom_cols,
            trait_strata=trait_strata,
            cluster_col=cluster_col,
            negative_indicator=negative_indicator,
            event_name='variant_count',
            denom_name='total_sequenced'
        )

        # --- TIME GRID EXPANSION ---
        # Generate an unbroken sequence of dates from the earliest to the latest record
        min_date, max_date = df['date_bin'].min(), df['date_bin'].max()
        all_dates = pd.period_range(min_date, max_date, freq=period_freq).to_timestamp().tolist() if not pd.isna(
            min_date) else []

        if pad_zeros:
            unique_levels = [all_dates] + [df[col].dropna().unique() for col in trait_strata[1:]]
            expanded_index = pd.MultiIndex.from_product(unique_levels, names=trait_strata)
        else:
            if len(trait_strata) > 1:
                # Only use strata combinations that actually appear in the data
                observed_strata = df[trait_strata[1:]].dropna().drop_duplicates()
                dates_df = pd.DataFrame({trait_strata[0]: all_dates})
                expanded_df = dates_df.merge(observed_strata, how='cross')
                expanded_index = pd.MultiIndex.from_frame(expanded_df[trait_strata])
            else:
                expanded_index = pd.Index(all_dates, name=trait_strata[0])

        inc_df = pd.DataFrame(index=expanded_index).join(events, how='left').fillna({'variant_count': 0}).reset_index()

        inc_df = inc_df.set_index(denom_cols)
        inc_df['total_sequenced'] = denoms
        inc_df = inc_df.reset_index().fillna({'total_sequenced': 0})

        # Unlike Prevalence, we do NOT drop rows where total_sequenced == 0.
        # A true 0 sequence volume is critical information for an epicurve gap.
        inc_df = inc_df.rename(columns={'date_bin': 'date'})

        # Standardize to 'target'
        if trait_col:
            inc_df['target'] = trait_col
        else:
            inc_df = inc_df.rename(columns={stratify_by[-1]: 'target'})

        return self._attach_metadata(inc_df, MetricType.INCIDENCE, stratify_by, trait_col, cluster_col, pad_zeros, {"freq": stored_freq})

    def transmission_network(
            self,
            clone_col: str,
            spatial_threshold_km: float = 10.0,
            temporal_threshold_days: int = 20,
            temporal_col: str = None
    ) -> TransmissionDistances:
        """
        Builds a sparse adjacency graph of transmission links.

        Args:
            clone_col: Column containing clone IDs (e.g., 'ST' or a custom cluster).
            spatial_threshold_km: Maximum distance in kilometers. Defaults to 10.0.
            temporal_threshold_days: Maximum time difference in days. Defaults to 20.

        Returns:
            A TransmissionDistances object representing the outbreak network.

        Raises:
            KeyError: If required columns ('latitude', 'longitude', 'date') are missing.
        """
        df = self._obj

        # 1. Validation Checks
        if clone_col not in df.columns:
            raise KeyError(f"Clone column '{clone_col}' not found in DataFrame.")

        # Intelligently check for your geo accessor/columns
        if not self.has_spatial:
            raise KeyError("Spatial clustering requires 'latitude' and 'longitude' columns. "
                           "Ensure geo accessors have parsed coordinates.")

        temporal_col = self._resolve_temporal_col(temporal_col)
        coords, raw_dates, _ = self._get_spatiotemporal_arrays(temporal_col)

        return TransmissionDistances.from_spatiotemporal(
            sample_ids=df['sample_id'],
            coords=coords,
            dates=raw_dates,
            clones=df[clone_col].values,
            spatial_threshold_km=spatial_threshold_km,
            temporal_threshold_days=temporal_threshold_days
        )

    def transmission_clusters(
            self,
            clone_col: str,
            spatial_threshold_km: float = 10.0,
            temporal_threshold_days: int = 20,
            temporal_col: str = None,
            network: TransmissionDistances = None
    ) -> pd.Series:
        """Extracts and formats categorical cluster labels from the transmission network."""
        df = self._obj
        temporal_col = self._resolve_temporal_col(temporal_col)

        if network is None:
            network = self.transmission_network(clone_col, spatial_threshold_km, temporal_threshold_days, temporal_col)

        labels = network.get_clusters()

        _, _, valid_mask = self._get_spatiotemporal_arrays(temporal_col)
        clone_mask = df[clone_col].notna().values

        labels_array = labels.astype(float).to_numpy(copy=True)
        labels_array[~valid_mask] = np.nan
        labels_array[~clone_mask] = np.nan

        res = pd.Series(labels_array, index=df.index, dtype="Int64",
                        name=f'{Domain.CLUSTER.value}_transmission_{spatial_threshold_km}km_{temporal_threshold_days}days')
        return res.astype("category").cat.as_ordered()


@pd.api.extensions.register_dataframe_accessor("geno")
class GenoAccessor:
    """
    Pandas accessor for genetic and trait-based operations.

    Provides methods for filtering determinants, checking for trait patterns,
    and sorting loci.

    Examples:
        >>> import pandas as pd
        >>> import seroepi.accessors
        >>> df = pd.DataFrame({'amr_blaKPC': [True, False], 'vir_ybt': [True, True]})
        >>> amr_matrix = df.geno.amr
    """
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    @property
    def genotype(self) -> pd.DataFrame:
        """Returns the Core Genotype matrix with the prefix removed from names."""
        return self._obj.filter(regex=f"^{Domain.GENOTYPE.value}_").rename(columns=lambda c: c.replace(f'{Domain.GENOTYPE.value}_', '', 1))

    @property
    def phenotype(self) -> pd.DataFrame:
        """Returns the Phenotype matrix with the prefix removed from names."""
        return self._obj.filter(regex=f"^{Domain.PHENOTYPE.value}_").rename(columns=lambda c: c.replace(f'{Domain.PHENOTYPE.value}_', '', 1))

    @property
    def amr(self) -> pd.DataFrame:
        """Returns the AMR determinant matrix with the prefix removed from names."""
        return self._obj.filter(regex=f"^{Domain.AMR.value}_").rename(columns=lambda c: c.replace(f'{Domain.AMR.value}_', '', 1))

    @property
    def virulence(self) -> pd.DataFrame:
        """Returns the Virulence marker matrix with the prefix removed from names."""
        return self._obj.filter(regex=f"^{Domain.VIRULENCE.value}_").rename(columns=lambda c: c.replace(f'{Domain.VIRULENCE.value}_', '', 1))

    def has_any(self, traits: list[str], domain: Union[str, Domain] = Domain.AMR) -> pd.Series:
        """
        Checks if isolates possess ANY of the specified traits.

        Args:
            traits: List of trait names (without prefix).
            domain: The domain prefix (e.g., Domain.AMR, Domain.VIRULENCE). Defaults to Domain.AMR.

        Returns:
            A boolean Series indicating presence of any specified trait.
        """
        domain_val = domain.value if isinstance(domain, Domain) else domain
        # Safely prepend the prefix and check if the columns actually exist
        trait_cols = [f"{domain_val}_{t}" for t in traits if f"{domain_val}_{t}" in self._obj.columns]

        if not trait_cols:
            # If none of the genes exist in the dataset, no isolate has them
            return pd.Series(False, index=self._obj.index)

        # Pandas matrix math: Check across the columns (axis=1) for any True values
        return self._obj[trait_cols].any(axis=1)

    def has_all(self, traits: list[str], domain: Union[str, Domain] = Domain.VIRULENCE) -> pd.Series:
        """
        Checks if isolates possess ALL of the specified traits.

        Args:
            traits: List of trait names.
            domain: Domain prefix (e.g., Domain.VIRULENCE). Defaults to Domain.VIRULENCE.

        Returns:
            A boolean Series.
        """
        domain_val = domain.value if isinstance(domain, Domain) else domain
        trait_cols = [f"{domain_val}_{t}" for t in traits]

        # If the user asks for a gene that isn't even in the dataset, they can't have 'all'
        missing_cols = set(trait_cols) - set(self._obj.columns)
        if missing_cols:
            return pd.Series(False, index=self._obj.index)

        # Matrix math: Check if ALL trait columns are True
        return self._obj[trait_cols].all(axis=1)

    def has_gene(self, gene_col: str, gene_name: str) -> pd.Series:
        """
        Searches for a specific gene within a comma-separated column.

        Args:
            gene_col: Column name containing gene lists.
            gene_name: The specific gene to find.

        Returns:
            A boolean Series.
        """
        # str.contains with na=False avoids allocating a new Series with fillna
        return self._obj[gene_col].str.contains(gene_name, regex=False, na=False)

    def sort_loci(self, locus_col: str) -> pd.DataFrame:
        """
        Sorts the DataFrame numerically by locus (e.g., K2 before K10).

        Args:
            locus_col: Column containing locus names.

        Returns:
            A sorted copy of the DataFrame.
        """
        df = self._obj.copy()
        # Extract the integer part of the locus (e.g., "K10" -> 10, "O2v1" -> 2)
        sort_key = df[locus_col].str.extract(r'(\d+)', expand=False).astype(float)
        return df.iloc[sort_key.sort_values().index]


@pd.api.extensions.register_dataframe_accessor("qc")
class QCAccessor:
    """
    Pandas accessor for quality control operations.

    Provides methods for filtering assemblies based on metrics like N50 and
    contig count.

    Examples:
        >>> import pandas as pd
        >>> import seroepi.accessors
        >>> df = pd.DataFrame({'qc_N50': [50000, 5000]})
        >>> clean_df = df.qc.filter_assemblies(min_n50=10000)
    """
    def __init__(self, pandas_obj: pd.DataFrame):
        self._obj = pandas_obj

    @property
    def metrics(self) -> pd.DataFrame:
        """Returns the QC metrics matrix with the prefix removed."""
        return self._obj.filter(regex=f"^{Domain.QC.value}_").rename(columns=lambda c: c.replace(f'{Domain.QC.value}_', '', 1))

    def filter_assemblies(self, min_n50: int = 10000, max_contigs: int = 500,
                          require_species: str = None) -> pd.DataFrame:
        """
        Filters genomes based on quality thresholds.

        Args:
            min_n50: Minimum N50 value. Defaults to 10000.
            max_contigs: Maximum number of contigs. Defaults to 500.
            require_species: Optional species name to filter for.

        Returns:
            A filtered copy of the DataFrame.
        """
        df = self._obj

        masks = []

        if f'{Domain.QC.value}_N50' in df.columns:
            # Coerce to numeric in case Kleborate spat out weird strings like '-'
            n50 = pd.to_numeric(df[f'{Domain.QC.value}_N50'], errors='coerce')
            masks.append((n50 >= min_n50) | n50.isna())

        if f'{Domain.QC.value}_contig_count' in df.columns:
            contigs = pd.to_numeric(df[f'{Domain.QC.value}_contig_count'], errors='coerce')
            masks.append((contigs <= max_contigs) | contigs.isna())

        if require_species and f'{Domain.QC.value}_species' in df.columns:
            masks.append(df[f'{Domain.QC.value}_species'].str.contains(require_species, case=False, na=False))
            
        if not masks:
            return df.copy()

        # Efficiently reduce the conditions using numpy logic
        final_mask = np.logical_and.reduce(masks)
        return df[final_mask].copy()

    def report(self) -> pd.Series:
        """
        Generates a summary report of dataset quality.

        Returns:
            A Series containing summary metrics (e.g., Total Warnings, Median N50).
        """
        metrics = self.metrics
        report = {}
        if 'QC_warnings' in metrics.columns:
            report['Total Warnings'] = (metrics['QC_warnings'] != '-').sum()
        if 'N50' in metrics.columns:
            report['Median N50'] = pd.to_numeric(metrics['N50'], errors='coerce').median()

        return pd.Series(report)