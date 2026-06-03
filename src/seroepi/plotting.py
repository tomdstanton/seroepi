from typing import Union, Any
from abc import ABC, abstractmethod
from json import load as json_load
from warnings import warn
from importlib.resources import files

import pandas as pd
import numpy as np
from scipy.stats import norm

import plotly.graph_objects as go
from plotly.graph_objects import Figure
from plotly.subplots import make_subplots

from seroepi.constants import PlotType, AggregationType, DistanceMetricType, Domain
from seroepi import estimators
from seroepi.formulation import Formulation
from seroepi.dist import DistancesBase


# Classes --------------------------------------------------------------------------------------------------------------
class BasePlotter(ABC):
    """
    Stateless base class for all plotting engines in seroepi.
    """
    # The Hero Palette: Electric Cyan and Neon Pink
    # Cyan is scientifically colorblind-safe while looking stunning on dark backgrounds.
    _MAIN_COLOUR = '#0EA5E9'
    # Translucent Cyan for Confidence Interval Ribbons (20% Opacity)
    _CI_COLOUR = 'rgba(14, 165, 233, 0.2)'
    # A secondary highlight color (Optional, but great for distinguishing target groups)
    _ACCENT_COLOUR = '#EC4899'  # Vibrant Neon Pink
    _FONT_COLOUR = '#94A3B8'  # Slate 400 (Highly readable on both light and dark backgrounds)
    _GRID_COLOUR = 'rgba(148, 163, 184, 0.2)'  # Subtle translucent grid lines
    # Global cache to prevent reading the file from disk multiple times
    _WORLD_GEOJSON = None
    
    # To be overridden by subclasses with the supported result types
    SUPPORTED_TYPES = ()

    @classmethod
    def _get_world_geojson(cls) -> dict:
        """
        Lazily loads and caches the internal world boundaries GeoJSON.

        Returns:
            A dictionary containing the GeoJSON data.
        """
        if cls._WORLD_GEOJSON is None:
            try:
                # Safely navigates the package structure regardless of where it's installed
                geojson_path = files('seroepi.data').joinpath('world_boundaries.geojson')
                with geojson_path.open(mode='r', encoding='utf-8') as f:
                    cls._WORLD_GEOJSON = json_load(f)
            except Exception as e:
                warn(f"Could not load internal world boundaries. Ensure the file exists: {e}")
                cls._WORLD_GEOJSON = {}
        return cls._WORLD_GEOJSON

    @classmethod
    def can_render(cls, result_obj: Any) -> bool:
        """Checks if the incoming result object is supported by this plotter."""
        # Safely extract the inner type if it's passed as a type hint or instance
        return isinstance(result_obj, cls.SUPPORTED_TYPES)

    @classmethod
    def _clean_label(cls, col_name: str) -> str:
        """Strips domain prefixes for clean UI rendering."""
        if not isinstance(col_name, str): return str(col_name)
        for domain in [Domain.GENOTYPE.value, Domain.PHENOTYPE.value, Domain.AMR.value, Domain.VIRULENCE.value]:
            prefix = f"{domain}_"
            if col_name.startswith(prefix):
                return col_name.replace(prefix, "").replace("_", " ")
        return col_name.replace("_", " ")

    @classmethod
    def get_colorscale(cls, transparent: bool = True) -> list:
        """Returns the standard Cyberpunk continuous color scale."""
        base_color = 'rgba(0,0,0,0)' if transparent else 'rgba(15, 23, 42, 0.4)'
        return [
            [0.0, base_color],
            [0.4, '#8B5CF6'],                # Deep Purple
            [0.7, cls._MAIN_COLOUR],         # Electric Cyan
            [1.0, cls._ACCENT_COLOUR]        # Neon Pink
        ]

    @classmethod
    def apply_theme(cls, fig: Figure) -> Figure:
        """Applies a universal transparent theme optimized for both light and dark web app modes."""
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color=cls._FONT_COLOUR),
            hoverlabel=dict(
                bgcolor="black",
                bordercolor=cls._MAIN_COLOUR,
                font_size=14,
                font_color="white"
            )
        )
        fig.update_xaxes(
            gridcolor=cls._GRID_COLOUR,
            zerolinecolor=cls._GRID_COLOUR,
            linecolor=cls._GRID_COLOUR
        )
        fig.update_yaxes(
            gridcolor=cls._GRID_COLOUR,
            zerolinecolor=cls._GRID_COLOUR,
            linecolor=cls._GRID_COLOUR
        )
        return fig

    @classmethod
    @abstractmethod
    def render(cls, result_obj: Any, **kwargs) -> 'Figure':
        """
        Renders the result object into a Plotly figure.

        Args:
            result_obj: The result object to visualize.
            **kwargs: Additional plotting arguments.

        Returns:
            A plotly Figure object.
        """
        pass


class CompositionBarPlotter(BasePlotter):
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', **kwargs) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        is_comp = result.aggregation_type == AggregationType.COMPOSITIONAL
        if not is_comp:
            raise ValueError("Composition Bar Plot strictly requires Compositional aggregation mode.")

        df = result.data.copy()
        target_col = 'target'
        group_cols = result.stratified_by
        strata_label = ', '.join(result.stratified_by) if result.stratified_by else "Global"
        title_prefix = "Sample Composition"

        fig = go.Figure()

        # We calculate the global rank of the targets to ensure the colors and stack
        # order remain perfectly consistent across all bars
        target_ranks = df.groupby(target_col)['estimate'].sum().sort_values(ascending=False).index

        # OPTIMIZATION: Pre-group the dataframe to prevent O(N^2) dataframe masking loops
        grouped_df = dict(tuple(df.groupby(target_col)))

        if not group_cols:
            # --- 1 VARIABLE: Global Composition ---
            for t in target_ranks:
                if (t_df := grouped_df.get(t)) is None or t_df.empty:
                    continue
                fig.add_trace(go.Bar(
                    x=["Global Formulation"],
                    y=t_df['estimate'],
                    name=str(t),
                    hovertemplate=f"<b>{cls._clean_label(t)}</b><br>Prevalence: %{{y:.1%}}<extra></extra>"
                ))
        else:
            # --- 2 VARIABLES: Grouped Composition ---
            group_col = group_cols[0]
            for t in target_ranks:
                if (t_df := grouped_df.get(t)) is None or t_df.empty:
                    continue
                fig.add_trace(go.Bar(
                    x=t_df[group_col],
                    y=t_df['estimate'],
                    name=str(t),
                    hovertemplate=f"<b>{cls._clean_label(t)}</b><br>{cls._clean_label(group_col)}: %{{x}}<br>Prevalence: %{{y:.1%}}<extra></extra>"
                ))

        # Apply the global theme and force the stacking mode
        return cls.apply_theme(fig.update_layout(
            barmode='stack',
            title=f"<b>{title_prefix}</b><br><sup>Stratified by {strata_label}</sup>",
            yaxis_title="Cumulative Prevalence",
            yaxis=dict(tickformat='.0%')  # Renders 0.8 as 80%
        ))


class CompositionHeatmapPlotter(BasePlotter):
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', **kwargs) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        df = result.data.copy()
        is_comp = result.aggregation_type == AggregationType.COMPOSITIONAL

        if not is_comp:
            # Trait Heatmap
            if len(result.stratified_by) != 2:
                raise ValueError(
                    f"Trait Heatmap strictly requires exactly 2 stratification variables. "
                    f"Found: {result.stratified_by}"
                )
            y_col = result.stratified_by[0]
            x_col = result.stratified_by[1]
            title_prefix = "Prevalence Matrix"
        else:
            y_col = 'target'
            group_cols = result.stratified_by
            if len(group_cols) != 1:
                raise ValueError(
                    f"Composition Heatmap strictly requires exactly 1 grouping variable alongside the target. "
                    f"Found target '{result.trait}' and groups: {group_cols}"
                )
            x_col = group_cols[0]
            title_prefix = "Density Matrix"

        # 2. Reshape from Long to Wide (The Z-Matrix)
        # fill_value=0 ensures that if a K-locus isn't found in a country, it shows as empty, not NaN
        pivot_df = df.pivot_table(
            index=y_col,
            columns=x_col,
            values='estimate',
            fill_value=0
        )

        # 3. Sort the Y-Axis so the most globally prevalent targets sit at the top of the heatmap
        pivot_df['total_burden'] = pivot_df.sum(axis=1)
        pivot_df = pivot_df.sort_values('total_burden', ascending=True).drop(columns=['total_burden'])

        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale=cls.get_colorscale(transparent=True),
            showscale=True,  # Shows the colorbar legend on the right
            colorbar=dict(tickformat='.0%'),
            xgap=1, ygap=1,  # Physically separate the cells into distinct tiles
            hovertemplate=f"<b>%{{y}}</b><br>{x_col}: %{{x}}<br>Prevalence: %{{z:.1%}}<extra></extra>"
        ))

        return cls.apply_theme(fig.update_layout(
            title=f"<b>{title_prefix}</b><br><sup>{y_col} vs {x_col}</sup>",
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title()
        ))


class ForestPlotter(BasePlotter):
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', sort_by: str = 'estimate', **kwargs) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        df = result.data.copy()
        is_comp = result.aggregation_type == AggregationType.COMPOSITIONAL

        if is_comp:
            y_col = 'target'
            group_cols = result.stratified_by
            color_col = group_cols[0] if group_cols else None
            
            # --- CLEARNESS FIX FOR COMPOSITIONAL DATA ---
            # 1. Truncate to top N variants to prevent vertical overcrowding
            top_n = kwargs.get('top_n', 20)
            target_totals = df.groupby(y_col)['estimate'].sum().sort_values(ascending=False)
            
            is_truncated = len(target_totals) > top_n
            if is_truncated:
                top_targets = target_totals.head(top_n).index
                
            # 2. Create an ordered array for the Y-axis 
            y_order = target_totals.index.tolist()
            df[y_col] = pd.Categorical(df[y_col], categories=y_order, ordered=True)
            df = df.sort_values([color_col, y_col]) if color_col else df.sort_values(y_col)
            
        else:
            y_col = result.stratified_by[0] if result.stratified_by else 'target'
            color_col = result.stratified_by[1] if len(result.stratified_by) > 1 else None
            is_truncated = False

        # Sort the dataframe so the highest prevalences appear at the top
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=False)
            
        y_order = df[y_col].drop_duplicates().tolist()

        fig = go.Figure()

        if color_col:
            # --- MULTI-VARIABLE: Grouped Forest Plot ---
            for group_name, group_df in df.groupby(color_col):
                fig.add_trace(go.Scatter(
                    x=group_df['estimate'],
                    y=group_df[y_col],
                    name=str(group_name),
                    mode='markers',
                    marker=dict(size=10, symbol='square'),
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=group_df['upper'] - group_df['estimate'],
                        arrayminus=group_df['estimate'] - group_df['lower'],
                        width=0,  # Removes the vertical caps for a cleaner "Tufte" look
                        thickness=2
                    ),
                    customdata=group_df[[color_col, 'lower', 'upper']].values,
                    hovertemplate=(
                        f"<b>{cls._clean_label(y_col)}:</b> %{{y}}<br>"
                        f"<b>{cls._clean_label(color_col)}:</b> %{{customdata[0]}}<br>"
                        "<b>Prevalence:</b> %{x:.1%}<br>"
                        "<b>95% CI:</b> %{customdata[1]:.1%} - %{customdata[2]:.1%}<extra></extra>"
                    )
                ))

            # THE MAGIC FIX: This prevents categorical points from overlapping!
            fig.update_layout(scattermode='group')

        else:
            # --- SINGLE-VARIABLE: Standard Forest Plot ---
            fig.add_trace(go.Scatter(
                x=df['estimate'],
                y=df[y_col],
                mode='markers',
                marker=dict(size=10, color=cls._MAIN_COLOUR, symbol='square'),
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=df['upper'] - df['estimate'],
                    arrayminus=df['estimate'] - df['lower'],
                    color=cls._CI_COLOUR,
                    width=0, thickness=2
                ),
                customdata=df[['lower', 'upper']].values,
                hovertemplate=(
                    f"<b>{cls._clean_label(y_col)}:</b> %{{y}}<br>"
                    "<b>Prevalence:</b> %{x:.1%}<br>"
                    "<b>95% CI:</b> %{customdata[0]:.1%} - %{customdata[1]:.1%}<extra></extra>"
                )
            ))
            
        title_prefix = "Prevalence Estimates"
        if is_truncated:
            title_prefix += f" (Top {top_n})"
            
        yaxis_kwargs = dict(
            autorange='reversed', 
            title=cls._clean_label(y_col),
            type='category',
            categoryorder='array',
            categoryarray=y_order
        )

        return cls.apply_theme(fig.update_layout(
            title=f"<b>{title_prefix}</b><br><sup>Trait: {cls._clean_label(result.trait)} | Stratified by {', '.join(result.stratified_by)}</sup>",
            xaxis_title="Prevalence (%)",
            yaxis=yaxis_kwargs,
            xaxis=dict(tickformat='.0%')  # Clean percentage formatting on the X-axis
        ))


class EpicurvePlotter(BasePlotter):
    SUPPORTED_TYPES = (estimators.IncidenceEstimates,)

    @classmethod
    def render(cls, result: 'estimators.IncidenceEstimates', **kwargs):
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        """
        Plots a classic Epidemic Curve (Cases over Time).
        Displays total sequencing volume in the background to visually address sampling bias.
        """
        data = result.data.copy().sort_values(by='date')

        fig = go.Figure()

        # Background Area: Total Sequenced Volume
        # Plotted first so it sits behind the bars
        fig.add_trace(go.Scatter(
            x=data['date'], y=data['total_sequenced'],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(51, 65, 85, 0.3)',  # Translucent Slate
            line=dict(color='#475569', width=1, dash='dot'),
            name='Total Sequenced Volume',
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Total Sequenced</b>: %{y}<extra></extra>"
        ))

        # Foreground Bar: Variant Counts
        fig.add_trace(go.Bar(
            x=data['date'], y=data['variant_count'],
            marker_color=cls._MAIN_COLOUR,
            name=f"{cls._clean_label(result.trait)} Cases",
            hovertemplate="<b>Date</b>: %{x|%Y-%m-%d}<br><b>Cases</b>: %{y}<extra></extra>"
        ))

        # Intelligently extract the Incidence Rate Ratio (IRR) to display in the subtitle
        subtitle = ""
        if not result.model_results.empty and 'IRR' in result.model_results.columns:
            irr = result.model_results['IRR'].iloc[0]
            if pd.notna(irr):
                direction = "Increasing" if irr > 1 else "Decreasing"
                subtitle = f"<br><sup>Trend: {direction} (IRR: {irr:.2f} per time step)</sup>"

        return cls.apply_theme(fig.update_layout(
            title=f"<b>Epidemic Curve: {cls._clean_label(result.trait)}</b>{subtitle}",
            xaxis=dict(title="Date", type='date'),
            yaxis=dict(title='Count'),
            barmode='overlay',  # Allows the bars and the area chart to overlap safely
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"
            )
        ))


class LongitudinalPrevalencePlotter(BasePlotter):
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', **kwargs):
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        """
        Plots a time-series line chart for longitudinal prevalence data.
        """
        # Intelligently check if a date/time column exists in the strata
        time_cols = [col for col in result.stratified_by if
                     'date' in col.lower() or 'year' in col.lower() or 'month' in col.lower()]

        if not time_cols:
            raise ValueError(
                "No temporal column detected in the strata. Ensure you stratified by a Date, Year, or Month column.")

        time_col = time_cols[0]
        data = result.data.sort_values(by=time_col)

        fig = go.Figure()

        # Add the translucent confidence band
        fig.add_trace(go.Scattergl(
            x=pd.concat([data[time_col], data[time_col][::-1]]),
            y=pd.concat([data['upper'], data['lower'][::-1]]),
            fill='toself',
            fillcolor=cls._CI_COLOUR,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            legendgroup="Prevalence"
        ))

        # Add the main trend line with neon ring markers
        fig.add_trace(go.Scatter(
            x=data[time_col], y=data['estimate'],
            mode='lines+markers',
            line=dict(color=cls._MAIN_COLOUR, width=3),
            marker=dict(size=8, color=cls._MAIN_COLOUR),
            name="Prevalence",
            hovertemplate="<b>Date</b>: %{x}<br><b>Prevalence</b>: %{y:.2%}<extra></extra>",
            legendgroup="Prevalence"
        ))

        return cls.apply_theme(fig.update_layout(
            title=f"<b>Longitudinal Prevalence of {cls._clean_label(result.trait)}</b>",
            xaxis=dict(title=time_col.title()),
            yaxis=dict(title='Prevalence', tickformat='.0%', range=[0, 1.05])
        ))


class CumulativeCoveragePlotter(BasePlotter):
    """
    Calculates cumulative population coverage.
    Crucial for designing multivalent vaccines (e.g., K-locus targeting).
    """
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates, dict)

    @classmethod
    def render(cls, result: Union['estimators.PrevalenceEstimates', dict],  max_valencies: int = None, **kwargs):
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")

        if isinstance(result, dict):
            res = result.get("res")
            formulation = result.get("formulation")
        else:
            res = result
            formulation = None

        if res.aggregation_type != AggregationType.COMPOSITIONAL:
            raise ValueError("Cumulative coverage strictly requires compositional prevalence estimates.")

        data = res.data.copy()
        
        # Idiomatic SciPy: Retrieve the exact Z-score for a 95% two-sided interval (~1.96)
        z_score = norm.ppf(0.975)
        
        # Extract SE from the existing CIs to mathematically preserve the complex 
        # shrinkage/smoothing applied by the upstream Bayesian or Spatial estimators!
        data['se'] = (data['upper'] - data['lower']) / (2 * z_score)
        data['var'] = data['se'] ** 2

        if formulation:
            target_order = formulation.get_formulation()
        else:
            # Sort strictly by raw count to simulate prioritizing the most common variants globally
            target_order = data.groupby('target')['event'].sum().sort_values(ascending=False).index.tolist()
            if max_valencies:
                target_order = target_order[:max_valencies]

        fig = go.Figure()
        group_cols = res.stratified_by

        import plotly.express as px
        colors = px.colors.qualitative.Plotly

        if not group_cols:
            # --- GLOBAL COVERAGE ---
            grouped = data.groupby('target', observed=True)['estimate'].sum().reindex(target_order).fillna(0)
            grouped_var = data.groupby('target', observed=True)['var'].sum().reindex(target_order).fillna(0)
            
            cum_prop = grouped.cumsum().clip(0, 1)
            cum_se = np.sqrt(grouped_var.cumsum())
            
            cum_lower = (cum_prop - z_score * cum_se).clip(0, 1)
            cum_upper = (cum_prop + z_score * cum_se).clip(0, 1)
            
            # Draw the translucent confidence ribbon
            fig.add_trace(go.Scatter(
                x=target_order + target_order[::-1],
                y=cum_upper.tolist() + cum_lower.tolist()[::-1],
                fill='toself',
                fillcolor=cls._MAIN_COLOUR,
                opacity=0.2,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=False,
                legendgroup="Cumulative Population Coverage"
            ))

            fig.add_trace(go.Scatter(
                x=target_order, y=cum_prop,
                mode='lines+markers',
                name='Cumulative Population Coverage',
                line=dict(color=cls._MAIN_COLOUR, width=3),
                marker=dict(size=8, color=cls._MAIN_COLOUR),
                customdata=np.column_stack((cum_lower.values, cum_upper.values)),
                hovertemplate="<b>%{x}</b><br>Cumulative Coverage: %{y:.1%}<br>95% CI: %{customdata[0]:.1%} - %{customdata[1]:.1%}<extra></extra>",
                legendgroup="Cumulative Population Coverage"
            ))
            strata_label = "Baseline"
        else:
            # --- STRATIFIED COVERAGE ---
            color_col = group_cols[0]
            strata_label = f"Stratified by {cls._clean_label(color_col)}"
            
            for i, (stratum, group_df) in enumerate(data.groupby(color_col, observed=True)):
                grouped = group_df.groupby('target', observed=True)['estimate'].sum().reindex(target_order).fillna(0)
                grouped_var = group_df.groupby('target', observed=True)['var'].sum().reindex(target_order).fillna(0)
                
                cum_prop = grouped.cumsum().clip(0, 1)
                cum_se = np.sqrt(grouped_var.cumsum())
                
                cum_lower = (cum_prop - z_score * cum_se).clip(0, 1)
                cum_upper = (cum_prop + z_score * cum_se).clip(0, 1)
                
                color = colors[i % len(colors)]
                
                # Draw the translucent confidence ribbon
                fig.add_trace(go.Scatter(
                    x=target_order + target_order[::-1],
                    y=cum_upper.tolist() + cum_lower.tolist()[::-1],
                    fill='toself',
                    fillcolor=color,
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=str(stratum)
                ))
                
                fig.add_trace(go.Scatter(
                    x=target_order, y=cum_prop,
                    mode='lines+markers',
                    name=str(stratum),
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color),
                    customdata=np.column_stack((cum_lower.values, cum_upper.values)),
                    hovertemplate=f"<b>%{{x}}</b><br>{cls._clean_label(color_col)}: {stratum}<br>Cumulative Coverage: %{{y:.1%}}<br>95% CI: %{{customdata[0]:.1%}} - %{{customdata[1]:.1%}}<extra></extra>",
                    legendgroup=str(stratum)
                ))

        return cls.apply_theme(fig.update_layout(
            title=f"<b>Cumulative Coverage</b><br><sup>Targeting top {len(target_order)} {cls._clean_label(res.trait)} variants | {strata_label}</sup>",
            xaxis=dict(title="Variant added to formulation", tickangle=45),
            yaxis=dict(title='Cumulative Population Coverage', tickformat='.0%', range=[0, 1.05]),
            hovermode="x unified"
        ))


class ChoroplethPlotter(BasePlotter):
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', geo_col: str = None, target_variant: str = None,
               feature_id_key: str = "properties.ADMIN"):
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        """
        Plots discrete regional prevalence on a map.
        """
        data = result.data.copy()

        if geo_col is None:
            # Auto-detect spatial column from the active strata
            spatial_candidates = [c for c in result.stratified_by if c.startswith(f"{Domain.SPATIAL.value}_")]
            if spatial_candidates:
                geo_col = spatial_candidates[0]
            elif len(result.stratified_by) == 1:
                geo_col = result.stratified_by[0]  # Fallback to the only available stratification
            else:
                raise ValueError("Cannot plot choropleth: Please stratify by a spatial column.")
        elif geo_col not in result.stratified_by:
            raise ValueError(f"Cannot plot choropleth: '{geo_col}' was not used as a stratification variable.")

        # --- THE FIX: Isolate the specific variant for compositional data ---
        target_name = result.trait
        if result.aggregation_type == AggregationType.COMPOSITIONAL:
            if target_variant is None:
                # Safely default to the most frequent variant if the user forgot to specify one
                target_variant = data.groupby('target')['event'].sum().idxmax()

            data = data[data['target'] == target_variant]
            target_name = f"{cls._clean_label(result.trait)}: {target_variant}"

        # Lazy-load the default map if none supplied
        geojson_data = cls._get_world_geojson()

        hover_text = (
                f"<b>{geo_col}</b>: " + data[geo_col].astype(str) +
                "<br><b>Prevalence</b>: " + data['estimate'].map('{:.2%}'.format) +
                "<br><b>95% CI</b>: [" + data['lower'].map('{:.2%}'.format) + ", " + data['upper'].map(
            '{:.2%}'.format) + "]"
        )

        fig = go.Figure(go.Choroplethmapbox(
            geojson=geojson_data,
            locations=data[geo_col],
            featureidkey=feature_id_key,
            z=data['estimate'],
            colorscale=cls.get_colorscale(transparent=False),
            zmin=0, zmax=1,
            marker_opacity=0.8,
            marker_line_width=1.5,
            marker_line_color='white',  # Clean white borders like Leaflet
            colorbar_title='Prevalence',
            text=hover_text,
            hoverinfo='text'
        ))

        title_prefix = "Regional Composition of" if result.aggregation_type == AggregationType.COMPOSITIONAL else "Regional Prevalence of"

        return cls.apply_theme(fig.update_layout(
            # Dynamically updates the title so the user knows exactly what is mapped!
            title=f"<b>{title_prefix} {target_name}</b>",
            mapbox_style="carto-positron",  # Clean, light map similar to CartoDB.Voyager
            mapbox_zoom=1,
            mapbox_center={"lat": 0, "lon": 0},
            margin={"r": 0, "t": 60, "l": 0, "b": 0}
        ))


class SpatialSurfacePlotter(BasePlotter):
    SUPPORTED_TYPES = (estimators.PrevalenceEstimates,)

    @classmethod
    def render(cls, result: 'estimators.PrevalenceEstimates', lat_col: str = 'lat', lon_col: str = 'lon'):
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        """
        Plots a continuous heatmap surface.
        Designed specifically for the dense grid output of the SpatialPrevalenceEstimator.
        """
        data = result.data.copy()

        if lat_col not in data.columns or lon_col not in data.columns:
            raise ValueError(f"Spatial surface requires '{lat_col}' and '{lon_col}' columns in the data.")

        hover_text = (
            "<b>Lat</b>: " + data[lat_col].round(3).astype(str) +
            " | <b>Lon</b>: " + data[lon_col].round(3).astype(str) +
            "<br><b>Predicted Prevalence</b>: " + data['estimate'].map('{:.2%}'.format)
        )

        fig = go.Figure(go.Densitymapbox(
            lat=data[lat_col],
            lon=data[lon_col],
            z=data['estimate'],
            radius=25,  # Size of the spatial smoothing kernel
            colorscale=cls.get_colorscale(transparent=True),
            zmin=0, zmax=1,
            opacity=0.85,
            text=hover_text,
            hoverinfo='text',
            colorbar_title='Predicted<br>Prevalence'
        ))

        title_prefix = "Spatial Composition of" if result.aggregation_type == AggregationType.COMPOSITIONAL else "Spatial Surface for"

        return cls.apply_theme(fig.update_layout(
            title=f"<b>{title_prefix} {cls._clean_label(result.trait)}</b><br><sup>Gaussian Process Prediction Surface</sup>",
            mapbox_style="carto-positron", # Clean light canvas
            mapbox_zoom=3,
            mapbox_center={"lat": data[lat_col].mean(), "lon": data[lon_col].mean()},
            margin={"r": 0, "t": 60, "l": 0, "b": 0}
        ))


class AlphaDiversityPlotter(BasePlotter):
    SUPPORTED_TYPES = (estimators.AlphaDiversityEstimates,)

    @classmethod
    def render(cls, result: 'estimators.AlphaDiversityEstimates',  metric: str = 'shannon',
                    sort_ascending: bool = False) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        """
        Plots a glowing lollipop chart for within-group diversity.
        """
        if metric not in result.metrics:
            raise ValueError(f"Metric '{metric}' not found. Available metrics: {result.metrics}")

        data = result.data.copy()

        # Identify the primary stratification column
        group_col = result.stratified_by[0] if result.stratified_by else 'Global'

        if sort_ascending is not None:
            data = data.sort_values(by=metric, ascending=sort_ascending)

        fig = go.Figure()

        # 1. The Stems (Thin glowing lines)
        # We use go.Bar with an ultra-thin width to create perfect vertical lines down to 0
        fig.add_trace(go.Bar(
            x=data[group_col],
            y=data[metric],
            width=0.02,  # Ultra-thin stem
            marker_color='rgba(14, 165, 233, 0.4)',  # Translucent Cyan
            hoverinfo='skip',
            showlegend=False
        ))

        # 2. The Lollipops (Neon markers)
        fig.add_trace(go.Scatter(
            x=data[group_col],
            y=data[metric],
            mode='markers',
            marker=dict(
                size=12,
                color=cls._MAIN_COLOUR
            ),
            name=metric.title(),
            hovertemplate=f"<b>%{{x}}</b><br>{metric.title()}: %{{y:.3f}}<br>Sequenced (n): %{{customdata}}<extra></extra>",
            customdata=data['n_samples']
        ))

        return cls.apply_theme(fig.update_layout(
            title=f"<b>Alpha Diversity ({metric.title()})</b><br><sup>Trait: {cls._clean_label(result.trait)}</sup>",
            xaxis=dict(title=group_col.title() if group_col != 'Global' else ''),
            yaxis=dict(title=f'{metric.title()} Index', rangemode='tozero'),  # Stems must anchor to 0
            showlegend=False
        ))


class BetaHeatmapPlotter(BasePlotter):
    SUPPORTED_TYPES = (estimators.BetaDiversityEstimates,)

    @classmethod
    def render(cls, result: 'estimators.BetaDiversityEstimates', mask_upper: bool = True) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        """
        Plots an N x N dissimilarity matrix.
        If mask_upper is True, perfectly formats it as an academic lower-triangle heatmap.
        """
        dist_matrix = result.data.copy()

        # The Pro-Move: Mask the upper triangle and the diagonal
        if mask_upper:
            # np.triu returns the upper triangle. We set those coordinates to NaN.
            mask = np.triu(np.ones(dist_matrix.shape, dtype=bool))
            dist_matrix = dist_matrix.mask(mask)

        # Hover text configuration
        labels = dist_matrix.columns.tolist()

        fig = go.Figure(data=go.Heatmap(
            z=dist_matrix.values,
            x=labels,
            y=labels,
            colorscale=cls.get_colorscale(transparent=True),
            zmin=0,
            zmax=1,  # Bray-Curtis and Jaccard are naturally bounded 0-1
            xgap=1, ygap=1,  # Physically separate the cells into distinct tiles
            hoverongaps=False,  # Prevents hovering over the NaN upper triangle
            hovertemplate="<b>Group 1:</b> %{y}<br><b>Group 2:</b> %{x}<br><b>Dissimilarity:</b> %{z:.3f}<extra></extra>",
            colorbar=dict(title=f"{result.metric.title()}<br>Distance")
        ))

        return cls.apply_theme(fig.update_layout(
            title=f"<b>Beta Diversity ({result.metric.title()})</b><br><sup>Trait: {cls._clean_label(result.trait)}</sup>",
            xaxis=dict(title='', tickangle=45),
            yaxis=dict(title='', autorange='reversed'),  # Reversing Y puts the triangle in the correct academic orientation
            margin=dict(l=20, r=20, t=60, b=80),  # Extra bottom margin for angled text
            width=700,
            height=700  # Force a square aspect ratio so the matrix cells are perfect squares
        ))


class StabilityBumpPlotter(BasePlotter):
    SUPPORTED_TYPES = (Formulation,)

    @classmethod
    def render(cls, formulation: 'Formulation', **kwargs) -> go.Figure:
        if not cls.can_render(formulation):
            raise TypeError(f"{cls.__name__} does not support {type(formulation).__name__}.")
        """
        Plots a Slopegraph (Bump Chart) showing how target ranks shift
        when different data groups are held out.
        """
        history = formulation.permutation_history.copy()
        baseline = formulation.rankings.copy()
        valency = formulation.max_valency

        fig = go.Figure()

        # We only want to plot the lines for targets that actually made it into the baseline _formulation
        top_targets = baseline.head(valency)['target'].tolist()

        # X-axis categories: Baseline first, then all the holdout permutations
        x_categories = ['Baseline'] + history['holdout_group'].unique().tolist()

        # OPTIMIZATION: Pivot the history table to convert O(N^2) boolean loops into O(1) lookups
        pivot_hist = history.pivot(index='target', columns='holdout_group', values='loo_rank')
        baseline_ranks = baseline.set_index('target')['baseline_rank']

        for target in top_targets:
            # Build the Y-coordinates (Ranks) mapping to the X-coordinates
            y_ranks = [baseline_ranks.get(target)]
            for group in x_categories[1:]:
                # Safely extract the pre-calculated rank directly from the 2D matrix
                if target in pivot_hist.index and group in pivot_hist.columns:
                    rank = pivot_hist.at[target, group]
                    y_ranks.append(rank if pd.notna(rank) else None)
                else:
                    y_ranks.append(None)

            fig.add_trace(go.Scatter(
                x=x_categories,
                y=y_ranks,
                mode='lines+markers',
                name=str(target),
                line=dict(width=2),  # Thinner lines to match coverage plots
                marker=dict(size=8),
                hovertemplate="<b>Holdout: %{x}</b><br>Rank: %{y}<extra></extra>"
            ))

        # Draw a subtle horizontal line representing the "Valency Cutoff"
        fig.add_hline(
            y=valency + 0.5,
            line_dash="dot",
            line_color="#EF4444",  # Red warning line
            annotation_text="Valency Cutoff",
            annotation_position="bottom right"
        )

        return cls.apply_theme(fig.update_layout(
            title=f"<b>Trait Priority Stability (LOO)</b><br><sup>Targeting Top {valency} {cls._clean_label(formulation.trait)} Variants</sup>",
            hovermode="x unified"
        )).update_xaxes(
            title="Excluded Group (Leave-One-Out)"
        ).update_yaxes(
            title="Priority Rank",
            autorange='reversed',  # Crucial: Rank 1 must be at the TOP of the Y-axis!
            dtick=1
        )


class NetworkPlotter(BasePlotter):
    SUPPORTED_TYPES = (DistancesBase,)

    @staticmethod
    def _build_edges(rows: np.ndarray, cols: np.ndarray, pos: np.ndarray) -> tuple[list, list]:
        """Helper to cleanly vectorize the line-break generation for Plotly networks."""
        if len(rows) == 0:
            return [], []
        ex = np.full(len(rows) * 3, None, dtype=object)
        ex[0::3], ex[1::3] = pos[rows, 0], pos[cols, 0]
        
        ey = np.full(len(rows) * 3, None, dtype=object)
        ey[0::3], ey[1::3] = pos[rows, 1], pos[cols, 1]
        return ex.tolist(), ey.tolist()

    @classmethod
    def render(cls, result: DistancesBase, df: pd.DataFrame = None, pos: np.ndarray = None,
               edge_type: str = 'snp', threshold: int = 20,
               color_col: str = None, trans_network: DistancesBase = None, **kwargs) -> go.Figure:
        if not cls.can_render(result):
            raise TypeError(f"{cls.__name__} does not support {type(result).__name__}.")
        """
        Plots an interactive force-directed network using MDS coordinates.
        """
        dense_dist = result.matrix.toarray().astype(float)

        # Fallback to calculate MDS if it wasn't passed in via a cache
        if pos is None:
            pos = result.layout()

        # Align the dataframe rows to perfectly match the distance matrix index
        if df is not None and 'sample_id' in df.columns:
            df_aligned = df.set_index('sample_id').reindex(result.index)
        else:
            df_aligned = pd.DataFrame(index=result.index)

        edge_x, edge_y = [], []
        title = "<b>Isolate Network</b><br><sup>Nodes positioned by distance layout (MDS)</sup>"

        if edge_type == "snp" and getattr(result, 'metric_type', None) in [DistanceMetricType.ABSOLUTE_DISTANCE, DistanceMetricType.RELATIVE_DISTANCE]:
            # OPTIMIZATION: Use numpy's upper triangle (k=1) to prevent drawing self-loops or duplicate edges
            adj = (dense_dist <= threshold)
            rows, cols = np.where(np.triu(adj, k=1))

            edge_x, edge_y = cls._build_edges(rows, cols, pos)

            title = f"<b>Genomic SNP Network</b><br><sup>Edges connect isolates ≤ {threshold} SNPs apart</sup>"

        elif edge_type == "trans":
            net = trans_network if trans_network is not None else result
            if getattr(net, 'metric_type', None) in [DistanceMetricType.ABSOLUTE_SIMILARITY, DistanceMetricType.RELATIVE_SIMILARITY]:
                from scipy.sparse import triu
                # OPTIMIZATION: Query the sparse matrix directly without blowing it up into a dense memory hog
                upper_adj = triu(net.matrix, k=1).tocoo()
                edge_x, edge_y = cls._build_edges(upper_adj.row, upper_adj.col, pos)

                title = f"<b>Transmission Network</b><br><sup>Edges connect isolates based on spatial/temporal proximity</sup>"

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.4, color='#888'), hoverinfo='none', mode='lines')

        if color_col and color_col in df_aligned.columns:
            color_series = df_aligned[color_col]
            # Safely add "Unknown" to the approved categories to maintain memory efficiency
            if isinstance(color_series.dtype, pd.CategoricalDtype) and "Unknown" not in color_series.cat.categories:
                color_series = color_series.cat.add_categories("Unknown")
                
            color_vals = color_series.fillna("Unknown").astype(str)
            color_map = {c: i for i, c in enumerate(color_vals.unique())}
            colors = [color_map[c] for c in color_vals]
            hover_text = [f"ID: {idx}<br>{color_col}: {c}" for idx, c in zip(result.index, color_vals)]
            marker_dict = dict(showscale=False, color=colors, colorscale='Turbo', size=10, line=dict(width=1, color='white'))
        else:
            hover_text = [f"ID: {idx}" for idx in result.index]
            marker_dict = dict(color=cls._MAIN_COLOUR, size=10, line=dict(width=1, color='white'))

        node_trace = go.Scatter(
            x=pos[:, 0], y=pos[:, 1], mode='markers',
            hovertext=hover_text, hoverinfo="text", marker=marker_dict
        )

        return cls.apply_theme(go.Figure(data=[edge_trace, node_trace]).update_layout(
            title=title,
            showlegend=False, hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(b=20, l=5, r=5, t=60)
        ))


class LongevityPlotter(BasePlotter):
    SUPPORTED_TYPES = (Formulation,)

    @classmethod
    def render(cls, formulation: 'Formulation', **kwargs) -> go.Figure:
        if not cls.can_render(formulation):
            raise TypeError(f"{cls.__name__} does not support {type(formulation).__name__}.")
        """
        Renders a dual-axis area chart displaying the absolute projected burden
        and the proportional coverage of a vaccine formulation over time.
        """
        forecast = kwargs.get('forecast')
        if forecast is None:
            raise ValueError("LongevityPlotter requires a 'forecast' (IncidenceEstimates) in kwargs.")
            
        df = formulation.evaluate_longevity(forecast)

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 1. Total Burden (The Threat)
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['total_cases'],
                name='Total Projected Cases',
                mode='lines',
                line=dict(color=cls._FONT_COLOUR, width=0),
                fill='tozeroy',
                fillcolor='rgba(71, 85, 105, 0.4)',  # Semi-transparent slate
                hovertemplate='%{y:.1f} Total Cases<extra></extra>'
            ),
            secondary_y=False,
        )

        # 2. Covered Burden (The Protection)
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['covered_cases'],
                name='Covered by Vaccine',
                mode='lines',
                line=dict(color=cls._MAIN_COLOUR, width=2),
                fill='tozeroy',
                fillcolor=cls._CI_COLOUR,  # Translucent Cyan
                hovertemplate='%{y:.1f} Covered<extra></extra>'
            ),
            secondary_y=False,
        )

        # 3. Coverage Percentage (The Executive Metric)
        fig.add_trace(
            go.Scatter(
                x=df['date'],
                y=df['coverage_pct'],
                name='Coverage (%)',
                mode='lines+markers',
                line=dict(color=cls._ACCENT_COLOUR, width=2, dash='dot'),
                marker=dict(size=6, symbol='circle'),
                hovertemplate='%{y:.1f}% Coverage<extra></extra>'
            ),
            secondary_y=True,
        )

        # 4. The "Danger Zone" Threshold Line
        fig.add_hline(
            y=70,
            secondary_y=True,
            line_dash="dash",
            line_color="#EF4444",  # Red-500
            annotation_text="70% Efficacy Target",
            annotation_position="bottom right",
            annotation_font=dict(color="#EF4444")
        )

        # Axis Formatting
        fig.update_yaxes(title_text="Absolute Case Burden", rangemode='tozero', secondary_y=False)
        fig.update_yaxes(
            title_text="Coverage (%)",
            range=[0, 105],  # Lock the percentage scale
            showgrid=False,  # Prevent gridline clashes with the primary y-axis
            secondary_y=True
        )

        return cls.apply_theme(fig.update_layout(
            title=f"<b>Longevity Forecast</b><br><sup>Vaccine Trait: {cls._clean_label(formulation.trait)}</sup>",
            hovermode="x unified",  # Combines all data points into one clean hover tooltip
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0)'  # Transparent legend background
            ),
            margin=dict(t=60, b=40, l=40, r=40)
        ))


# Router ---------------------------------------------------------------------------------------------------------------
_PLOTTER_MAP = {
    PlotType.COMPOSITION_BAR: CompositionBarPlotter,
    PlotType.COMPOSITION_HEATMAP: CompositionHeatmapPlotter,
    PlotType.FOREST: ForestPlotter,
    PlotType.EPICURVE: EpicurvePlotter,
    PlotType.LONGITUDINAL_PREVALENCE: LongitudinalPrevalencePlotter,
    PlotType.CUMULATIVE_COVERAGE: CumulativeCoveragePlotter,
    PlotType.CHOROPLETH: ChoroplethPlotter,
    PlotType.SPATIAL_SURFACE: SpatialSurfacePlotter,
    PlotType.ALPHA_DIVERSITY: AlphaDiversityPlotter,
    PlotType.BETA_HEATMAP: BetaHeatmapPlotter,
    PlotType.STABILITY_BUMP: StabilityBumpPlotter,
    PlotType.NETWORK: NetworkPlotter,
    PlotType.LONGEVITY: LongevityPlotter,
}

def render_plot(result_obj: Any, plot_type: PlotType, **kwargs) -> go.Figure:
    """
    A central router that invokes the correct plotter for the desired plot type.
    """
    if plotter := _PLOTTER_MAP.get(plot_type, None):
        return plotter.render(result_obj, **kwargs)
    available = list(_PLOTTER_MAP.keys())
    raise ValueError(f"Plot type '{plot_type}' is not registered. Available: {available}")
